"""
ARN v9 REST API Server
========================
Production-ready FastAPI wrapper that turns ARN into a service.

Endpoints:
    POST /v1/memory/store     — Store a new memory
    POST /v1/memory/recall    — Recall relevant memories
    POST /v1/memory/context   — Get formatted context window
    POST /v1/memory/maintain  — Run consolidation
    GET  /v1/memory/stats     — Get system statistics
    GET  /v1/health           — Health check
    DELETE /v1/memory/agent   — Delete all agent data

Multi-tenancy:
    Each agent_id gets isolated storage. No cross-agent data leakage.
    Optional API key auth via X-API-Key header.

Deployment:
    # Local/Pi:
    uvicorn arn.api.server:app --host 0.0.0.0 --port 8742

    # Docker:
    docker run -p 8742:8742 -v arn_data:/data arn-v9-api

    # Production (with workers):
    uvicorn arn.api.server:app --host 0.0.0.0 --port 8742 --workers 1
    # NOTE: workers=1 because the embedding model is ~90MB per process.
    # For higher throughput, put a reverse proxy in front and scale
    # horizontally with separate containers per worker.
"""

import os
import sys
import time
import json
import shutil
import logging
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field



from arn.plugin import ARNPlugin

logger = logging.getLogger("arn.api")

# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT = os.environ.get("ARN_DATA_ROOT", os.path.expanduser("~/.arn_data"))
API_KEY = os.environ.get("ARN_API_KEY", None)  # Set to enable auth
MAX_AGENTS = int(os.environ.get("ARN_MAX_AGENTS", "100"))
RATE_LIMIT_RPM = int(os.environ.get("ARN_RATE_LIMIT_RPM", "300"))  # requests per minute
MAX_CONTENT_LENGTH = int(os.environ.get("ARN_MAX_CONTENT_LENGTH", "10000"))  # chars


# =========================================================
# PYDANTIC MODELS
# =========================================================

class StoreRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$',
                          description="Agent namespace identifier")
    content: str = Field(..., min_length=1, max_length=10000,
                         description="Text content to store")
    importance: float = Field(0.5, ge=0.0, le=1.0,
                              description="Importance score 0.0-1.0")
    tags: List[str] = Field(default_factory=list,
                            description="Categorical tags")
    source: str = Field("api", max_length=50,
                        description="Source identifier")
    context: dict = Field(default_factory=dict,
                          description="Additional context metadata")


class StoreResponse(BaseModel):
    stored: bool
    episode_id: int
    prediction_error: float
    domain: Optional[str]
    surprising: bool


class RecallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    query: str = Field(..., min_length=1, max_length=5000,
                       description="Natural language query")
    top_k: int = Field(5, ge=1, le=50,
                       description="Number of results to return")
    memory_types: Optional[List[str]] = Field(
        None, description="Filter: 'episodic', 'semantic', or both")


class RecallResult(BaseModel):
    content: str
    score: float
    type: str
    similarity: float
    importance: Optional[float] = None
    confidence: Optional[float] = None
    evidence_count: Optional[int] = None
    has_contradictions: Optional[bool] = None
    age_hours: Optional[float] = None


class RecallResponse(BaseModel):
    results: List[RecallResult]
    query: str
    agent_id: str
    latency_ms: float


class ContextRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    query: Optional[str] = Field(None, max_length=5000)
    max_tokens: int = Field(1000, ge=100, le=10000)


class ContextResponse(BaseModel):
    context: str
    agent_id: str
    latency_ms: float


class MaintainRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')


class StatsResponse(BaseModel):
    agent_id: str
    total_experiences: int
    consolidation_count: int
    episodic_count: int
    semantic_count: int
    working_memory_active: int
    storage_mb: float
    embedding_model_loaded: bool
    embedding_dim: int
    cache_hit_rate: float
    columns: list


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    agents_loaded: int
    data_root: str


class DeleteAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64,
                          pattern=r'^[a-zA-Z0-9_\-]+$')
    confirm: bool = Field(False, description="Must be true to delete")


# =========================================================
# AGENT POOL (lazy-loaded, cached plugins per agent_id)
# =========================================================

class AgentPool:
    """
    Manages ARNPlugin instances per agent_id.
    Lazy-loaded: first request for an agent_id creates the plugin.
    All plugins share the same embedding model in memory.
    """

    def __init__(self, data_root: str, max_agents: int = 100):
        self._plugins: dict[str, ARNPlugin] = {}
        self._data_root = data_root
        self._max_agents = max_agents
        self._access_times: dict[str, float] = {}

    def get(self, agent_id: str) -> ARNPlugin:
        if agent_id not in self._plugins:
            if len(self._plugins) >= self._max_agents:
                self._evict_oldest()

            self._plugins[agent_id] = ARNPlugin(
                agent_id=agent_id,
                data_root=self._data_root,
                auto_consolidate=True,
                consolidation_threshold=128,
            )

        self._access_times[agent_id] = time.time()
        return self._plugins[agent_id]

    def _evict_oldest(self):
        """Evict the least recently used agent to make room."""
        if not self._access_times:
            return
        oldest = min(self._access_times, key=self._access_times.get)
        plugin = self._plugins.pop(oldest, None)
        self._access_times.pop(oldest, None)
        if plugin:
            plugin.shutdown()
            logger.info(f"Evicted agent '{oldest}' from pool")

    def delete_agent(self, agent_id: str):
        """Delete all data for an agent."""
        plugin = self._plugins.pop(agent_id, None)
        self._access_times.pop(agent_id, None)
        if plugin:
            plugin.shutdown()

        agent_dir = os.path.join(self._data_root, agent_id)
        if os.path.exists(agent_dir):
            shutil.rmtree(agent_dir)
            logger.info(f"Deleted agent data: {agent_dir}")

    @property
    def loaded_count(self) -> int:
        return len(self._plugins)

    def shutdown_all(self):
        for agent_id, plugin in self._plugins.items():
            plugin.shutdown()
        self._plugins.clear()
        self._access_times.clear()


# =========================================================
# RATE LIMITER
# =========================================================

class RateLimiter:
    """Simple sliding-window rate limiter per agent_id."""

    def __init__(self, rpm: int = 300):
        self._rpm = rpm
        self._windows: dict[str, list] = defaultdict(list)

    def check(self, agent_id: str) -> bool:
        now = time.time()
        window = self._windows[agent_id]
        # Remove timestamps older than 60s
        self._windows[agent_id] = [t for t in window if now - t < 60]
        if len(self._windows[agent_id]) >= self._rpm:
            return False
        self._windows[agent_id].append(now)
        return True


# =========================================================
# APP SETUP
# =========================================================

pool: Optional[AgentPool] = None
rate_limiter: Optional[RateLimiter] = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, rate_limiter, start_time
    start_time = time.time()
    
    # Fail fast: verify the embedding model loads before accepting requests.
    # A degraded ARN (hash vectors) is worse than no ARN — it returns
    # confident-looking garbage. Better to refuse to start.
    from arn.core.embeddings import EmbeddingEngine
    _test_engine = EmbeddingEngine(use_model=True)
    if _test_engine.is_degraded:
        logger.critical(
            "FATAL: Embedding model could not be loaded. "
            "The API server CANNOT function without real embeddings. "
            "Install sentence-transformers: pip install sentence-transformers "
            "and ensure the model can be downloaded (internet on first run) "
            "or pre-cached at ~/.cache/huggingface/hub/"
        )
        raise RuntimeError(
            "ARN API cannot start: embedding model unavailable. "
            "Install: pip install sentence-transformers"
        )
    del _test_engine  # Free the test instance
    
    pool = AgentPool(data_root=DATA_ROOT, max_agents=MAX_AGENTS)
    rate_limiter = RateLimiter(rpm=RATE_LIMIT_RPM)
    logger.info(f"ARN v9 API started. Data root: {DATA_ROOT}")
    yield
    pool.shutdown_all()
    logger.info("ARN v9 API shut down.")


app = FastAPI(
    title="ARN API",
    description="Brain-inspired persistent memory for AI agents. Local, free, Pi-compatible.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AUTH & RATE LIMIT MIDDLEWARE
# =========================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Optional API key auth. Only enforced if ARN_API_KEY is set."""
    if API_KEY is not None:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def check_rate_limit(request: Request):
    """Rate limit check. Extracts agent_id from body if possible."""
    # Rate limiting is applied per-route since agent_id is in the body
    pass


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns "ok" if the server is running and embeddings are functional.
    Returns "degraded" if any loaded agent has lost its embedding model
    (shouldn't happen in normal operation, but catches runtime corruption).
    """
    status = "ok"
    
    # Check if any loaded agent is in degraded mode
    if pool:
        for agent_id, plugin in pool._plugins.items():
            if plugin._arn.embedder.is_degraded:
                status = "degraded"
                break
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=round(time.time() - start_time, 1),
        agents_loaded=pool.loaded_count if pool else 0,
        data_root=DATA_ROOT,
    )


@app.post("/v1/memory/store", response_model=StoreResponse,
          dependencies=[Depends(verify_api_key)])
async def store_memory(req: StoreRequest):
    """
    Store a new memory for an agent.

    The memory is encoded into a 384-dim semantic vector, processed
    through domain columns for prediction error, and persisted to
    SQLite + memmap storage.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    result = plugin.store(
        content=req.content,
        importance=req.importance,
        tags=req.tags,
        source=req.source,
        context=req.context,
    )

    return StoreResponse(
        stored=result["stored"],
        episode_id=result["episode_id"],
        prediction_error=round(result["prediction_error"], 4),
        domain=result.get("domain"),
        surprising=result.get("surprising", False),
    )


@app.post("/v1/memory/recall", response_model=RecallResponse,
          dependencies=[Depends(verify_api_key)])
async def recall_memory(req: RecallRequest):
    """
    Recall relevant memories for a query.

    Searches both episodic (specific events) and semantic (consolidated
    knowledge) memory, scoring by semantic similarity with importance
    and recency as minor factors.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.time()
    plugin = pool.get(req.agent_id)
    raw_results = plugin.recall(
        query=req.query,
        top_k=req.top_k,
        memory_types=req.memory_types,
    )
    latency = (time.time() - start) * 1000

    results = [RecallResult(**r) for r in raw_results]

    return RecallResponse(
        results=results,
        query=req.query,
        agent_id=req.agent_id,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/memory/context", response_model=ContextResponse,
          dependencies=[Depends(verify_api_key)])
async def get_context(req: ContextRequest):
    """
    Get a formatted context window for LLM prompt injection.

    Returns a markdown-formatted string containing working memory
    contents and relevant long-term memories, suitable for prepending
    to a system prompt.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    start = time.time()
    plugin = pool.get(req.agent_id)
    context = plugin.get_context_window(
        query=req.query,
        max_tokens=req.max_tokens,
    )
    latency = (time.time() - start) * 1000

    return ContextResponse(
        context=context,
        agent_id=req.agent_id,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/memory/maintain",
          dependencies=[Depends(verify_api_key)])
async def maintain(req: MaintainRequest):
    """
    Run memory consolidation for an agent.

    Clusters episodic memories into semantic knowledge, detects
    contradictions, and prunes old low-importance episodes.
    Call during idle periods.
    """
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plugin = pool.get(req.agent_id)
    stats = plugin.maintain()
    return stats


@app.get("/v1/memory/stats/{agent_id}", response_model=StatsResponse,
         dependencies=[Depends(verify_api_key)])
async def get_stats(agent_id: str):
    """Get comprehensive statistics for an agent's memory system."""
    plugin = pool.get(agent_id)
    raw = plugin.get_stats()

    return StatsResponse(
        agent_id=agent_id,
        total_experiences=raw["total_experiences"],
        consolidation_count=raw["consolidation_count"],
        episodic_count=raw["episodic_count"],
        semantic_count=raw["semantic_count"],
        working_memory_active=raw["working_memory_active"],
        storage_mb=round(raw["storage"]["total_size_mb"], 2),
        embedding_model_loaded=raw["embeddings"]["model_loaded"],
        embedding_dim=raw["embeddings"]["embedding_dim"],
        cache_hit_rate=round(raw["embeddings"]["cache_hit_rate"], 4),
        columns=raw["columns"],
    )


@app.delete("/v1/memory/agent",
            dependencies=[Depends(verify_api_key)])
async def delete_agent(req: DeleteAgentRequest):
    """
    Delete ALL data for an agent. Irreversible.
    Set confirm=true to actually delete.
    """
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to delete all agent data. This is irreversible."
        )

    pool.delete_agent(req.agent_id)
    return {"deleted": True, "agent_id": req.agent_id}


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "arn.api.server:app",
        host="0.0.0.0",
        port=8742,
        reload=False,
        log_level="info",
    )

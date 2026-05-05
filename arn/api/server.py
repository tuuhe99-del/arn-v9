"""
ARN REST API Server
========================
Production hardened. Changes vs original:

  1. Auth ON by default — ARN_API_KEY must be set or server refuses to start.
     Set ARN_API_KEY_DISABLED=1 only in a trusted private network.
  2. Rate-limit counters stored in SQLite so they survive restarts.
  3. Atomic backup endpoint + scheduled hourly backup.
  4. /v1/metrics endpoint for monitoring.
  5. Per-agent threading handled by StorageEngine locks (see persistence.py).

Endpoints:
    POST   /v1/memory/store
    POST   /v1/memory/recall
    POST   /v1/memory/context
    POST   /v1/memory/maintain
    GET    /v1/memory/stats/{agent_id}
    GET    /v1/health
    GET    /v1/metrics
    POST   /v1/backup
    DELETE /v1/memory/agent
"""

import os
import sys
import time
import json
import shutil
import sqlite3
import logging
import threading
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

_api_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(os.path.dirname(_api_dir))
sys.path.insert(0, _package_root)

from arn.plugin import ARNPlugin

logger = logging.getLogger("arn.api")

# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT          = os.environ.get("ARN_DATA_ROOT", os.path.expanduser("~/.arn_data"))
API_KEY            = os.environ.get("ARN_API_KEY", "")
AUTH_DISABLED      = os.environ.get("ARN_API_KEY_DISABLED", "").strip().lower() in ("1", "true", "yes")
MAX_AGENTS         = int(os.environ.get("ARN_MAX_AGENTS", "100"))
RATE_LIMIT_RPM     = int(os.environ.get("ARN_RATE_LIMIT_RPM", "300"))
MAX_CONTENT_LENGTH = int(os.environ.get("ARN_MAX_CONTENT_LENGTH", "10000"))
BACKUP_DIR         = os.environ.get("ARN_BACKUP_DIR", os.path.join(DATA_ROOT, "_backups"))
BACKUP_INTERVAL_S  = int(os.environ.get("ARN_BACKUP_INTERVAL_S", str(60 * 60)))


# =========================================================
# PERSISTENT RATE LIMITER
# =========================================================

class PersistentRateLimiter:
    """
    Sliding-window rate limiter backed by SQLite.
    Survives server restarts — counters are not lost on restart.
    """

    def __init__(self, db_path: str, rpm: int = 300):
        self._rpm = rpm
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_windows (
                    agent_id TEXT NOT NULL,
                    ts       REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_agent ON rate_windows(agent_id, ts)"
            )
            conn.commit()

    def check(self, agent_id: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "DELETE FROM rate_windows WHERE agent_id=? AND ts<?",
                    (agent_id, window_start)
                )
                count = conn.execute(
                    "SELECT COUNT(*) FROM rate_windows WHERE agent_id=?",
                    (agent_id,)
                ).fetchone()[0]
                if count >= self._rpm:
                    return False
                conn.execute(
                    "INSERT INTO rate_windows(agent_id, ts) VALUES(?,?)",
                    (agent_id, now)
                )
                conn.commit()
        return True

    def cleanup(self):
        cutoff = time.time() - 60.0
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM rate_windows WHERE ts<?", (cutoff,))
                conn.commit()


# =========================================================
# AGENT POOL
# =========================================================

class AgentPool:
    def __init__(self, data_root: str, max_agents: int = 100):
        self._plugins: dict[str, ARNPlugin] = {}
        self._data_root = data_root
        self._max_agents = max_agents
        self._access_times: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, agent_id: str) -> ARNPlugin:
        with self._lock:
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
        if not self._access_times:
            return
        oldest = min(self._access_times, key=self._access_times.get)
        plugin = self._plugins.pop(oldest, None)
        self._access_times.pop(oldest, None)
        if plugin:
            plugin.shutdown()
            logger.info(f"Evicted agent '{oldest}' from pool")

    def delete_agent(self, agent_id: str):
        with self._lock:
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
        with self._lock:
            for plugin in self._plugins.values():
                plugin.shutdown()
            self._plugins.clear()
            self._access_times.clear()


# =========================================================
# BACKUP
# =========================================================

def run_backup(data_root: str, backup_dir: str) -> dict:
    """
    Atomic backup of all agent data.
    Uses SQLite VACUUM INTO for a clean, consistent snapshot per agent.
    Keeps the last 5 backups.
    """
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_dest = Path(backup_dir) / f"_tmp_{ts}"
    final_dest = Path(backup_dir) / ts

    try:
        tmp_dest.mkdir(parents=True)
        backed = 0
        data_path = Path(data_root)

        for agent_dir in data_path.iterdir():
            if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
                continue
            for model_dir in agent_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                dest = tmp_dest / agent_dir.name / model_dir.name
                dest.mkdir(parents=True, exist_ok=True)

                db_src = model_dir / "arn_metadata.db"
                if db_src.exists():
                    dest_db = str(dest / "arn_metadata.db")
                    with sqlite3.connect(str(db_src)) as conn:
                        conn.execute(f"VACUUM INTO '{dest_db}'")

                for npy in model_dir.glob("*.npy"):
                    shutil.copy2(str(npy), str(dest / npy.name))

                backed += 1

        tmp_dest.rename(final_dest)
        logger.info(f"Backup complete: {final_dest} ({backed} dirs)")

        # Prune: keep last 5
        all_backups = sorted(
            [d for d in Path(backup_dir).iterdir()
             if d.is_dir() and not d.name.startswith("_")],
            key=lambda d: d.name
        )
        for old in all_backups[:-5]:
            shutil.rmtree(str(old))
            logger.info(f"Pruned old backup: {old}")

        return {"ok": True, "path": str(final_dest), "agent_model_dirs": backed, "ts": ts}

    except Exception as exc:
        logger.error(f"Backup failed: {exc}", exc_info=True)
        if tmp_dest.exists():
            shutil.rmtree(str(tmp_dest), ignore_errors=True)
        return {"ok": False, "error": str(exc)}


# =========================================================
# METRICS
# =========================================================

class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.requests_total = 0
        self.requests_429 = 0
        self.requests_500 = 0
        self.store_calls = 0
        self.recall_calls = 0
        self.last_backup_ts: Optional[str] = None
        self.last_backup_ok: Optional[bool] = None

    def inc(self, field: str, by: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + by)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "requests_total": self.requests_total,
                "requests_429_rate_limited": self.requests_429,
                "requests_500_errors": self.requests_500,
                "store_calls": self.store_calls,
                "recall_calls": self.recall_calls,
                "last_backup_ts": self.last_backup_ts,
                "last_backup_ok": self.last_backup_ok,
            }


# =========================================================
# PYDANTIC MODELS
# =========================================================

class StoreRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    content: str = Field(..., min_length=1, max_length=10000)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    source: str = Field("api", max_length=50)
    context: dict = Field(default_factory=dict)
    time_context: str = Field("current", pattern=r'^(past|current|future|timeless)$')
    memory_type: str = Field("episode", max_length=50)
    scope: str = Field("global", max_length=100)
    priority: str = Field("normal", pattern=r'^(low|normal|high|critical)$')
    name: Optional[str] = Field(None, max_length=100)

class StoreResponse(BaseModel):
    stored: bool
    episode_id: int
    prediction_error: float
    domain: Optional[str]
    surprising: bool

class RecallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(5, ge=1, le=50)
    memory_types: Optional[List[str]] = None

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
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    query: Optional[str] = Field(None, max_length=5000)
    max_tokens: int = Field(1000, ge=100, le=10000)

class ContextResponse(BaseModel):
    context: str
    agent_id: str
    latency_ms: float

class MaintainRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')

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
    auth_enabled: bool

class DeleteAgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    confirm: bool = Field(False)

class ForgetRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    episode_ids: Optional[List[int]] = None
    forget_all: bool = Field(False)



class IdentitySetRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, max_length=1000)
    must: List[str] = Field(default_factory=list)
    must_not: List[str] = Field(default_factory=list)

class AgentRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')

class RuleAddRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-\*]+$')
    rule: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field("high", pattern=r'^(low|normal|high|critical)$')

class ProcedureAddRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    name: str = Field(..., min_length=1, max_length=120)
    steps: List[str] = Field(..., min_length=1)
    success: Optional[str] = Field(None, max_length=2000)
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')

class ProcedureRecallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    query: str = Field(..., min_length=1, max_length=5000)
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    top_k: int = Field(5, ge=1, le=20)

class ErrorLessonAddRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    mistake: str = Field(..., min_length=1, max_length=5000)
    fix: Optional[str] = Field(None, max_length=5000)
    lesson: Optional[str] = Field(None, max_length=5000)
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    task: Optional[str] = Field(None, max_length=500)

class ErrorLessonListRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    query: Optional[str] = Field(None, max_length=5000)
    limit: int = Field(10, ge=1, le=50)

class ContextPacketRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    query: Optional[str] = Field(None, max_length=5000)
    task: Optional[str] = Field(None, max_length=1000)
    max_tokens: int = Field(2000, ge=100, le=10000)

class ContextPacketResponse(BaseModel):
    context: str
    agent_id: str
    target_agent: str
    latency_ms: float


class ShareSendRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    from_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    to_agents: List[str] = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=10000)
    task: Optional[str] = Field(None, max_length=1000)
    importance: float = Field(0.75, ge=0.0, le=1.0)
    priority: str = Field("high", pattern=r'^(low|normal|high|critical)$')
    tags: List[str] = Field(default_factory=list)

class ShareListRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    target_agent: Optional[str] = Field(None, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')
    direction: Optional[str] = Field(None, pattern=r'^(inbox|outbox)$')
    query: Optional[str] = Field(None, max_length=5000)
    task: Optional[str] = Field(None, max_length=1000)
    limit: int = Field(20, ge=1, le=100)

# =========================================================
# APP GLOBALS
# =========================================================

pool: Optional[AgentPool] = None
rate_limiter: Optional[PersistentRateLimiter] = None
metrics: Optional[Metrics] = None
_backup_stop = threading.Event()


def _backup_loop(data_root: str, backup_dir: str, interval_s: int):
    while not _backup_stop.wait(timeout=interval_s):
        result = run_backup(data_root, backup_dir)
        if metrics:
            metrics.last_backup_ts = result.get("ts")
            metrics.last_backup_ok = result.get("ok")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, rate_limiter, metrics

    # Security gate
    if not AUTH_DISABLED and not API_KEY:
        raise RuntimeError(
            "ARN API cannot start: ARN_API_KEY is not set.\n"
            "Set it: export ARN_API_KEY=your-secret-here\n"
            "Disable auth on a private network only: export ARN_API_KEY_DISABLED=1"
        )

    # Embedding sanity check
    from arn.core.embeddings import EmbeddingEngine
    _test = EmbeddingEngine(use_model=True)
    if _test.is_degraded:
        raise RuntimeError(
            "ARN API cannot start: embedding model unavailable.\n"
            "Fix: pip install sentence-transformers"
        )
    del _test

    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)

    rate_db = os.path.join(DATA_ROOT, "_rate_limits.db")
    rate_limiter = PersistentRateLimiter(rate_db, rpm=RATE_LIMIT_RPM)
    pool = AgentPool(data_root=DATA_ROOT, max_agents=MAX_AGENTS)
    metrics = Metrics()

    _backup_stop.clear()
    t = threading.Thread(
        target=_backup_loop,
        args=(DATA_ROOT, BACKUP_DIR, BACKUP_INTERVAL_S),
        daemon=True,
        name="arn-backup",
    )
    t.start()

    logger.info(f"ARN API started. auth={'disabled' if AUTH_DISABLED else 'enabled'}")
    yield

    _backup_stop.set()
    pool.shutdown_all()
    logger.info("ARN API shut down.")


from arn import __version__ as ARN_VERSION

app = FastAPI(
    title="ARN API",
    description="Local semantic memory for AI agents",
    version=ARN_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AUTH
# =========================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if AUTH_DISABLED:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =========================================================
# MIDDLEWARE
# =========================================================

@app.middleware("http")
async def count_requests(request: Request, call_next):
    if metrics:
        metrics.inc("requests_total")
    response = await call_next(request)
    if metrics:
        if response.status_code == 429:
            metrics.inc("requests_429")
        elif response.status_code >= 500:
            metrics.inc("requests_500")
    return response


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    """Serve the web UI."""
    ui_path = Path(__file__).parent / "ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    status = "ok"
    if pool:
        for plugin in pool._plugins.values():
            if plugin._arn.embedder.is_degraded:
                status = "degraded"
                break
    return HealthResponse(
        status=status,
        version=ARN_VERSION,
        uptime_seconds=round(time.time() - metrics.start_time, 1) if metrics else 0,
        agents_loaded=pool.loaded_count if pool else 0,
        data_root=DATA_ROOT,
        auth_enabled=not AUTH_DISABLED,
    )


@app.get("/v1/version")
async def get_version():
    """Current server version — no auth required so the UI can read it."""
    return {"version": ARN_VERSION}


@app.get("/v1/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics():
    if not metrics:
        raise HTTPException(status_code=503, detail="Server not ready")
    snap = metrics.snapshot()
    snap["agents_loaded"] = pool.loaded_count if pool else 0
    snap["rate_limit_rpm"] = RATE_LIMIT_RPM
    snap["max_agents"] = MAX_AGENTS
    return snap


@app.post("/v1/backup", dependencies=[Depends(verify_api_key)])
async def trigger_backup():
    result = run_backup(DATA_ROOT, BACKUP_DIR)
    if metrics:
        metrics.last_backup_ts = result.get("ts")
        metrics.last_backup_ok = result.get("ok")
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Backup failed"))
    return result


@app.post("/v1/memory/store", response_model=StoreResponse,
          dependencies=[Depends(verify_api_key)])
async def store_memory(req: StoreRequest):
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if metrics:
        metrics.inc("store_calls")
    plugin = pool.get(req.agent_id)
    result = plugin.store(
        content=req.content,
        importance=req.importance,
        tags=req.tags,
        source=req.source,
        context=req.context,
        time_context=req.time_context,
        memory_type=req.memory_type,
        scope=req.scope,
        priority=req.priority,
        name=req.name,
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
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if metrics:
        metrics.inc("recall_calls")
    start = time.time()
    plugin = pool.get(req.agent_id)
    raw = plugin.recall(query=req.query, top_k=req.top_k, memory_types=req.memory_types)
    latency = (time.time() - start) * 1000
    return RecallResponse(
        results=[RecallResult(**r) for r in raw],
        query=req.query,
        agent_id=req.agent_id,
        latency_ms=round(latency, 2),
    )


@app.post("/v1/memory/context", response_model=ContextResponse,
          dependencies=[Depends(verify_api_key)])
async def get_context(req: ContextRequest):
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    start = time.time()
    plugin = pool.get(req.agent_id)
    context = plugin.get_context_window(query=req.query, max_tokens=req.max_tokens)
    latency = (time.time() - start) * 1000
    return ContextResponse(context=context, agent_id=req.agent_id,
                           latency_ms=round(latency, 2))


@app.post("/v1/memory/maintain", dependencies=[Depends(verify_api_key)])
async def maintain(req: MaintainRequest):
    if not rate_limiter.check(req.agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    plugin = pool.get(req.agent_id)
    return plugin.maintain()


@app.get("/v1/memory/stats/{agent_id}", response_model=StatsResponse,
         dependencies=[Depends(verify_api_key)])
async def get_stats(agent_id: str):
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


@app.delete("/v1/memory/agent", dependencies=[Depends(verify_api_key)])
async def delete_agent(req: DeleteAgentRequest):
    if not req.confirm:
        raise HTTPException(status_code=400,
                            detail="Set confirm=true to delete. This is irreversible.")
    pool.delete_agent(req.agent_id)
    return {"deleted": True, "agent_id": req.agent_id}


@app.get("/v1/memory/list/{agent_id}", dependencies=[Depends(verify_api_key)])
async def list_memories(agent_id: str, limit: int = 50, type: str = "all"):
    """List stored memories for an agent."""
    plugin = pool.get(agent_id)
    return plugin.list_memories(limit=limit, memory_type=type)


@app.get("/v1/memory/search/{agent_id}", dependencies=[Depends(verify_api_key)])
async def search_memories(agent_id: str, q: str, limit: int = 20):
    """Keyword search across an agent's memories."""
    plugin = pool.get(agent_id)
    return plugin.search_memories(q, limit=limit)


@app.post("/v1/memory/forget", dependencies=[Depends(verify_api_key)])
async def forget_memories(req: ForgetRequest):
    """Delete specific memories or all memories for an agent."""
    if req.forget_all and not req.episode_ids:
        # Safety: require explicit forget_all flag
        pass
    plugin = pool.get(req.agent_id)
    result = plugin.forget(episode_ids=req.episode_ids, forget_all=req.forget_all)
    return result



# =========================================================
# HUMAN MEMORY CORE ENDPOINTS
# =========================================================

def _check_rate(agent_id: str):
    if not rate_limiter.check(agent_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.post("/v1/human/identity/set", dependencies=[Depends(verify_api_key)])
async def api_set_identity(req: IdentitySetRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.set_identity(
        agent=req.target_agent,
        name=req.name,
        role=req.role,
        must=req.must,
        must_not=req.must_not,
    )

@app.post("/v1/human/identity/get", dependencies=[Depends(verify_api_key)])
async def api_get_identity(req: AgentRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.get_identity(req.target_agent or req.agent_id)

@app.post("/v1/human/rule/add", dependencies=[Depends(verify_api_key)])
async def api_add_rule(req: RuleAddRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.add_rule(req.target_agent, req.rule, priority=req.priority)

@app.post("/v1/human/rule/list", dependencies=[Depends(verify_api_key)])
async def api_list_rules(req: AgentRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.list_rules(req.target_agent)

@app.post("/v1/human/procedure/add", dependencies=[Depends(verify_api_key)])
async def api_add_procedure(req: ProcedureAddRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.add_procedure(
        name=req.name,
        steps=req.steps,
        success=req.success,
        agent=req.target_agent,
    )

@app.post("/v1/human/procedure/recall", dependencies=[Depends(verify_api_key)])
async def api_recall_procedures(req: ProcedureRecallRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.recall_procedures(req.query, agent=req.target_agent, top_k=req.top_k)

@app.post("/v1/human/error/add", dependencies=[Depends(verify_api_key)])
async def api_add_error_lesson(req: ErrorLessonAddRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.add_error_lesson(
        mistake=req.mistake,
        fix=req.fix,
        lesson=req.lesson,
        agent=req.target_agent,
        task=req.task,
    )

@app.post("/v1/human/error/list", dependencies=[Depends(verify_api_key)])
async def api_list_error_lessons(req: ErrorLessonListRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.list_error_lessons(
        agent=req.target_agent,
        query=req.query,
        limit=req.limit,
    )

@app.post("/v1/human/context-packet", response_model=ContextPacketResponse,
          dependencies=[Depends(verify_api_key)])
async def api_context_packet(req: ContextPacketRequest):
    _check_rate(req.agent_id)
    start = time.time()
    plugin = pool.get(req.agent_id)
    target = req.target_agent or req.agent_id
    context = plugin.build_context_packet(
        query=req.query,
        agent=target,
        task=req.task,
        max_tokens=req.max_tokens,
    )
    latency = (time.time() - start) * 1000
    return ContextPacketResponse(
        context=context,
        agent_id=req.agent_id,
        target_agent=target,
        latency_ms=round(latency, 2),
    )

@app.post("/v1/human/share/send", dependencies=[Depends(verify_api_key)])
async def api_share_send(req: ShareSendRequest):
    _check_rate(req.agent_id)
    for target in req.to_agents:
        if not target or not target.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail=f"Invalid target agent: {target!r}")
    plugin = pool.get(req.agent_id)
    return plugin.share_memory(
        content=req.content,
        to_agents=req.to_agents,
        from_agent=req.from_agent or req.agent_id,
        task=req.task,
        importance=req.importance,
        priority=req.priority,
        tags=req.tags,
    )

@app.post("/v1/human/share/list", dependencies=[Depends(verify_api_key)])
async def api_share_list(req: ShareListRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.list_shared_memories(
        agent=req.target_agent or req.agent_id,
        direction=req.direction,
        query=req.query,
        task=req.task,
        limit=req.limit,
    )

@app.post("/v1/human/share/inbox", dependencies=[Depends(verify_api_key)])
async def api_share_inbox(req: ShareListRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.list_shared_memories(
        agent=req.target_agent or req.agent_id,
        direction="inbox",
        query=req.query,
        task=req.task,
        limit=req.limit,
    )

@app.post("/v1/human/share/outbox", dependencies=[Depends(verify_api_key)])
async def api_share_outbox(req: ShareListRequest):
    _check_rate(req.agent_id)
    plugin = pool.get(req.agent_id)
    return plugin.list_shared_memories(
        agent=req.target_agent or req.agent_id,
        direction="outbox",
        query=req.query,
        task=req.task,
        limit=req.limit,
    )

@app.get("/v1/memory/graph/{agent_id}", dependencies=[Depends(verify_api_key)])
async def graph_query(agent_id: str, entity: str, include_superseded: bool = False):
    """Return all known facts about an entity from the fact graph."""
    plugin = pool.get(agent_id)
    return plugin.graph_query(entity, include_superseded=include_superseded)


@app.get("/v1/memory/graph/{agent_id}/history", dependencies=[Depends(verify_api_key)])
async def fact_history(agent_id: str, entity: str, relation: str):
    """Return the full evolution history of a specific fact."""
    plugin = pool.get(agent_id)
    return plugin.fact_history(entity, relation)


@app.get("/v1/memory/graph/{agent_id}/summary", dependencies=[Depends(verify_api_key)])
async def graph_summary(agent_id: str):
    """Return fact graph size and entity sample."""
    plugin = pool.get(agent_id)
    return plugin.graph_summary()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("arn.api.server:app", host="0.0.0.0", port=8742,
                reload=False, log_level="info")

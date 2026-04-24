"""
ARN → Raw / any-framework adapter
Simple Python API wrapper for use with any agent system.
"""
from pathlib import Path
import os


class ARNMemory:
    """
    Minimal wrapper for use with any agent framework.
    Drop-in memory layer — no framework dependencies.

    Usage:
        from arn.adapters.raw import ARNMemory
        mem = ARNMemory()
        mem.store("User prefers dark mode", importance=0.7)
        context = mem.recall("UI preferences")
        # Inject context into your system prompt
    """

    def __init__(self, data_dir: str = None, model_tier: str = None, agent_id: str = "default"):
        if data_dir is None:
            data_dir = os.environ.get("ARN_DATA_DIR", str(Path.home() / ".arn_data" / "default"))
        # Use ARNPlugin so we get temporal tagging support
        from arn.plugin import ARNPlugin
        # ARNPlugin uses data_root/agent_id — derive data_root from data_dir
        data_root = str(Path(data_dir).parent)
        kwargs = {"agent_id": agent_id, "data_root": data_root}
        if model_tier:
            from arn.core.embeddings import MODEL_CONFIGS
            cfg = MODEL_CONFIGS.get(model_tier)
            if cfg:
                kwargs["embedding_model"] = cfg["name"]
        self._plugin = ARNPlugin(**kwargs)

    def store(self, fact: str, importance: float = 0.5,
              source: str = "user", time_context: str = "current"):
        """Store a fact. importance 0.0-1.0 (higher = more important)."""
        self._plugin.store(fact, importance=importance,
                           source=source, time_context=time_context)

    def recall(self, query: str, top_k: int = 5) -> list:
        """Recall relevant memories. Returns list of {content, score} dicts."""
        return self._plugin.recall(query, top_k=top_k)

    def context(self, query: str, max_tokens: int = 1200) -> str:
        """Get a formatted context block ready to inject into a system prompt."""
        return self._plugin.get_context_window(query, max_tokens=max_tokens)

    def close(self):
        self._plugin.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

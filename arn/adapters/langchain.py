"""
ARN → LangChain adapter
Provides LangChain-compatible Tool wrappers around ARN recall/store.
"""
from pathlib import Path


def get_tools(data_dir: str = None, importance: float = 0.7, top_k: int = 3):
    """
    Returns a list of LangChain Tool objects for ARN memory.
    Requires: pip install langchain

    Usage:
        from arn.adapters.langchain import get_tools
        tools = get_tools(data_dir="~/.arn_data/default")
        agent = initialize_agent(tools, llm, ...)
    """
    try:
        from langchain.tools import tool as lc_tool
    except ImportError:
        raise ImportError("langchain not installed. Run: pip install langchain")

    import os
    if data_dir is None:
        data_dir = os.environ.get("ARN_DATA_DIR", str(Path.home() / ".arn_data" / "default"))

    from arn.core import ARNv9
    _arn = ARNv9(data_dir=data_dir)

    @lc_tool
    def remember(fact: str) -> str:
        """Store a fact worth remembering about the user or conversation."""
        _arn.perceive(fact, importance=importance)
        return f"stored: {fact[:60]}"

    @lc_tool
    def recall_memory(query: str) -> str:
        """Recall relevant memories before answering a question."""
        hits = _arn.recall(query, top_k=top_k)
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"[{h['score']:.2f}] {h['content']}" for h in hits)

    return [remember, recall_memory]

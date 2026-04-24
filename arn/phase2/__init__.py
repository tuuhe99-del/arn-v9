"""
ARN Phase 2 — Automatic Memory Injection Layer
================================================
Upgrades ARN from a manually-called plugin into a passive memory layer.
Memory is absorbed and injected automatically — the agent just knows.
"""

from .memory_llm import MemoryAugmentedLLM

# Friendly alias
AutoInject = MemoryAugmentedLLM

__all__ = ["MemoryAugmentedLLM", "AutoInject"]

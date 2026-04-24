"""
ARN — Adaptive Reasoning Network
=================================
Brain-inspired persistent memory for AI agents.
Runs locally. Costs $0/month. Works on a Raspberry Pi.

Quick start:
    from arn import ARNv9
    arn = ARNv9(data_dir="~/.arn_data")
    arn.perceive("User prefers Python", importance=0.8)
    results = arn.recall("what language does user prefer?")
    arn.close()

With auto-inject (Phase 2):
    from arn import AutoInject
    ai = AutoInject(agent_id="my_agent")
    context_block = ai.get_context("user message here")
    ai.absorb("user message here")  # passive learning
"""

from .core import ARNv9, EmbeddingEngine
from .plugin import ARNPlugin
from .phase2 import AutoInject

__version__ = "1.0.0"
__all__ = ["ARNv9", "EmbeddingEngine", "ARNPlugin", "AutoInject"]

"""
ARN v9 Memory-Augmented LLM Wrapper
======================================
Level 2 integration: memory IS part of the AI, not a tool it calls.

Instead of the model deciding when to store/recall, this wrapper:
1. Auto-stores every user message (passive learning)
2. Auto-stores every assistant response (self-knowledge)
3. Auto-recalls relevant memories before every LLM call
4. Injects memories into the system prompt so the model "just knows"
5. Detects importance automatically from content signals

To the user, the model simply remembers. No tool calls, no "let me
check my memory." It just knows.

Works with any LLM that accepts an OpenAI-compatible chat format:
- Local models via Ollama (Qwen, Phi, Llama, Mistral, Gemma)
- llama.cpp server
- LM Studio / LocalAI / vLLM
- Any OpenAI-compatible API
- Custom callback function

Usage:
    from arn.memory_llm import MemoryAugmentedLLM

    # With Ollama
    llm = MemoryAugmentedLLM(
        agent_id="my_agent",
        backend="ollama",
        model="qwen2.5:0.5b",
    )

    # Conversation — memory is automatic
    response = llm.chat("Hey, my name is Mohamed")
    # ARN auto-stores: "Hey, my name is Mohamed" (importance: 0.9)

    response = llm.chat("What's my name?")
    # ARN recalls the name, injects it into prompt, model responds correctly

    # Next session — memories persist
    llm2 = MemoryAugmentedLLM(agent_id="my_agent", backend="ollama", model="qwen2.5:0.5b")
    response = llm2.chat("Do you remember me?")
    # Yes — because ARN persisted the memories to disk
"""

import os
import re
import time
import json
import logging
from typing import List, Dict, Optional, Callable, Any

from arn.plugin import ARNPlugin

logger = logging.getLogger("arn.memory_llm")


# =========================================================
# IMPORTANCE DETECTION
# =========================================================

_HIGH_IMPORTANCE_PATTERNS = [
    # Identity
    (r'\bmy name is\b', 0.9),
    (r'\bi am\b.*\b(developer|engineer|student|designer|manager|doctor|teacher)\b', 0.8),
    (r'\bi work (at|for|with)\b', 0.8),
    (r'\bi live in\b', 0.7),
    (r'\bi\'m from\b', 0.7),
    # Preferences
    (r'\bi (prefer|like|love|hate|use|want)\b', 0.7),
    (r'\bmy favorite\b', 0.7),
    (r'\bi always\b', 0.6),
    (r'\bi never\b', 0.6),
    # Decisions
    (r'\bwe decided\b', 0.8),
    (r'\bremember (that|this)\b', 0.8),
    (r'\bdon\'t forget\b', 0.8),
    (r'\bimportant:\b', 0.8),
    # Technical
    (r'\bruns on\b', 0.6),
    (r'\bdeployed (on|to|at)\b', 0.7),
    (r'\bthe (bug|error|issue) (is|was)\b', 0.7),
    (r'\bfixed (the|a|an)\b', 0.6),
    # Temporal
    (r'\bi used to\b', 0.6),
    (r'\bi (recently|just) (started|stopped|switched|changed)\b', 0.7),
    (r'\bno longer\b', 0.6),
]

_LOW_IMPORTANCE_PATTERNS = [
    r'^(hi|hello|hey|thanks|thank you|ok|okay|sure|got it|cool|nice|great)\s*[.!?]*$',
    r'^(good morning|good evening|good night|goodbye|bye)\s*[.!?]*$',
    r'^(yes|no|maybe|idk|lol|haha)\s*[.!?]*$',
]

_NEVER_STORE_PATTERNS = [
    r'\b(password|passwd|api[_-]?key|secret[_-]?key|token|ssn|social security)\s*(is|=|:)\s*\S+',
    r'\b[A-Za-z0-9+/]{40,}={0,2}\b',  # Long base64-like strings (potential keys)
]


def detect_importance(text: str) -> float:
    """
    Auto-detect importance of a message.
    Returns 0.0 for content that must NOT be stored (credentials).
    Returns 0.1-0.2 for chitchat.
    Returns 0.3-0.9 for substantive content.
    """
    lower = text.lower().strip()

    for pattern in _NEVER_STORE_PATTERNS:
        if re.search(pattern, lower):
            return 0.0

    for pattern in _LOW_IMPORTANCE_PATTERNS:
        if re.match(pattern, lower):
            return 0.15

    max_importance = 0.3
    for pattern, importance in _HIGH_IMPORTANCE_PATTERNS:
        if re.search(pattern, lower):
            max_importance = max(max_importance, importance)

    word_count = len(lower.split())
    if word_count < 4:
        max_importance = min(max_importance, 0.3)
    elif word_count > 20:
        max_importance = max(max_importance, 0.4)

    return max_importance


def detect_time_context(text: str) -> str:
    """Auto-detect temporal context from message content."""
    lower = text.lower()
    past = ['used to', 'previously', 'before', 'back when', 'no longer',
            'stopped', 'quit', 'gave up', 'switched from']
    future = ['going to', 'plan to', 'will be', 'want to', 'thinking about',
              'considering', 'next week', 'next month', 'soon']

    if sum(1 for s in past if s in lower) > sum(1 for s in future if s in lower):
        return 'past'
    elif sum(1 for s in future if s in lower) > 0:
        return 'future'
    return 'current'


# =========================================================
# MEMORY PROMPT BUILDER
# =========================================================

def build_memory_system_prompt(
    base_prompt: str,
    memories: List[dict],
) -> str:
    """
    Inject memories into the system prompt so the model reads them
    as "things I know" rather than "things a tool returned."
    """
    strong = [m for m in memories
              if m.get('confidence_tier', 'medium') in ('high', 'medium')]

    if not strong:
        return base_prompt

    lines = []
    for m in strong[:10]:
        content = m['content']
        tc = m.get('time_context', 'current')
        if tc == 'past':
            content = f"(Previously) {content}"
        elif tc == 'future':
            content = f"(Planned) {content}"
        if m.get('has_contradictions'):
            content += " [conflicting info exists]"
        lines.append(f"- {content}")

    return f"""{base_prompt}

<memory>
Things you know from previous conversations with this user:
{chr(10).join(lines)}

Use this knowledge naturally. Don't mention that you're reading from memory.
If something seems outdated or contradictory, mention the discrepancy naturally.
If nothing here is relevant, just respond normally.
</memory>"""


# =========================================================
# BACKEND ADAPTERS
# =========================================================

class OllamaBackend:
    """Adapter for Ollama local LLM server."""

    def __init__(self, model: str = "qwen2.5:0.5b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[dict], **kwargs) -> str:
        import urllib.request
        url = f"{self.base_url}/api/chat"
        payload = json.dumps({
            "model": self.model, "messages": messages, "stream": False, **kwargs,
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")


class OpenAICompatibleBackend:
    """Adapter for any OpenAI-compatible API (llama.cpp, vLLM, LM Studio, LocalAI)."""

    def __init__(self, model: str = "local-model",
                 base_url: str = "http://localhost:8080",
                 api_key: str = "not-needed"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, messages: List[dict], **kwargs) -> str:
        import urllib.request
        url = f"{self.base_url}/v1/chat/completions"
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]


class CallbackBackend:
    """Adapter for any custom function. Pass a function that takes messages → string."""

    def __init__(self, fn: Callable[[List[dict]], str]):
        self.fn = fn

    def generate(self, messages: List[dict], **kwargs) -> str:
        return self.fn(messages)


# =========================================================
# MEMORY-AUGMENTED LLM
# =========================================================

class MemoryAugmentedLLM:
    """
    An LLM that remembers. Memory is automatic, not tool-based.

    Every message is passively absorbed. Every response draws on
    relevant memories. The model doesn't "use a memory tool" —
    it just knows things from previous conversations.
    """

    def __init__(
        self,
        agent_id: str = "default",
        data_root: str = None,
        backend: Any = None,
        model: str = None,
        base_url: str = None,
        system_prompt: str = None,
        auto_consolidate: bool = True,
        consolidation_threshold: int = 128,
        recall_top_k: int = 8,
        min_store_importance: float = 0.1,
        store_assistant_responses: bool = True,
    ):
        if data_root is None:
            data_root = os.path.expanduser("~/.arn_data")

        self.memory = ARNPlugin(
            agent_id=agent_id,
            data_root=data_root,
            auto_consolidate=auto_consolidate,
            consolidation_threshold=consolidation_threshold,
        )

        self.backend = self._init_backend(backend, model, base_url)

        self.system_prompt = system_prompt or (
            "You are a helpful assistant. You have a good memory and remember "
            "details from previous conversations with this user. Be natural "
            "and conversational."
        )
        self.recall_top_k = recall_top_k
        self.min_store_importance = min_store_importance
        self.store_assistant = store_assistant_responses
        self.history: List[dict] = []
        self._turn_count = 0
        self._auto_stores = 0
        self._auto_recalls = 0
        self._last_maintain = time.time()

    def _init_backend(self, backend, model, base_url):
        if backend is None:
            return None
        if isinstance(backend, str):
            if backend == "ollama":
                return OllamaBackend(
                    model=model or "qwen2.5:0.5b",
                    base_url=base_url or "http://localhost:11434",
                )
            elif backend in ("openai", "openai-compatible", "llamacpp", "lmstudio"):
                return OpenAICompatibleBackend(
                    model=model or "local-model",
                    base_url=base_url or "http://localhost:8080",
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
        return backend

    def chat(self, user_message: str, **kwargs) -> str:
        """
        Send a message and get a response. Memory is fully automatic.

        1. Auto-stores the user message (passive learning)
        2. Auto-recalls relevant memories
        3. Injects memories into system prompt
        4. Calls the LLM
        5. Auto-stores the response
        6. Returns the response
        """
        self._turn_count += 1

        # --- Auto-store user message ---
        importance = detect_importance(user_message)
        if importance > 0.0 and importance >= self.min_store_importance:
            self.memory.store(
                content=user_message,
                importance=importance,
                source="user",
                time_context=detect_time_context(user_message),
                context={"turn": self._turn_count, "role": "user"},
            )
            self._auto_stores += 1

        # --- Auto-recall relevant memories ---
        memories = self.memory.recall(query=user_message, top_k=self.recall_top_k)
        self._auto_recalls += 1

        # --- Build augmented prompt ---
        augmented_system = build_memory_system_prompt(self.system_prompt, memories)

        # --- Call LLM ---
        if self.backend is None:
            self.history.append({"role": "user", "content": user_message})
            return None

        messages = [
            {"role": "system", "content": augmented_system},
            *self.history,
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.backend.generate(messages, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = f"[Error: {e}]"

        # --- Auto-store assistant response ---
        if self.store_assistant and response and not response.startswith("[Error"):
            resp_importance = detect_importance(response)
            if resp_importance >= 0.25:
                self.memory.store(
                    content=response,
                    importance=min(resp_importance, 0.5),
                    source="assistant",
                    time_context="current",
                    context={"turn": self._turn_count, "role": "assistant"},
                )

        # --- Update history ---
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 40:
            self.history = self.history[-40:]

        # --- Periodic maintenance ---
        if time.time() - self._last_maintain > 300:
            try:
                self.memory.maintain()
                self._last_maintain = time.time()
            except Exception:
                pass

        return response

    def remember(self, content: str, importance: float = 0.8,
                 time_context: str = "current") -> dict:
        """Explicitly inject a memory outside of conversation flow."""
        return self.memory.store(
            content=content, importance=importance,
            time_context=time_context, source="explicit",
        )

    def what_do_you_know(self, topic: str = None, top_k: int = 10) -> List[dict]:
        """Inspect what the memory system knows. For debugging."""
        return self.memory.recall(query=topic or "important facts", top_k=top_k)

    def get_stats(self) -> dict:
        """Get combined memory + wrapper stats."""
        stats = self.memory.get_stats()
        stats["wrapper"] = {
            "turns": self._turn_count,
            "auto_stores": self._auto_stores,
            "auto_recalls": self._auto_recalls,
            "history_length": len(self.history),
        }
        return stats

    def clear_session(self):
        """Clear conversation history but keep long-term memory."""
        self.history.clear()
        self._turn_count = 0

    def close(self):
        self.memory.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

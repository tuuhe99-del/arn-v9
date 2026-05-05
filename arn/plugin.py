"""
ARN OpenClaw Plugin Interface
==================================
Provides the standardized API that OpenClaw agents use to interact
with ARN's cognitive memory system.

This replaces AGENTS.md and MEMORY.md with a live, brain-inspired
memory system that learns, consolidates, and recalls adaptively.

Usage in OpenClaw:
    from arn.plugin import ARNPlugin
    
    plugin = ARNPlugin(agent_id="agent_001", data_root="./memory")
    
    # Agent stores experience
    plugin.store("User prefers Python", importance=0.8, tags=["preference"])
    
    # Agent recalls relevant context
    context = plugin.recall("What language does the user like?", top_k=5)
    
    # Background maintenance (call periodically)
    plugin.maintain()
    
    # Clean shutdown
    plugin.shutdown()
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional, Any

from .core.cognitive import ARNv9
from .core.embeddings import normalize_tier
from .core.conflicts import extract_claim, find_conflicts, fact_key, auto_extract, is_change_statement
from .core.factgraph import FactGraph

logger = logging.getLogger("arn.plugin")

# Simple human-inspired memory labels. These are metadata tags over the
# existing episodic/semantic storage, not a second complicated database.
MEMORY_TYPES = {
    "episode", "fact", "preference", "identity", "rule", "procedure",
    "error", "lesson", "task", "decision", "conflict", "shared",
}
PRIORITY_WEIGHT = {"low": 0.35, "normal": 0.55, "high": 0.75, "critical": 0.95}


def _scope_matches(stored: str, wanted: list[str]) -> bool:
    return stored in wanted or stored == "global"


class ARNPlugin:
    """
    OpenClaw-compatible memory plugin powered by ARN.
    
    Each agent gets its own isolated memory namespace while sharing
    the embedding model for efficiency.
    """
    
    def __init__(self, agent_id: str = "default",
                 data_root: str = "./arn_data",
                 use_embeddings: bool = True,
                 episodic_capacity: int = 4096,
                 semantic_capacity: int = 2048,
                 consolidation_threshold: int = 128,
                 auto_consolidate: bool = True,
                 embedding_tier: Optional[str] = None):
        
        self.agent_id = agent_id
        self.data_root = data_root
        self.embedding_tier = normalize_tier(embedding_tier)
        # Keep each embedding tier in its own vector store. This prevents
        # crashes/corruption when switching between 384-dim, 768-dim, and
        # 1024-dim models.
        data_dir = os.path.join(data_root, agent_id, f"model-{self.embedding_tier}")
        
        self._arn = ARNv9(
            data_dir=data_dir,
            use_embeddings=use_embeddings,
            episodic_capacity=episodic_capacity,
            semantic_capacity=semantic_capacity,
            consolidation_threshold=consolidation_threshold,
            auto_consolidate=auto_consolidate,
            embedding_tier=self.embedding_tier,
        )
        
        # Check if embeddings loaded — if not, warn loudly
        if self._arn.embedder.is_degraded:
            logger.critical(
                f"ARN plugin '{agent_id}' is DEGRADED: no embedding model loaded. "
                "Memory recall will return random results. "
                "Fix: pip install sentence-transformers"
            )
            import warnings
            warnings.warn(
                "ARN is running without semantic embeddings. "
                "All memory operations will return meaningless results. "
                "Install sentence-transformers to fix this.",
                RuntimeWarning,
                stacklevel=2,
            )
        
        self._last_maintain = time.time()
        self._disk_warn_mb = int(os.environ.get("ARN_DISK_WARN_MB", "2000"))
        self._graph = FactGraph(self._arn.storage)
        logger.info(f"ARN plugin initialized for agent '{agent_id}' using tier '{self.embedding_tier}'")
    
    # ===========================================
    # PRIMARY API (what agents call)
    # ===========================================
    
    def store(self, content: str, importance: float = 0.5,
              tags: List[str] = None, source: str = "agent",
              context: dict = None,
              time_context: str = "current",
              memory_type: str = "episode",
              scope: str = "global",
              priority: str = "normal",
              name: str = None) -> dict:
        """
        Store a new experience/fact/observation.
        
        Args:
            content: Text content to remember
            importance: 0.0-1.0 scale (0=trivial, 1=critical)
            tags: Optional categorical tags
            source: Where this came from (agent, user, tool, etc.)
            context: Additional context metadata
            time_context: One of 'past', 'current', 'future'.
                          Embedding models can't distinguish "used to prefer X"
                          from "currently prefers Y" — this tag lets recall
                          filter correctly when the agent explicitly marks it.
                          Default 'current'.
        
        Returns:
            Dict with episode_id, prediction_error, domain analysis
        """
        if time_context not in ('past', 'current', 'future', 'timeless'):
            raise ValueError(
                f"time_context must be 'past', 'current', 'future', or 'timeless', got '{time_context}'"
            )
        memory_type = (memory_type or "episode").lower().strip()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {sorted(MEMORY_TYPES)}, got '{memory_type}'")
        priority = (priority or "normal").lower().strip()
        if priority not in PRIORITY_WEIGHT:
            raise ValueError("priority must be low, normal, high, or critical")
        if priority in ("high", "critical"):
            importance = max(importance, PRIORITY_WEIGHT[priority])

        ctx = context or {}
        ctx['memory_type'] = memory_type
        ctx['scope'] = scope or 'global'
        ctx['priority'] = priority
        if name:
            ctx['name'] = name
        if tags:
            ctx['tags'] = tags
        ctx['source'] = source
        ctx['time_context'] = time_context

        # Soft disk limit warning
        stats = self._arn.storage.get_storage_stats()
        if stats['total_size_mb'] > self._disk_warn_mb:
            logger.warning(
                f"ARN storage for agent '{self.agent_id}' is {stats['total_size_mb']:.0f} MB "
                f"(warning threshold: {self._disk_warn_mb} MB). "
                f"Run 'arn maintain' to consolidate, or increase ARN_DISK_WARN_MB."
            )

        # Auto-extract all detectable claims and tag memory automatically.
        # Agents no longer need to manually classify what they're storing.
        all_claims = auto_extract(content)
        if all_claims:
            auto_tags = list({c.relation for c in all_claims})
            existing_tags = ctx.get('tags', [])
            ctx['tags'] = list(set(existing_tags + auto_tags))
            ctx['auto_claims'] = [c.to_dict() for c in all_claims]

        # If the content signals a factual change, boost importance slightly
        # so it ranks above older versions of the same fact in recall.
        if is_change_statement(content):
            importance = min(1.0, importance + 0.1)
            ctx['is_change'] = True

        # Store-time contradiction detection. Consolidation still detects
        # cluster-level contradictions later, but this catches user preference
        # and fact conflicts immediately so OpenClaw can react in the next turn.
        claim = extract_claim(content)
        if claim:
            ctx['claim'] = claim.to_dict()
        conflicts = find_conflicts(content, self._arn.storage.get_all_episodes(consolidated=None))
        if conflicts:
            ctx['contradiction_candidates'] = conflicts[-10:]

        result = self._arn.perceive(
            content=content,
            importance=importance,
            context=ctx,
            source=source
        )

        new_episode_id = result['episode_id']

        # Temporal supersession + fact graph update
        if all_claims:
            for c in all_claims:
                fk = fact_key(c)
                # Check if this claim supersedes an older one
                for conflict in conflicts:
                    old_id = conflict.get('old_episode_id')
                    if old_id:
                        self._arn.storage.supersede_episode(old_id, new_episode_id, fk)
                        self._graph.mark_superseded(c.subject, c.relation,
                                                    conflict['old_claim']['object'])
                # Add this claim to the graph
                self._graph.add_claim(
                    subject=c.subject,
                    relation=c.relation,
                    obj=c.object,
                    episode_id=new_episode_id,
                    confidence=importance,
                )
        elif claim and conflicts:
            # Fallback: single claim path
            fk = fact_key(claim)
            for conflict in conflicts:
                old_id = conflict.get('old_episode_id')
                if old_id:
                    self._arn.storage.supersede_episode(old_id, new_episode_id, fk)

        return {
            'stored': True,
            'episode_id': new_episode_id,
            'prediction_error': result['prediction_error'],
            'domain': result['best_domain'],
            'surprising': result['surprise_count'] >= 3,
            'claim': claim.to_dict() if claim else None,
            'contradictions': conflicts[-10:] if conflicts else [],
            'superseded': len(conflicts),
            'auto_claims': len(all_claims),
        }
    
    # Temporal keyword detection for query intent
    _PAST_KEYWORDS = {
        'used to', 'previously', 'before', 'earlier', 'formerly',
        'in the past', 'was', 'were', 'had been', 'no longer',
    }
    _CURRENT_KEYWORDS = {
        'currently', 'now', 'right now', 'at the moment', 'these days',
        'present', 'today', 'still', 'at present',
    }
    _FUTURE_KEYWORDS = {
        'will', 'going to', 'plan to', 'intend to', 'in the future',
        'next', 'upcoming', 'soon', 'eventually',
    }
    
    @classmethod
    def _detect_query_temporal(cls, query: str) -> Optional[str]:
        """
        Detect if a query expresses temporal intent.
        Returns 'past', 'current', 'future', or None.
        """
        q = query.lower()
        if any(kw in q for kw in cls._CURRENT_KEYWORDS):
            return 'current'
        if any(kw in q for kw in cls._PAST_KEYWORDS):
            return 'past'
        if any(kw in q for kw in cls._FUTURE_KEYWORDS):
            return 'future'
        if 'always' in q or 'usually' in q or 'in general' in q:
            return 'timeless'
        return None
    
    def recall(self, query: str, top_k: int = 5,
               memory_types: List[str] = None,
               time_filter: Optional[str] = None) -> List[dict]:
        """
        Recall relevant memories for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results
            memory_types: Filter by type ("episodic", "semantic", or both)
            time_filter: Explicit temporal filter ('past'/'current'/'future').
                         If None, detects from query keywords automatically.
        
        Returns:
            List of memory results with content, score, type, metadata
        """
        include_ep = True
        include_sem = True
        
        if memory_types:
            include_ep = "episodic" in memory_types
            include_sem = "semantic" in memory_types
        
        # Detect temporal intent: explicit param wins, else infer from query
        temporal_intent = time_filter or self._detect_query_temporal(query)
        
        # Get more results than needed so we have room to re-rank by temporal match
        fetch_k = top_k * 3 if temporal_intent else top_k
        
        results = self._arn.recall(
            query=query,
            top_k=fetch_k,
            include_episodic=include_ep,
            include_semantic=include_sem
        )
        
        # Apply temporal re-ranking if intent detected
        if temporal_intent:
            for r in results:
                ep_time_ctx = r.get('context', {}).get('time_context', 'current')
                if ep_time_ctx == temporal_intent or ep_time_ctx == 'timeless':
                    r['score'] += 0.3  # Strong boost for matching temporal context
                elif ep_time_ctx != 'current' and temporal_intent == 'current':
                    r['score'] -= 0.2  # Penalize past/future when query wants current
                elif ep_time_ctx == 'current' and temporal_intent in ('past', 'future'):
                    r['score'] -= 0.2  # Penalize current when query wants past/future
            results.sort(key=lambda r: r['score'], reverse=True)
            results = results[:top_k]
        
        # Simplify output for agent consumption
        simplified = []
        for r in results:
            entry = {
                'content': r['content'],
                'score': round(r['score'], 4),
                'type': r['type'],
                'similarity': round(r['similarity'], 4),
                # Always surface confidence tier — agents need this to
                # know when a result is trustworthy vs speculative
                'confidence_tier': r.get('confidence_tier', 'medium'),
            }
            # Include time_context if present (helps agent reason about recency)
            ep_ctx = r.get('context', {})
            if ep_ctx.get('time_context'):
                entry['time_context'] = ep_ctx['time_context']
            
            if r['type'] == 'episodic':
                entry['importance'] = r.get('importance', 0)
                entry['age_hours'] = round(
                    (time.time() - r.get('created_at', time.time())) / 3600, 1
                )
            elif r['type'] == 'semantic':
                entry['confidence'] = r.get('confidence', 0)
                entry['evidence_count'] = r.get('evidence_count', 0)
                if r.get('contradictions'):
                    entry['has_contradictions'] = True
            
            simplified.append(entry)
        
        return simplified
    
    def get_context_window(self, query: str = None, max_tokens: int = 2000) -> str:
        """
        Get a formatted context string suitable for injection into
        an LLM prompt. This is the primary way agents use ARN.
        
        Args:
            query: Optional query to focus the context retrieval
            max_tokens: Approximate token budget (~4 chars per token)
        
        Returns:
            Formatted string of relevant memories
        """
        char_budget = max_tokens * 4
        
        # Get working memory contents
        wm_items = self._arn.working_memory.get_active()
        
        # Get relevant long-term memories
        if query:
            lt_results = self.recall(query, top_k=10)
        else:
            # No query — return most recent/important
            lt_results = self.recall("recent important information", top_k=10)
        
        # Build context string
        parts = []
        current_chars = 0
        
        # Working memory (most recent active context)
        if wm_items:
            parts.append("## Active Context (Working Memory)")
            for slot in wm_items[:5]:
                line = f"- {slot.content}"
                if current_chars + len(line) > char_budget:
                    break
                parts.append(line)
                current_chars += len(line)
        
        # Long-term memories
        if lt_results:
            parts.append("\n## Relevant Memories")
            for r in lt_results:
                prefix = "📌" if r['type'] == 'semantic' else "💭"
                score_str = f"[{r['score']:.2f}]"
                line = f"{prefix} {score_str} {r['content']}"
                if r.get('time_context'):
                    line += f" ({r['time_context']})"
                if r.get('has_contradictions') or r.get('contradictions'):
                    line += " ⚠️ (contradicting info exists)"
                
                if current_chars + len(line) > char_budget:
                    break
                parts.append(line)
                current_chars += len(line)
        
        return "\n".join(parts)
    


    # ===========================================
    # HUMAN MEMORY CORE HELPERS
    # ===========================================

    def _recent_by_type(self, memory_type: str, scopes: list[str] = None,
                        limit: int = 20) -> list[dict]:
        """Return recent episodes tagged with a memory_type and scope."""
        scopes = scopes or ["global"]
        rows = []
        for ep in self._arn.storage.get_all_episodes(consolidated=None):
            ctx = ep.get("context") or {}
            if ctx.get("memory_type", "episode") != memory_type:
                continue
            if not _scope_matches(ctx.get("scope", "global"), scopes):
                continue
            row = dict(ep)
            row["memory_type"] = ctx.get("memory_type", "episode")
            row["scope"] = ctx.get("scope", "global")
            row["priority"] = ctx.get("priority", "normal")
            row["name"] = ctx.get("name")
            rows.append(row)
        rows.sort(key=lambda r: (PRIORITY_WEIGHT.get(r.get("priority", "normal"), 0.5),
                                 r.get("importance", 0), r.get("created_at", 0)), reverse=True)
        return rows[:limit]

    def _recall_by_type(self, query: str, memory_type: str, scopes: list[str],
                        top_k: int = 5) -> list[dict]:
        """Semantic recall, then filter by ARN memory_type/scope."""
        raw = self._arn.recall(query=query or memory_type, top_k=max(top_k * 5, 10),
                               include_episodic=True, include_semantic=False)
        out = []
        for r in raw:
            ctx = r.get("context") or {}
            if ctx.get("memory_type", "episode") != memory_type:
                continue
            if not _scope_matches(ctx.get("scope", "global"), scopes):
                continue
            item = dict(r)
            item["memory_type"] = memory_type
            item["scope"] = ctx.get("scope", "global")
            item["priority"] = ctx.get("priority", "normal")
            item["name"] = ctx.get("name")
            out.append(item)
        return out[:top_k]

    def set_identity(self, agent: str, name: str = None, role: str = None,
                     must: list[str] = None, must_not: list[str] = None) -> dict:
        agent = agent.strip()
        lines = [f"Identity for agent '{agent}'."]
        if name:
            lines.append(f"Name: {name}.")
        if role:
            lines.append(f"Role: {role}.")
        for item in must or []:
            lines.append(f"Must: {item}.")
        for item in must_not or []:
            lines.append(f"Must not: {item}.")
        return self.store("\n".join(lines), importance=0.95, source="identity",
                          tags=["identity", agent], memory_type="identity",
                          scope=f"agent:{agent}", priority="critical", name=agent)

    def get_identity(self, agent: str) -> list[dict]:
        return self._recent_by_type("identity", scopes=[f"agent:{agent}"], limit=5)

    def add_rule(self, agent: str, rule: str, priority: str = "high") -> dict:
        scope = "global" if agent in ("global", "all", "*") else f"agent:{agent}"
        return self.store(rule, importance=PRIORITY_WEIGHT.get(priority, 0.75),
                          source="rule", tags=["rule", agent], memory_type="rule",
                          scope=scope, priority=priority)

    def list_rules(self, agent: str = None) -> list[dict]:
        scopes = ["global"]
        if agent and agent not in ("global", "all", "*"):
            scopes.append(f"agent:{agent}")
        return self._recent_by_type("rule", scopes=scopes, limit=50)

    def add_procedure(self, name: str, steps: list[str], success: str = None,
                      agent: str = None) -> dict:
        lines = [f"Procedure: {name}"]
        for idx, step in enumerate(steps or [], 1):
            lines.append(f"Step {idx}: {step}")
        if success:
            lines.append(f"Success criteria: {success}")
        scope = f"agent:{agent}" if agent else "global"
        return self.store("\n".join(lines), importance=0.85, source="procedure",
                          tags=["procedure", name], memory_type="procedure",
                          scope=scope, priority="high", name=name)

    def recall_procedures(self, query: str, agent: str = None, top_k: int = 5) -> list[dict]:
        scopes = ["global"] + ([f"agent:{agent}"] if agent else [])
        return self._recall_by_type(query, "procedure", scopes=scopes, top_k=top_k)

    def add_error_lesson(self, mistake: str, fix: str = None, lesson: str = None,
                         agent: str = None, task: str = None) -> dict:
        lines = []
        if task:
            lines.append(f"Task: {task}")
        lines.append(f"Mistake: {mistake}")
        if fix:
            lines.append(f"Fix: {fix}")
        if lesson:
            lines.append(f"Lesson: {lesson}")
        scope = f"agent:{agent}" if agent else "global"
        return self.store("\n".join(lines), importance=0.8, source="error",
                          tags=["error", "lesson"] + ([agent] if agent else []),
                          memory_type="error", scope=scope, priority="high")

    def list_error_lessons(self, agent: str = None, query: str = None, limit: int = 10) -> list[dict]:
        scopes = ["global"] + ([f"agent:{agent}"] if agent else [])
        if query:
            return self._recall_by_type(query, "error", scopes=scopes, top_k=limit)
        return self._recent_by_type("error", scopes=scopes, limit=limit)

    def share_memory(self, content: str, to_agents: list[str], from_agent: str = None,
                     task: str = None, importance: float = 0.75,
                     priority: str = "high", tags: list[str] = None) -> dict:
        """
        Share useful task knowledge with other agents.

        This intentionally keeps cross-agent communication simple: ARN duplicates
        the shared note into each recipient agent's own memory namespace. That
        avoids a central message broker while still letting each agent recall the
        note naturally before it replies.
        """
        from_agent = (from_agent or self.agent_id).strip()
        cleaned = []
        for agent in to_agents or []:
            agent = (agent or "").strip()
            if agent and agent not in cleaned:
                cleaned.append(agent)
        if not cleaned:
            raise ValueError("to_agents must include at least one target agent")

        lines = [f"Shared memory from agent '{from_agent}'."]
        if task:
            lines.append(f"Task: {task}")
        lines.append(f"Content: {content}")
        shared_text = "\n".join(lines)
        base_tags = ["shared", "cross_agent", from_agent] + (tags or [])

        deliveries = []

        # Sender outbox copy. This makes it visible in the sender history.
        sender_result = self.store(
            content=shared_text,
            importance=importance,
            tags=base_tags + ["outbox"],
            source="cross_agent",
            context={
                "cross_agent": True,
                "direction": "outbox",
                "from_agent": from_agent,
                "to_agents": cleaned,
                "task": task,
            },
            memory_type="shared",
            scope=f"agent:{from_agent}",
            priority=priority,
            name=task or "cross-agent-share",
        )
        deliveries.append({"agent": from_agent, "direction": "outbox", "episode_id": sender_result["episode_id"]})

        # Recipient inbox copies. Keeping a local copy per recipient means
        # normal recall/context commands work even if only one agent is loaded.
        for target in cleaned:
            if target == self.agent_id:
                target_plugin = self
                close_after = False
            else:
                target_plugin = ARNPlugin(
                    agent_id=target,
                    data_root=self.data_root,
                    embedding_tier=self.embedding_tier,
                    auto_consolidate=True,
                    consolidation_threshold=128,
                )
                close_after = True
            try:
                result = target_plugin.store(
                    content=shared_text,
                    importance=importance,
                    tags=base_tags + ["inbox", target],
                    source="cross_agent",
                    context={
                        "cross_agent": True,
                        "direction": "inbox",
                        "from_agent": from_agent,
                        "to_agents": cleaned,
                        "recipient_agent": target,
                        "task": task,
                    },
                    memory_type="shared",
                    scope=f"agent:{target}",
                    priority=priority,
                    name=task or "cross-agent-share",
                )
                deliveries.append({"agent": target, "direction": "inbox", "episode_id": result["episode_id"]})
            finally:
                if close_after:
                    target_plugin.shutdown()

        return {
            "ok": True,
            "from_agent": from_agent,
            "to_agents": cleaned,
            "task": task,
            "deliveries": deliveries,
        }

    def list_shared_memories(self, agent: str = None, direction: str = None,
                             query: str = None, task: str = None,
                             limit: int = 20) -> list[dict]:
        """List or recall cross-agent shared memories for an agent."""
        agent = agent or self.agent_id
        scopes = ["global", f"agent:{agent}"]
        # If a task is provided, prefer deterministic task filtering over
        # semantic search. This ensures task handoffs are not missed when an
        # install is running in degraded/no-embedding mode.
        if task:
            rows = self._recent_by_type("shared", scopes=scopes, limit=max(limit * 5, 50))
            rows = [r for r in rows if (r.get("context") or {}).get("task") == task or task.lower() in r.get("content", "").lower()]
        else:
            q = query or ""
            rows = self._recall_by_type(q, "shared", scopes=scopes, top_k=limit) if q else self._recent_by_type("shared", scopes=scopes, limit=limit)
        if direction:
            rows = [r for r in rows if (r.get("context") or {}).get("direction") == direction]
        return rows[:limit]

    def build_context_packet(self, query: str = None, agent: str = None,
                             task: str = None, max_tokens: int = 2000) -> str:
        """Build a simple, human-like context packet for an agent step."""
        agent = agent or self.agent_id
        scopes = ["global", f"agent:{agent}"]
        q = " ".join(x for x in [task, query] if x) or "current task"
        char_budget = max_tokens * 4
        parts: list[str] = []

        def add_section(title: str, lines: list[str]) -> None:
            nonlocal parts, char_budget
            if not lines:
                return
            block = [f"## {title}"] + lines
            text = "\n".join(block)
            if sum(len(x) for x in parts) + len(text) <= char_budget:
                parts.extend(block)
                parts.append("")

        identities = [f"- {m['content']}" for m in self.get_identity(agent)[:2]]
        add_section("Identity", identities)

        rules = [f"- [{m.get('priority','normal')}] {m['content']}" for m in self.list_rules(agent)[:8]]
        add_section("Rules", rules)

        if task:
            add_section("Current Task", [f"- {task}"])

        shared = [f"- {m['content']}" for m in self.list_shared_memories(agent=agent, query=q, limit=5)]
        add_section("Shared Agent Notes", shared)

        procedures = [f"- {m['content']}" for m in self.recall_procedures(q, agent=agent, top_k=3)]
        add_section("Relevant Procedures", procedures)

        errors = [f"- {m['content']}" for m in self.list_error_lessons(agent=agent, query=q, limit=3)]
        add_section("Past Errors / Lessons", errors)

        prefs = [f"- {m['content']}" for m in self._recall_by_type(q, "preference", scopes, top_k=4)]
        add_section("Relevant Preferences", prefs)

        facts = [f"- {m['content']}" for m in self._recall_by_type(q, "fact", scopes, top_k=4)]
        add_section("Relevant Facts", facts)

        # Fall back to normal semantic context if typed memory is empty.
        if not parts:
            return self.get_context_window(query=query or task, max_tokens=max_tokens)
        return "\n".join(parts).strip()

    # ===========================================
    # OPENCLAW / CLI HOOK HELPERS
    # ===========================================

    def on_message_received(self, message: str, role: str = "user",
                            importance: float = 0.5,
                            time_context: str = "current") -> dict:
        """
        Hook for inbound messages. Use this when OpenClaw receives a user,
        tool, or system message and you want ARN to remember the useful parts.
        """
        return self.store(
            content=message,
            importance=importance,
            tags=["hook", "received", role],
            source=role,
            context={"hook": "message_received", "role": role},
            time_context=time_context,
        )

    def before_reply(self, query: str, max_tokens: int = 1000, task: str = None) -> str:
        """
        Hook before the agent replies. Returns a ready-to-inject memory block
        based on the current user request or task topic.
        """
        return self.build_context_packet(query=query, task=task, max_tokens=max_tokens)

    def on_message_sent(self, message: str, role: str = "assistant",
                        importance: float = 0.4,
                        time_context: str = "current") -> dict:
        """
        Hook for outbound agent messages. Store decisions, summaries, or task
        results sent by the agent. Avoid storing every small chat reply.
        """
        return self.store(
            content=message,
            importance=importance,
            tags=["hook", "sent", role],
            source=role,
            context={"hook": "message_sent", "role": role},
            time_context=time_context,
        )

    def on_tool_result(self, tool_name: str, result: str,
                       importance: float = 0.6,
                       time_context: str = "current") -> dict:
        """
        Hook for command/tool output. Store important diagnostics, fixes,
        file paths, errors, and completed actions.
        """
        return self.store(
            content=f"Tool result from {tool_name}: {result}",
            importance=importance,
            tags=["hook", "tool", tool_name],
            source="tool",
            context={"hook": "tool_result", "tool_name": tool_name},
            time_context=time_context,
        )

    # ===========================================
    # PRIVACY CONTROLS
    # ===========================================

    def list_memories(self, limit: int = 100, memory_type: str = "all") -> list[dict]:
        """
        Return recent memories in a user-readable format.
        memory_type: 'all', 'episodic', or 'semantic'
        """
        results = []
        if memory_type in ("all", "episodic"):
            for ep in self._arn.storage.get_all_episodes(consolidated=None, limit=limit):
                results.append({
                    "id": ep["id"],
                    "type": "episodic",
                    "content": ep["content"],
                    "importance": ep["importance"],
                    "created_at": ep["created_at"],
                    "source": ep.get("source", "unknown"),
                    "tags": ep.get("context", {}).get("tags", []),
                })
        if memory_type in ("all", "semantic"):
            for sem in self._arn.storage.get_all_semantics():
                results.append({
                    "id": sem["id"],
                    "type": "semantic",
                    "content": sem.get("schema", {}).get("representative_content", sem["concept_label"]),
                    "confidence": sem["confidence"],
                    "evidence_count": sem["evidence_count"],
                    "created_at": sem["created_at"],
                })
        results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return results[:limit]

    def forget(self, episode_ids: list[int] = None, forget_all: bool = False) -> dict:
        """
        Delete specific memories by ID, or wipe everything.
        Returns how many were deleted.
        """
        if forget_all:
            eps = self._arn.storage.get_all_episodes(consolidated=None)
            sems = self._arn.storage.get_all_semantics()
            if eps:
                self._arn.storage.delete_episodes([e["id"] for e in eps])
            if sems:
                self._arn.storage.delete_semantics([s["id"] for s in sems])
            return {"deleted_episodes": len(eps), "deleted_semantic": len(sems), "ok": True}
        if episode_ids:
            self._arn.storage.delete_episodes(episode_ids)
            return {"deleted_episodes": len(episode_ids), "ok": True}
        return {"deleted_episodes": 0, "ok": True}

    def search_memories(self, text: str, limit: int = 20) -> list[dict]:
        """Simple keyword search across memory content (no embeddings needed)."""
        needle = text.lower()
        results = []
        for ep in self._arn.storage.get_all_episodes(consolidated=None):
            if needle in ep["content"].lower():
                results.append({
                    "id": ep["id"],
                    "type": "episodic",
                    "content": ep["content"],
                    "importance": ep["importance"],
                    "created_at": ep["created_at"],
                })
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return results[:limit]


    # ===========================================
    # FACT GRAPH QUERIES
    # ===========================================

    def graph_query(self, entity: str, include_superseded: bool = False) -> list[dict]:
        """
        Return all known facts about an entity from the fact graph.
        Much faster than semantic recall for specific entity lookups.
        """
        return self._graph.get_facts_about(entity, include_superseded=include_superseded)

    def fact_history(self, subject: str, relation: str) -> list[dict]:
        """
        Full history of how a fact evolved. Includes superseded entries.
        Answers: "what did the user prefer before they switched?"
        """
        all_facts = self._graph.get_facts_about(subject, include_superseded=True)
        relevant = [f for f in all_facts
                    if f["subject"] == subject.lower() and f["relation"] == relation.lower()]
        relevant.sort(key=lambda f: f["created_at"], reverse=True)
        return relevant

    def graph_summary(self) -> dict:
        """Return a summary of the fact graph for monitoring."""
        return self._graph.summary()

    def list_contradictions(self, limit: int = 50) -> list[dict]:
        """Return stored contradiction candidates from episodic metadata."""
        found: list[dict] = []
        for ep in self._arn.storage.get_all_episodes(consolidated=None):
            ctx = ep.get('context') or {}
            for item in ctx.get('contradiction_candidates', []) or []:
                enriched = dict(item)
                enriched['new_episode_id'] = ep.get('id')
                enriched['new_content'] = ep.get('content')
                enriched['new_time_context'] = ctx.get('time_context', 'current')
                enriched['new_created_at'] = ep.get('created_at')
                found.append(enriched)
        found.sort(key=lambda x: x.get('new_created_at') or 0, reverse=True)
        return found[:limit]

    def semantic_selftest(self) -> dict:
        """
        Verify the core promise: store a fact and recall it by meaning, not
        keyword overlap. Requires a real embedding model; degraded hash vectors
        fail clearly.
        """
        before = self.store(
            "Mohamed prefers Python",
            importance=0.95,
            tags=["selftest", "preference", "coding"],
            source="selftest",
            time_context="current",
        )
        results = self.recall("what does the user like to code in?", top_k=3)
        top = results[0] if results else None
        ok = bool(top and "python" in top.get("content", "").lower() and top.get("confidence_tier") in {"medium", "high"})
        return {
            "ok": ok,
            "embedding_degraded": self._arn.embedder.is_degraded,
            "stored": before,
            "query": "what does the user like to code in?",
            "top_result": top,
            "results": results,
            "explanation": "This passes only when semantic embeddings connect 'prefers Python' with 'like to code in'.",
        }

    def maintain(self):
        """
        Run maintenance tasks. Call this during idle periods.
        - Consolidation (episodic → semantic)
        - Working memory decay
        """
        # Consolidate if enough unconsolidated episodes
        stats = self._arn.consolidate()
        
        # Decay working memory
        elapsed = time.time() - self._last_maintain
        self._arn.working_memory.decay(elapsed_seconds=elapsed)
        self._last_maintain = time.time()
        
        return stats
    
    def get_stats(self) -> dict:
        """Get system statistics for monitoring."""
        stats = self._arn.get_stats()
        stats['agent_id'] = self.agent_id
        stats['fact_graph'] = self._graph.summary()
        return stats
    
    def shutdown(self):
        """Clean shutdown — persist all state."""
        self._arn.close()
        logger.info(f"ARN plugin shut down for agent '{self.agent_id}'")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()

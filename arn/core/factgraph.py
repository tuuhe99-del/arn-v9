"""
ARN Fact Graph
==============
Lightweight entity-relation-entity graph built from extracted claims.
Uses networkx (stdlib-weight, no server needed) stored as JSON in SQLite.

This closes the biggest architectural gap vs Cognee/Zep: ARN can now
answer "what do we know about X?" through graph traversal, not just
vector similarity.

Design:
  - Each Claim (subject → relation → object) becomes a directed edge.
  - Nodes are entity strings. Edges carry relation, timestamp, episode_id,
    confidence, and superseded flag.
  - Graph is persisted as JSON in the system_state table so it travels
    with the agent's SQLite DB — no separate file, no Neo4j.
  - Kept intentionally simple: no inference, no ontology, no embeddings.
    It's a structured index on top of what the claim extractor already does.
"""
from __future__ import annotations

import json
import time
import logging
from typing import Optional

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

logger = logging.getLogger("arn.factgraph")

_STATE_KEY = "fact_graph_v1"


class FactGraph:
    """
    In-memory networkx DiGraph, persisted to ARN's SQLite via system_state.
    One instance lives per ARNPlugin (per agent).
    """

    def __init__(self, storage):
        self._storage = storage
        self._graph: Optional[object] = None  # nx.DiGraph when available
        self._available = _NX_AVAILABLE
        if self._available:
            self._load()
        else:
            logger.info("networkx not installed — fact graph disabled. pip install networkx")

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self):
        raw = self._storage.get_state(_STATE_KEY)
        if raw:
            try:
                data = json.loads(raw)
                self._graph = nx.node_link_graph(data, edges="edges")
                return
            except Exception as e:
                logger.warning(f"Could not restore fact graph: {e}. Starting fresh.")
        self._graph = nx.DiGraph()

    def _save(self):
        if not self._available or self._graph is None:
            return
        try:
            data = nx.node_link_data(self._graph, edges="edges")
            self._storage.set_state(_STATE_KEY, json.dumps(data))
        except Exception as e:
            logger.warning(f"Could not persist fact graph: {e}")

    # ── Write ─────────────────────────────────────────────────────────────

    def add_claim(self, subject: str, relation: str, obj: str,
                  episode_id: int, confidence: float = 0.7,
                  superseded: bool = False):
        """Add or update an edge for a claim. Saves immediately."""
        if not self._available:
            return
        now = time.time()
        s, o = subject.lower().strip(), obj.lower().strip()
        self._graph.add_node(s)
        self._graph.add_node(o)
        # Each edge is keyed by (s, o, relation) — store as edge attributes.
        # networkx DiGraph allows one edge per (u,v) pair; we embed relation
        # in a list so multiple relations between same nodes are preserved.
        existing = self._graph.edges.get((s, o), {})
        rels = existing.get("relations", [])
        # Remove old entry for this relation if it exists
        rels = [r for r in rels if r["relation"] != relation]
        rels.append({
            "relation": relation,
            "episode_id": episode_id,
            "confidence": confidence,
            "created_at": now,
            "superseded": superseded,
        })
        self._graph.add_edge(s, o, relations=rels, last_updated=now)
        self._save()

    def mark_superseded(self, subject: str, relation: str, obj: str):
        """Flag an edge relation as superseded when a newer fact replaces it."""
        if not self._available:
            return
        s, o = subject.lower().strip(), obj.lower().strip()
        edge = self._graph.edges.get((s, o))
        if not edge:
            return
        rels = edge.get("relations", [])
        for r in rels:
            if r["relation"] == relation:
                r["superseded"] = True
        self._graph.edges[s, o]["relations"] = rels
        self._save()

    # ── Query ─────────────────────────────────────────────────────────────

    def get_facts_about(self, entity: str, include_superseded: bool = False) -> list[dict]:
        """
        Return all current facts where entity is the subject or object.
        By default excludes superseded facts.
        """
        if not self._available or self._graph is None:
            return []
        entity = entity.lower().strip()
        results = []
        # Outgoing edges (entity is subject)
        for _, obj, data in self._graph.out_edges(entity, data=True):
            for rel in data.get("relations", []):
                if rel.get("superseded") and not include_superseded:
                    continue
                results.append({
                    "subject": entity,
                    "relation": rel["relation"],
                    "object": obj,
                    "confidence": rel["confidence"],
                    "episode_id": rel["episode_id"],
                    "created_at": rel["created_at"],
                    "superseded": rel.get("superseded", False),
                })
        # Incoming edges (entity is object)
        for subj, _, data in self._graph.in_edges(entity, data=True):
            for rel in data.get("relations", []):
                if rel.get("superseded") and not include_superseded:
                    continue
                results.append({
                    "subject": subj,
                    "relation": rel["relation"],
                    "object": entity,
                    "confidence": rel["confidence"],
                    "episode_id": rel["episode_id"],
                    "created_at": rel["created_at"],
                    "superseded": rel.get("superseded", False),
                })
        results.sort(key=lambda r: r["created_at"], reverse=True)
        return results

    def get_related_entities(self, entity: str, max_hops: int = 2) -> list[str]:
        """
        Return entities reachable from entity within max_hops.
        Useful for expanding a recall query to related nodes.
        """
        if not self._available or self._graph is None:
            return []
        entity = entity.lower().strip()
        if entity not in self._graph:
            return []
        try:
            reachable = nx.single_source_shortest_path_length(
                self._graph, entity, cutoff=max_hops
            )
            return [n for n in reachable if n != entity]
        except Exception:
            return []

    def summary(self) -> dict:
        if not self._available or self._graph is None:
            return {"available": False}
        return {
            "available": True,
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "entities": list(self._graph.nodes)[:20],  # sample
        }

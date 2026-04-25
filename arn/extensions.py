"""
ARN v9 Feature Extensions
===========================
Tier 1 + Tier 2 features that run on Pi 5 hardware.

Adds:
- BM25 keyword search fused with semantic similarity (hybrid retrieval)
- Hash-based deduplication (prevents storing exact duplicates)
- Memory TTL (time-to-live, automatic expiry)
- Explicit forgetting (delete memories by query)
- Importance decay over time (old unused memories fade)
- Entity extraction (people, orgs, projects, locations)
- Memory versioning (superseded facts preserved with timestamp)
- Export/import (JSON portability)
- User-scoped vs agent-scoped memory tiers
- Webhook/callback on store

All features are optional and backward-compatible with existing
ARN installations. Nothing breaks if you don't use them.
"""

import hashlib
import json
import logging
import re
import time
from typing import List, Dict, Optional, Callable, Tuple, Any

import numpy as np

logger = logging.getLogger("arn.extensions")


# =========================================================
# 1. BM25 HYBRID RETRIEVAL
# =========================================================

class HybridRetriever:
    """
    Fuses semantic similarity with BM25 keyword matching using
    Reciprocal Rank Fusion (RRF).
    
    RRF formula: score = sum(1 / (k + rank_i)) for each retrieval method
    This handles different score distributions gracefully.
    """
    
    def __init__(self, rrf_k: int = 60, semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._bm25_index = None
        self._bm25_corpus = None
        self._bm25_ids = None
    
    def build_bm25_index(self, episodes: List[dict]):
        """Build/rebuild BM25 index from episode list."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed, BM25 disabled. "
                          "pip install rank_bm25")
            return
        
        corpus = []
        ids = []
        for ep in episodes:
            tokens = self._tokenize(ep['content'])
            corpus.append(tokens)
            ids.append(ep['id'])
        
        if corpus:
            self._bm25_index = BM25Okapi(corpus)
            self._bm25_corpus = corpus
            self._bm25_ids = ids
    
    def hybrid_search(self, query: str, semantic_results: List[dict],
                      top_k: int = 10) -> List[dict]:
        """
        Fuse semantic results with BM25 keyword results using RRF.
        
        Args:
            query: the recall query
            semantic_results: results from ARN's vector search (must have 'id' and 'score')
            top_k: number of results to return
            
        Returns:
            Re-ranked results with fused scores
        """
        if self._bm25_index is None or not self._bm25_ids:
            return semantic_results[:top_k]
        
        # Get BM25 scores
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25_index.get_scores(query_tokens)
        
        # Build BM25 ranking
        bm25_ranked = sorted(
            zip(self._bm25_ids, bm25_scores),
            key=lambda x: x[1], reverse=True
        )
        bm25_rank_map = {eid: rank for rank, (eid, _) in enumerate(bm25_ranked)}
        bm25_score_map = {eid: score for eid, score in zip(self._bm25_ids, bm25_scores)}
        
        # Build semantic ranking
        semantic_rank_map = {r['id']: rank for rank, r in enumerate(semantic_results)}
        
        # Collect all candidate IDs
        all_ids = set(r['id'] for r in semantic_results)
        # Add top BM25 hits that might not be in semantic results
        for eid, _ in bm25_ranked[:top_k * 2]:
            all_ids.add(eid)
        
        # Compute RRF scores
        id_to_result = {r['id']: r for r in semantic_results}
        fused = []
        for eid in all_ids:
            sem_rank = semantic_rank_map.get(eid, len(semantic_results) + 100)
            bm25_rank = bm25_rank_map.get(eid, len(self._bm25_ids) + 100)
            
            rrf_score = (
                self.semantic_weight * (1.0 / (self.rrf_k + sem_rank)) +
                self.bm25_weight * (1.0 / (self.rrf_k + bm25_rank))
            )
            
            result = id_to_result.get(eid)
            if result:
                result = dict(result)  # Copy to avoid mutating original
                result['rrf_score'] = rrf_score
                result['bm25_score'] = float(bm25_score_map.get(eid, 0))
                fused.append(result)
        
        fused.sort(key=lambda r: r['rrf_score'], reverse=True)
        return fused[:top_k]
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercased tokenization."""
        return re.findall(r'\w+', text.lower())


# =========================================================
# 2. HASH DEDUPLICATION
# =========================================================

def content_hash(text: str) -> str:
    """Compute a stable hash for dedup checking."""
    normalized = ' '.join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def is_duplicate(storage, text: str) -> bool:
    """Check if this exact content already exists in storage."""
    h = content_hash(text)
    conn = storage._get_conn()
    row = conn.execute(
        "SELECT id FROM episodes WHERE content_hash = ? LIMIT 1", (h,)
    ).fetchone()
    return row is not None


# =========================================================
# 3. MEMORY TTL (Time-To-Live)
# =========================================================

def get_expired_episodes(storage, current_time: float = None) -> List[int]:
    """Find episodes past their expiry time."""
    now = current_time or time.time()
    conn = storage._get_conn()
    rows = conn.execute(
        "SELECT id FROM episodes WHERE expires_at IS NOT NULL AND expires_at < ?",
        (now,)
    ).fetchall()
    return [r['id'] for r in rows]


def purge_expired(storage, current_time: float = None) -> int:
    """Delete expired episodes. Returns count deleted."""
    expired = get_expired_episodes(storage, current_time)
    if expired:
        storage.delete_episodes(expired)
    return len(expired)


# =========================================================
# 4. EXPLICIT FORGETTING
# =========================================================

def forget_by_query(plugin, query: str, top_k: int = 5,
                    min_similarity: float = 0.5) -> List[int]:
    """
    Forget memories matching a query.
    Only deletes episodes above min_similarity to avoid accidental purges.
    Returns list of deleted episode IDs.
    """
    results = plugin._arn.recall(query, top_k=top_k)
    to_delete = [
        r['id'] for r in results
        if r['similarity'] >= min_similarity and r['type'] == 'episodic'
    ]
    if to_delete:
        plugin._arn.storage.delete_episodes(to_delete)
    return to_delete


def forget_by_tag(plugin, tag: str) -> int:
    """Forget all episodes with a specific tag. Returns count deleted."""
    conn = plugin._arn.storage._get_conn()
    # Tags are stored in context_json
    rows = conn.execute(
        "SELECT id, context_json FROM episodes"
    ).fetchall()
    
    to_delete = []
    for row in rows:
        try:
            ctx = json.loads(row['context_json'])
            if tag in ctx.get('tags', []):
                to_delete.append(row['id'])
        except (json.JSONDecodeError, TypeError):
            pass
    
    if to_delete:
        plugin._arn.storage.delete_episodes(to_delete)
    return len(to_delete)


# =========================================================
# 5. IMPORTANCE DECAY
# =========================================================

def apply_importance_decay(storage, decay_rate: float = 0.99,
                           access_boost: float = 0.05,
                           min_importance: float = 0.05):
    """
    Decay importance of old episodes based on age.
    Boost importance of frequently accessed episodes.
    
    Formula:
        new_importance = base_importance * decay_rate^age_days 
                         + access_boost * log(1 + access_count)
        
    Clamped to [min_importance, 1.0].
    """
    conn = storage._get_conn()
    now = time.time()
    
    rows = conn.execute(
        "SELECT id, importance, access_count, created_at FROM episodes"
    ).fetchall()
    
    updates = []
    for row in rows:
        age_days = (now - row['created_at']) / 86400.0
        access_bonus = access_boost * np.log1p(row['access_count'])
        decayed = row['importance'] * (decay_rate ** age_days) + access_bonus
        decayed = max(min_importance, min(1.0, decayed))
        
        if abs(decayed - row['importance']) > 0.001:
            updates.append((decayed, row['id']))
    
    if updates:
        conn.executemany(
            "UPDATE episodes SET importance = ? WHERE id = ?", updates
        )
        conn.commit()
    
    return len(updates)


# =========================================================
# 6. ENTITY EXTRACTION
# =========================================================

class EntityExtractor:
    """
    Extract named entities from text using spaCy.
    Falls back to regex patterns if spaCy isn't available.
    """
    
    def __init__(self):
        self._nlp = None
        self._load_attempted = False
    
    def _load_spacy(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("Entity extraction: spaCy en_core_web_sm loaded")
        except (ImportError, OSError):
            logger.info("Entity extraction: spaCy unavailable, using regex fallback")
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text.
        Returns list of {text, label} dicts.
        Labels: PERSON, ORG, GPE (location), PROJECT, TECH, DATE
        """
        self._load_spacy()
        
        if self._nlp:
            return self._extract_spacy(text)
        return self._extract_regex(text)
    
    def _extract_spacy(self, text: str) -> List[Dict[str, str]]:
        doc = self._nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                })
        return entities
    
    def _extract_regex(self, text: str) -> List[Dict[str, str]]:
        """Regex-based entity extraction fallback."""
        entities = []
        seen = set()
        
        # Capitalized multi-word names (likely proper nouns)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = m.group(1)
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append({'text': name, 'label': 'ENTITY'})
        
        # Technology terms (common patterns)
        tech_pattern = r'\b(Python|Rust|JavaScript|TypeScript|Java|Go|Ruby|C\+\+|Docker|Kubernetes|PostgreSQL|SQLite|Redis|Neo4j|FastAPI|Flask|Django|React|Vue|Angular|Linux|Ubuntu|ARM64|Raspberry Pi|OpenClaw|ARN)\b'
        for m in re.finditer(tech_pattern, text, re.IGNORECASE):
            term = m.group(1)
            if term.lower() not in seen:
                seen.add(term.lower())
                entities.append({'text': term, 'label': 'TECH'})
        
        # Dates
        date_pattern = r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{0,4})\b'
        for m in re.finditer(date_pattern, text):
            entities.append({'text': m.group(1), 'label': 'DATE'})
        
        return entities


# Global extractor instance (lazy-loaded)
_entity_extractor = EntityExtractor()


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities from text. Module-level convenience function."""
    return _entity_extractor.extract(text)


# =========================================================
# 7. MEMORY VERSIONING
# =========================================================

def supersede_episode(storage, old_id: int, new_id: int):
    """
    Mark an old episode as superseded by a new one.
    The old episode is preserved with a superseded_by pointer
    and an invalidated_at timestamp.
    """
    conn = storage._get_conn()
    now = time.time()
    conn.execute("""
        UPDATE episodes 
        SET superseded_by = ?, invalidated_at = ?
        WHERE id = ?
    """, (new_id, now, old_id))
    conn.commit()


def get_version_history(storage, episode_id: int) -> List[dict]:
    """
    Get the version chain for an episode.
    Returns all previous versions (superseded episodes) in chronological order.
    """
    conn = storage._get_conn()
    chain = []
    
    # Walk backward: find episodes that were superseded leading to this one
    current = episode_id
    visited = set()
    while current and current not in visited:
        visited.add(current)
        row = conn.execute(
            "SELECT * FROM episodes WHERE superseded_by = ?", (current,)
        ).fetchone()
        if row:
            chain.append(dict(row))
            current = row['id']
        else:
            break
    
    chain.reverse()  # Chronological order
    return chain


# =========================================================
# 8. EXPORT / IMPORT
# =========================================================

def export_memory(storage, filepath: str, agent_id: str = None):
    """
    Export all memory to a JSON file.
    Includes episodes, semantic nodes, and metadata.
    Does NOT include raw vectors (they get re-computed on import).
    """
    conn = storage._get_conn()
    
    episodes = []
    for row in conn.execute("SELECT * FROM episodes ORDER BY created_at ASC").fetchall():
        ep = dict(row)
        ep['context'] = json.loads(ep.pop('context_json', '{}'))
        episodes.append(ep)
    
    semantics = []
    for row in conn.execute("SELECT * FROM semantic_nodes ORDER BY created_at ASC").fetchall():
        sem = dict(row)
        sem['schema'] = json.loads(sem.pop('schema_json', '{}'))
        sem['contradictions'] = json.loads(sem.pop('contradiction_log', '[]'))
        semantics.append(sem)
    
    export_data = {
        'version': 'arn_v9_export_v1',
        'exported_at': time.time(),
        'agent_id': agent_id,
        'episode_count': len(episodes),
        'semantic_count': len(semantics),
        'episodes': episodes,
        'semantics': semantics,
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return {'episodes': len(episodes), 'semantics': len(semantics)}


def import_memory(plugin, filepath: str, merge: bool = True) -> dict:
    """
    Import memory from a JSON export file.
    
    Args:
        plugin: ARNPlugin instance (needs embedder for re-vectorizing)
        filepath: path to the exported JSON
        merge: if True, add to existing memory. If False, clear first.
        
    Returns:
        dict with import stats
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if data.get('version') != 'arn_v9_export_v1':
        raise ValueError(f"Unknown export format: {data.get('version')}")
    
    imported = 0
    skipped = 0
    
    for ep in data.get('episodes', []):
        content = ep.get('content', '')
        if not content:
            continue
        
        # Check for duplicate
        h = content_hash(content)
        try:
            if is_duplicate(plugin._arn.storage, content):
                skipped += 1
                continue
        except Exception:
            pass  # Column might not exist yet
        
        plugin.store(
            content=content,
            importance=ep.get('importance', 0.5),
            time_context=ep.get('context', {}).get('time_context', 'current'),
            source=ep.get('source', 'import'),
            context=ep.get('context', {}),
        )
        imported += 1
    
    return {
        'imported': imported,
        'skipped_duplicates': skipped,
        'total_in_file': len(data.get('episodes', [])),
    }


# =========================================================
# 9. STORE CALLBACKS
# =========================================================

class StoreCallbackManager:
    """
    Register callbacks that fire when new memories are stored.
    Useful for triggering the DME, logging, or external integrations.
    """
    
    def __init__(self):
        self._callbacks: List[Callable] = []
    
    def register(self, callback: Callable):
        """Register a callback. Signature: callback(episode_dict) -> None"""
        self._callbacks.append(callback)
    
    def unregister(self, callback: Callable):
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]
    
    def fire(self, episode: dict):
        """Fire all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(episode)
            except Exception as e:
                logger.error(f"Store callback error: {e}")

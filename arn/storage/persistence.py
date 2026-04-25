"""
ARN v9 Persistence Layer
=========================
SQLite for metadata + memory-mapped NumPy arrays for vectors.

Design choices:
- SQLite WAL mode for crash safety and concurrent reads
- Memory-mapped vectors for zero-copy access (OS handles paging)
- Atomic writes via temp-file-then-rename for vector files
- Batch operations to minimize SD card wear

Storage layout:
  {data_dir}/
    arn_metadata.db          # SQLite: all metadata
    episodic_vectors.npy     # memmap: N x 384 float32
    semantic_vectors.npy     # memmap: M x 384 float32
"""

import sqlite3
import numpy as np
import os
import shutil
import json
import time
import hashlib
import logging
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

from ..core.embeddings import EMBEDDING_DIM

logger = logging.getLogger("arn.storage")

# Schema version for migrations
SCHEMA_VERSION = 2


class StorageEngine:
    """
    Persistent storage backend for ARN v9.
    
    Handles:
    - Episode metadata and vectors
    - Semantic memory metadata and vectors  
    - System configuration and stats
    - Crash-safe writes with WAL mode
    """
    
    def __init__(self, data_dir: str, max_episodes: int = 4096,
                 max_semantics: int = 2048, embedding_dim: int = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "arn_metadata.db"
        self.episodic_vec_path = self.data_dir / "episodic_vectors.npy"
        self.semantic_vec_path = self.data_dir / "semantic_vectors.npy"
        
        self.max_episodes = max_episodes
        self.max_semantics = max_semantics
        
        # Dimension handling: use provided, else default to legacy constant.
        # If existing vector files exist with a different dim, respect THAT.
        if embedding_dim is None:
            embedding_dim = EMBEDDING_DIM
        
        # If vectors already exist on disk, infer the dim from them
        # to preserve backward compatibility with existing deployments.
        if self.episodic_vec_path.exists():
            try:
                existing = np.load(str(self.episodic_vec_path), mmap_mode='r')
                existing_dim = existing.shape[1]
                if existing_dim != embedding_dim:
                    logger.warning(
                        f"Existing vectors have dim={existing_dim} but engine "
                        f"configured for dim={embedding_dim}. Using existing dim "
                        f"to preserve data. Delete the data directory to switch models."
                    )
                    embedding_dim = existing_dim
                del existing  # Close the mmap before reopening below
            except Exception:
                pass
        
        self.embedding_dim = embedding_dim
        
        # Initialize database
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        
        # Initialize vector stores
        self._episodic_vectors: Optional[np.ndarray] = None
        self._semantic_vectors: Optional[np.ndarray] = None
        self._init_vectors()
        
        # Write buffer for batched operations
        self._pending_episode_writes: List[Tuple[int, np.ndarray]] = []
        self._pending_semantic_writes: List[Tuple[int, np.ndarray]] = []
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=2000")
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        """Create tables if they don't exist, migrate if needed."""
        conn = self._get_conn()
        
        # Create schema_version table first (needed to check version)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        conn.commit()
        
        # Check and run migrations BEFORE creating tables with new columns
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is not None and existing[0] < SCHEMA_VERSION:
            self._migrate_schema(conn, existing[0])
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
            conn.commit()
        
        # Now create all tables (safe for both fresh installs and migrated dbs)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT,
                context_json TEXT DEFAULT '{}',
                importance REAL DEFAULT 0.5,
                prediction_error REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                replay_priority REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                last_accessed REAL,
                consolidated INTEGER DEFAULT 0,
                source TEXT DEFAULT 'user',
                expires_at REAL,
                superseded_by INTEGER,
                invalidated_at REAL,
                user_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec_index INTEGER NOT NULL,
                concept_label TEXT NOT NULL,
                confidence REAL DEFAULT 0.1,
                evidence_count INTEGER DEFAULT 0,
                contradiction_log TEXT DEFAULT '[]',
                schema_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_episodes_importance 
                ON episodes(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_created 
                ON episodes(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_consolidated 
                ON episodes(consolidated);
            CREATE INDEX IF NOT EXISTS idx_episodes_hash
                ON episodes(content_hash);
            CREATE INDEX IF NOT EXISTS idx_episodes_expires
                ON episodes(expires_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_user
                ON episodes(user_id);
            CREATE INDEX IF NOT EXISTS idx_semantic_confidence 
                ON semantic_nodes(confidence DESC);
            
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                mention_count INTEGER DEFAULT 1,
                user_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS entity_episodes (
                entity_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                PRIMARY KEY (entity_id, episode_id),
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_entity_text
                ON entities(text);
            CREATE INDEX IF NOT EXISTS idx_entity_label
                ON entities(label);
        """)
        
        # Set schema version for fresh installs
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is None:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        
        conn.commit()
    
    def _migrate_schema(self, conn, from_version: int):
        """Migrate database schema from older versions."""
        if from_version < 2:
            # v1 → v2: add new columns to episodes, create entities tables
            migrations = [
                "ALTER TABLE episodes ADD COLUMN content_hash TEXT",
                "ALTER TABLE episodes ADD COLUMN expires_at REAL",
                "ALTER TABLE episodes ADD COLUMN superseded_by INTEGER",
                "ALTER TABLE episodes ADD COLUMN invalidated_at REAL",
                "ALTER TABLE episodes ADD COLUMN user_id TEXT",
            ]
            for sql in migrations:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  # Column may already exist
            
            # Backfill content_hash for existing episodes
            rows = conn.execute("SELECT id, content FROM episodes WHERE content_hash IS NULL").fetchall()
            for row in rows:
                normalized = ' '.join(row['content'].lower().split())
                h = hashlib.sha256(normalized.encode()).hexdigest()[:16]
                conn.execute("UPDATE episodes SET content_hash = ? WHERE id = ?", (h, row['id']))
            
            # Create indexes for new columns
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_expires ON episodes(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)",
            ]:
                conn.execute(idx_sql)
            
            logger.info(f"Migrated schema v1 → v2 ({len(rows)} episodes hash-backfilled)")
    
    def _init_vectors(self):
        """Initialize or load memory-mapped vector files."""
        # Episodic vectors
        if self.episodic_vec_path.exists():
            self._episodic_vectors = np.load(
                str(self.episodic_vec_path), mmap_mode='r+'
            )
            logger.info(f"Loaded episodic vectors: {self._episodic_vectors.shape}")
        else:
            self._episodic_vectors = np.zeros(
                (self.max_episodes, self.embedding_dim), dtype=np.float32
            )
            np.save(str(self.episodic_vec_path), self._episodic_vectors)
            self._episodic_vectors = np.load(
                str(self.episodic_vec_path), mmap_mode='r+'
            )
        
        # Semantic vectors
        if self.semantic_vec_path.exists():
            self._semantic_vectors = np.load(
                str(self.semantic_vec_path), mmap_mode='r+'
            )
            logger.info(f"Loaded semantic vectors: {self._semantic_vectors.shape}")
        else:
            self._semantic_vectors = np.zeros(
                (self.max_semantics, self.embedding_dim), dtype=np.float32
            )
            np.save(str(self.semantic_vec_path), self._semantic_vectors)
            self._semantic_vectors = np.load(
                str(self.semantic_vec_path), mmap_mode='r+'
            )
    
    # =========================================================
    # EPISODIC MEMORY OPERATIONS
    # =========================================================
    
    def store_episode(self, content: str, vector: np.ndarray,
                      context: dict = None, importance: float = 0.5,
                      prediction_error: float = 0.0,
                      source: str = 'user',
                      expires_at: float = None,
                      user_id: str = None) -> int:
        """Store a new episodic memory. Returns episode ID."""
        conn = self._get_conn()
        now = time.time()
        
        # Content hash for deduplication
        normalized = ' '.join(content.lower().split())
        c_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        # Find next available vector index
        # CRITICAL: Use MAX(vec_index)+1, NOT COUNT(*).
        # COUNT-based allocation causes collisions when episodes are
        # consolidated or deleted — new episodes get indices that
        # already belong to other episodes, overwriting their vectors.
        row = conn.execute("SELECT MAX(vec_index) FROM episodes").fetchone()
        vec_index = (row[0] + 1) if row[0] is not None else 0
        
        # Handle overflow
        if vec_index >= self.max_episodes:
            vec_index = self._find_free_episode_slot(conn)
        
        # Store vector
        if vec_index < self._episodic_vectors.shape[0]:
            self._episodic_vectors[vec_index] = vector
        else:
            logger.warning(f"Vector index {vec_index} out of bounds, expanding")
            self._expand_episodic_vectors()
            self._episodic_vectors[vec_index] = vector
        
        # Store metadata
        cursor = conn.execute("""
            INSERT INTO episodes (vec_index, content, content_hash, context_json, 
                                  importance, prediction_error, created_at, source,
                                  expires_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vec_index,
            content,
            c_hash,
            json.dumps(context or {}),
            importance,
            prediction_error,
            now,
            source,
            expires_at,
            user_id,
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_episode(self, episode_id: int) -> Optional[dict]:
        """Retrieve episode by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)
    
    def get_all_episodes(self, consolidated: Optional[bool] = None,
                         limit: int = None) -> List[dict]:
        """Retrieve episodes with optional filtering."""
        conn = self._get_conn()
        query = "SELECT * FROM episodes"
        params = []
        
        if consolidated is not None:
            query += " WHERE consolidated = ?"
            params.append(int(consolidated))
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(r) for r in rows]
    
    def get_episode_vectors(self, episode_ids: List[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Get vectors for episodes. Returns (vectors_matrix, vec_indices).
        If episode_ids is None, returns all unconsolidated episode vectors.
        """
        conn = self._get_conn()
        
        if episode_ids is None:
            rows = conn.execute(
                "SELECT id, vec_index FROM episodes WHERE consolidated=0"
            ).fetchall()
        else:
            placeholders = ','.join('?' * len(episode_ids))
            rows = conn.execute(
                f"SELECT id, vec_index FROM episodes WHERE id IN ({placeholders})",
                episode_ids
            ).fetchall()
        
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        
        indices = [r['vec_index'] for r in rows]
        ids = [r['id'] for r in rows]
        vectors = self._episodic_vectors[indices].copy()
        return vectors, ids
    
    def update_episode_access(self, episode_id: int):
        """Increment access count and update last_accessed."""
        conn = self._get_conn()
        conn.execute("""
            UPDATE episodes 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (time.time(), episode_id))
        conn.commit()
    
    def mark_episodes_consolidated(self, episode_ids: List[int]):
        """Mark episodes as consolidated."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        conn.execute(
            f"UPDATE episodes SET consolidated = 1 WHERE id IN ({placeholders})",
            episode_ids
        )
        conn.commit()
    
    def delete_episodes(self, episode_ids: List[int]):
        """Delete episodes permanently."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(episode_ids))
        conn.execute(
            f"DELETE FROM episodes WHERE id IN ({placeholders})",
            episode_ids
        )
        conn.commit()
    
    def count_episodes(self, consolidated: Optional[bool] = None) -> int:
        conn = self._get_conn()
        if consolidated is None:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated=?",
            (int(consolidated),)
        ).fetchone()[0]
    
    # =========================================================
    # SEMANTIC MEMORY OPERATIONS
    # =========================================================
    
    def store_semantic(self, concept_label: str, vector: np.ndarray,
                       confidence: float = 0.1, evidence_count: int = 1,
                       schema: dict = None) -> int:
        """Store a new semantic memory node. Returns node ID."""
        conn = self._get_conn()
        now = time.time()
        
        count = conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
        vec_index = count
        
        if vec_index >= self._semantic_vectors.shape[0]:
            self._expand_semantic_vectors()
        
        self._semantic_vectors[vec_index] = vector
        
        cursor = conn.execute("""
            INSERT INTO semantic_nodes (vec_index, concept_label, confidence,
                                       evidence_count, schema_json, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            vec_index, concept_label, confidence,
            evidence_count, json.dumps(schema or {}), now, now
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_semantic(self, node_id: int, vector: np.ndarray = None,
                        confidence: float = None, evidence_count: int = None,
                        contradiction_log: list = None, schema: dict = None):
        """Update an existing semantic node."""
        conn = self._get_conn()
        
        if vector is not None:
            row = conn.execute(
                "SELECT vec_index FROM semantic_nodes WHERE id=?", (node_id,)
            ).fetchone()
            if row:
                self._semantic_vectors[row['vec_index']] = vector
        
        updates = []
        params = []
        
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if evidence_count is not None:
            updates.append("evidence_count = ?")
            params.append(evidence_count)
        if contradiction_log is not None:
            updates.append("contradiction_log = ?")
            params.append(json.dumps(contradiction_log))
        if schema is not None:
            updates.append("schema_json = ?")
            params.append(json.dumps(schema))
        
        updates.append("last_updated = ?")
        params.append(time.time())
        params.append(node_id)
        
        conn.execute(
            f"UPDATE semantic_nodes SET {', '.join(updates)} WHERE id = ?",
            params
        )
        conn.commit()
    
    def get_all_semantics(self) -> List[dict]:
        """Retrieve all semantic nodes."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM semantic_nodes ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_semantic(r) for r in rows]
    
    def get_semantic_vectors(self) -> Tuple[np.ndarray, List[int]]:
        """Get all semantic vectors. Returns (matrix, node_ids)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, vec_index FROM semantic_nodes"
        ).fetchall()
        
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        
        indices = [r['vec_index'] for r in rows]
        ids = [r['id'] for r in rows]
        vectors = self._semantic_vectors[indices].copy()
        return vectors, ids
    
    def count_semantics(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
    
    def delete_semantics(self, node_ids: List[int]):
        """Delete semantic nodes."""
        conn = self._get_conn()
        placeholders = ','.join('?' * len(node_ids))
        conn.execute(
            f"DELETE FROM semantic_nodes WHERE id IN ({placeholders})",
            node_ids
        )
        conn.commit()
    
    # =========================================================
    # SYSTEM STATE
    # =========================================================
    
    def get_state(self, key: str, default: str = None) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ?", (key,)
        ).fetchone()
        return row['value'] if row else default
    
    def set_state(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()
    
    # =========================================================
    # ENTITY OPERATIONS
    # =========================================================
    
    def store_entity(self, text: str, label: str, episode_id: int,
                     user_id: str = None) -> int:
        """
        Store or update an entity mention. Links it to the source episode.
        Returns entity ID.
        """
        conn = self._get_conn()
        now = time.time()
        
        # Check if entity already exists (case-insensitive)
        row = conn.execute(
            "SELECT id, mention_count FROM entities WHERE LOWER(text) = LOWER(?)",
            (text,)
        ).fetchone()
        
        if row:
            entity_id = row['id']
            conn.execute("""
                UPDATE entities SET last_seen = ?, mention_count = mention_count + 1
                WHERE id = ?
            """, (now, entity_id))
        else:
            cursor = conn.execute("""
                INSERT INTO entities (text, label, first_seen, last_seen, mention_count, user_id)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (text, label, now, now, user_id))
            entity_id = cursor.lastrowid
        
        # Link entity to episode (ignore if already linked)
        conn.execute("""
            INSERT OR IGNORE INTO entity_episodes (entity_id, episode_id)
            VALUES (?, ?)
        """, (entity_id, episode_id))
        
        conn.commit()
        return entity_id
    
    def search_entities(self, text: str = None, label: str = None,
                        limit: int = 20) -> List[dict]:
        """Search entities by text and/or label."""
        conn = self._get_conn()
        query = "SELECT * FROM entities WHERE 1=1"
        params = []
        
        if text:
            query += " AND LOWER(text) LIKE LOWER(?)"
            params.append(f"%{text}%")
        if label:
            query += " AND label = ?"
            params.append(label)
        
        query += " ORDER BY mention_count DESC LIMIT ?"
        params.append(limit)
        
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    
    def get_entity_episodes(self, entity_id: int) -> List[dict]:
        """Get all episodes linked to an entity."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT e.* FROM episodes e
            JOIN entity_episodes ee ON e.id = ee.episode_id
            WHERE ee.entity_id = ?
            ORDER BY e.created_at DESC
        """, (entity_id,)).fetchall()
        return [self._row_to_episode(r) for r in rows]
    
    def get_episode_entities(self, episode_id: int) -> List[dict]:
        """Get all entities mentioned in an episode."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT ent.* FROM entities ent
            JOIN entity_episodes ee ON ent.id = ee.entity_id
            WHERE ee.episode_id = ?
        """, (episode_id,)).fetchall()
        return [dict(r) for r in rows]
    
    # =========================================================
    # INTERNAL HELPERS
    # =========================================================
    
    def _row_to_episode(self, row) -> dict:
        return {
            'id': row['id'],
            'vec_index': row['vec_index'],
            'content': row['content'],
            'context': json.loads(row['context_json']),
            'importance': row['importance'],
            'prediction_error': row['prediction_error'],
            'access_count': row['access_count'],
            'replay_priority': row['replay_priority'],
            'created_at': row['created_at'],
            'last_accessed': row['last_accessed'],
            'consolidated': bool(row['consolidated']),
            'source': row['source'],
            'content_hash': row['content_hash'] if 'content_hash' in row.keys() else None,
            'expires_at': row['expires_at'] if 'expires_at' in row.keys() else None,
            'superseded_by': row['superseded_by'] if 'superseded_by' in row.keys() else None,
            'invalidated_at': row['invalidated_at'] if 'invalidated_at' in row.keys() else None,
            'user_id': row['user_id'] if 'user_id' in row.keys() else None,
        }
    
    def _row_to_semantic(self, row) -> dict:
        return {
            'id': row['id'],
            'vec_index': row['vec_index'],
            'concept_label': row['concept_label'],
            'confidence': row['confidence'],
            'evidence_count': row['evidence_count'],
            'contradiction_log': json.loads(row['contradiction_log']),
            'schema': json.loads(row['schema_json']),
            'created_at': row['created_at'],
            'last_updated': row['last_updated'],
            'access_count': row['access_count'],
        }
    
    def _find_free_episode_slot(self, conn) -> int:
        """Find a reusable vector slot from consolidated/deleted episodes."""
        # Find lowest-importance consolidated episode and reuse its slot
        row = conn.execute("""
            SELECT vec_index FROM episodes 
            WHERE consolidated = 1 
            ORDER BY importance ASC LIMIT 1
        """).fetchone()
        if row:
            return row['vec_index']
        # Worst case: reuse slot 0
        return 0
    
    def _expand_episodic_vectors(self):
        """Double the episodic vector capacity."""
        old_size = self._episodic_vectors.shape[0]
        new_size = old_size * 2
        new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_vectors[:old_size] = self._episodic_vectors[:]
        np.save(str(self.episodic_vec_path), new_vectors)
        self._episodic_vectors = np.load(
            str(self.episodic_vec_path), mmap_mode='r+'
        )
        logger.info(f"Expanded episodic vectors: {old_size} -> {new_size}")
    
    def _expand_semantic_vectors(self):
        """Double the semantic vector capacity."""
        old_size = self._semantic_vectors.shape[0]
        new_size = old_size * 2
        new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_vectors[:old_size] = self._semantic_vectors[:]
        np.save(str(self.semantic_vec_path), new_vectors)
        self._semantic_vectors = np.load(
            str(self.semantic_vec_path), mmap_mode='r+'
        )
        logger.info(f"Expanded semantic vectors: {old_size} -> {new_size}")
    
    def flush(self):
        """Flush all pending writes to disk."""
        if self._conn:
            self._conn.commit()
        if self._episodic_vectors is not None:
            self._episodic_vectors.flush()
        if self._semantic_vectors is not None:
            self._semantic_vectors.flush()
    
    def close(self):
        """Close all resources."""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def get_storage_stats(self) -> dict:
        """Return storage statistics."""
        db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        ep_vec_size = os.path.getsize(self.episodic_vec_path) if self.episodic_vec_path.exists() else 0
        sem_vec_size = os.path.getsize(self.semantic_vec_path) if self.semantic_vec_path.exists() else 0
        
        return {
            'db_size_kb': db_size / 1024,
            'episodic_vectors_kb': ep_vec_size / 1024,
            'semantic_vectors_kb': sem_vec_size / 1024,
            'total_size_kb': (db_size + ep_vec_size + sem_vec_size) / 1024,
            'total_size_mb': (db_size + ep_vec_size + sem_vec_size) / 1024 / 1024,
            'episode_count': self.count_episodes(),
            'semantic_count': self.count_semantics(),
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

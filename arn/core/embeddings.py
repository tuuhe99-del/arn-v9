"""
ARN v9 Embedding Engine
========================
Supports multiple embedding models with different size/quality tradeoffs.

Model options (set via ARN_EMBEDDING_MODEL env var or EmbeddingEngine param):

| Model tier      | Model                           | Size    | Dim  | MTEB  | Notes |
|-----------------|---------------------------------|---------|------|-------|-------|
| nano (default)  | all-MiniLM-L6-v2                | ~22MB   | 384  | 56.3  | Fast, Pi-friendly |
| small           | all-mpnet-base-v2               | ~420MB  | 768  | 57.8  | Balanced |
| base (RECO)     | BAAI/bge-base-en-v1.5           | ~440MB  | 768  | 63.6  | Best retrieval quality |
| base-e5         | intfloat/e5-base-v2             | ~440MB  | 768  | 61.5  | Good general-purpose |

The "base" tier models use QUERY/PASSAGE prefix asymmetry — queries get
a "Represent this sentence..." or "query:" prefix while stored passages
get different (or no) prefix. This is what actually moves the needle
on the temporal/paraphrase problems, not just raw dimension count.

For Pi 5 deployment:
- nano:  ~90MB RAM, ~30ms/encode
- base:  ~500MB RAM, ~80ms/encode — still viable on 8GB Pi 5
"""

import os
import numpy as np
from typing import List, Optional, Union
import hashlib
import logging

# Suppress noisy HuggingFace/safetensors warnings before any ML imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['SAFETENSORS_FAST_GPU'] = '0'
import warnings as _w
_w.filterwarnings('ignore', message='.*Unauthenticated.*')
_w.filterwarnings('ignore', message='.*huggingface.*')
_w.filterwarnings('ignore', category=FutureWarning)
for _name in ('sentence_transformers', 'transformers', 'huggingface_hub', 
              'safetensors', 'huggingface_hub.utils'):
    logging.getLogger(_name).setLevel(logging.ERROR)

logger = logging.getLogger("arn.embeddings")


# =========================================================
# MODEL REGISTRY
# =========================================================

MODEL_CONFIGS = {
    'nano': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'query_prefix': '',
        'passage_prefix': '',
        'approx_ram_mb': 90,
        # Empirically calibrated from stress test data
        # "low" means: even the top match is weak — caller should be skeptical
        'low_conf_threshold': 0.40,
        'high_conf_threshold': 0.55,
    },
    'small': {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'dim': 768,
        'query_prefix': '',
        'passage_prefix': '',
        'approx_ram_mb': 420,
        'low_conf_threshold': 0.40,
        'high_conf_threshold': 0.55,
    },
    'base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dim': 768,
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
        'approx_ram_mb': 440,
        # bge scores compress higher
        'low_conf_threshold': 0.55,
        'high_conf_threshold': 0.65,
    },
    'base-e5': {
        'name': 'intfloat/e5-base-v2',
        'dim': 768,
        'query_prefix': 'query: ',
        'passage_prefix': 'passage: ',
        'approx_ram_mb': 440,
        'low_conf_threshold': 0.78,
        'high_conf_threshold': 0.85,
    },
    # Large tier — high-end desktop / server / GPU machine
    # ~1.3GB RAM, significantly better semantic understanding
    'large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
        'approx_ram_mb': 1300,
        'low_conf_threshold': 0.55,
        'high_conf_threshold': 0.68,
    },
    # XL tier — GPU strongly recommended, best available quality
    # ~2.5GB RAM (CPU) or VRAM (GPU), MTEB top-tier
    'xl': {
        'name': 'intfloat/e5-mistral-7b-instruct',
        'dim': 4096,
        'query_prefix': 'Instruct: Retrieve semantically similar text\nQuery: ',
        'passage_prefix': '',
        'approx_ram_mb': 2500,  # fp16; 14GB+ for full fp32
        'low_conf_threshold': 0.50,
        'high_conf_threshold': 0.65,
    },
}

# Default tier — can be overridden via env or parameter
DEFAULT_TIER = os.environ.get('ARN_EMBEDDING_TIER', 'nano')


class EmbeddingEngine:
    """
    Semantic embedding engine with tiered model support and query/passage
    asymmetry for retrieval quality.
    """
    
    def __init__(self, use_model: bool = True, cache_size: int = 1024,
                 tier: Optional[str] = None):
        """
        Args:
            use_model: if False, uses hash fallback (for unit tests only)
            cache_size: LRU cache size for encoded strings
            tier: one of 'nano', 'small', 'base', 'base-e5'.
                  Defaults to ARN_EMBEDDING_TIER env or 'nano'.
        """
        self._tier = tier or DEFAULT_TIER
        if self._tier not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown tier '{self._tier}'. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )
        
        self._config = MODEL_CONFIGS[self._tier]
        self.embedding_dim = self._config['dim']
        
        self._model = None
        self._use_model = use_model
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_size = cache_size
        self._encode_count = 0
        self._cache_hits = 0
        self._degraded_warned = False
        
        if use_model:
            self._load_model()
    
    def _load_model(self):
        """Lazy-load the configured sentence transformer model."""
        try:
            # Suppress noisy HuggingFace warnings that alarm non-technical users
            import warnings
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            warnings.filterwarnings('ignore', message='.*Unauthenticated.*')
            warnings.filterwarnings('ignore', message='.*huggingface.*')
            
            import logging as _log
            for noisy in ('sentence_transformers', 'transformers', 
                         'huggingface_hub', 'safetensors'):
                _log.getLogger(noisy).setLevel(_log.ERROR)
            
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model ({self._tier}): {self._config['name']}")
            
            # Suppress ALL model-load noise including C-level stderr writes
            # from safetensors (BertModel LOAD REPORT). Python-level redirects
            # don't catch these because they write to the raw fd, not sys.stderr.
            import sys
            _old_out, _old_err = sys.stdout, sys.stderr
            _devnull = os.open(os.devnull, os.O_WRONLY)
            _old_stderr_fd = os.dup(2)
            _old_stdout_fd = os.dup(1)
            os.dup2(_devnull, 2)
            os.dup2(_devnull, 1)
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                self._model = SentenceTransformer(
                    self._config['name'],
                    device='cpu'
                )
            finally:
                # Restore everything
                sys.stdout.close()
                sys.stderr.close()
                os.dup2(_old_stderr_fd, 2)
                os.dup2(_old_stdout_fd, 1)
                os.close(_devnull)
                os.close(_old_stderr_fd)
                os.close(_old_stdout_fd)
                sys.stdout = _old_out
                sys.stderr = _old_err
            logger.info(
                f"Loaded {self._tier} model — dim={self._config['dim']}, "
                f"~{self._config['approx_ram_mb']}MB RAM"
            )
        except ImportError:
            self._model = None
            self._use_model = False
            logger.critical(
                "sentence-transformers is NOT installed. "
                "ARN is running in DEGRADED MODE with hash-based vectors. "
                "Recall, consolidation, and contradiction detection WILL NOT WORK. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            self._model = None
            self._use_model = False
            logger.critical(
                f"Failed to load embedding model '{self._config['name']}': {e}. "
                "ARN is running in DEGRADED MODE. Memory will not function correctly."
            )
    
    @property
    def is_degraded(self) -> bool:
        """True if running without real embeddings. Memory is non-functional."""
        return self._model is None
    
    @property
    def tier(self) -> str:
        return self._tier
    
    def _prefix_for_mode(self, mode: str) -> str:
        """
        Return the appropriate prefix for the given mode.
        
        Args:
            mode: 'query' (for recall queries) or 'passage' (for stored content)
        """
        if mode == 'query':
            return self._config['query_prefix']
        elif mode == 'passage':
            return self._config['passage_prefix']
        return ''
    
    def encode(self, text: str, mode: str = 'passage') -> np.ndarray:
        """
        Encode a single text string to a normalized vector.
        
        Args:
            text: the text to encode
            mode: 'query' (retrieval query) or 'passage' (stored content).
                  Some models (bge, e5) use different prefixes for each.
        """
        prefix = self._prefix_for_mode(mode)
        full_text = prefix + text if prefix else text
        
        cache_key = f"{mode}:{full_text[:500]}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()
        
        self._encode_count += 1
        
        if self._model is not None:
            vec = self._model.encode(
                [full_text],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].astype(np.float32)
        else:
            if not self._degraded_warned:
                logger.warning(
                    "encode() called without embedding model — returning random vector. "
                    "This result has NO semantic meaning."
                )
                self._degraded_warned = True
            vec = self._hash_encode(full_text)
        
        # LRU cache
        self._cache[cache_key] = vec.copy()
        self._cache_order.append(cache_key)
        if len(self._cache_order) > self._cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)
        
        return vec
    
    def encode_batch(self, texts: List[str], mode: str = 'passage') -> np.ndarray:
        """
        Encode multiple texts at once. All texts use the same mode.
        Returns (N, dim) array of normalized vectors.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        prefix = self._prefix_for_mode(mode)
        prefixed = [prefix + t if prefix else t for t in texts]
        
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, full_text in enumerate(prefixed):
            cache_key = f"{mode}:{full_text[:500]}"
            if cache_key in self._cache:
                results[i] = self._cache[cache_key].copy()
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(full_text)
        
        if uncached_texts:
            self._encode_count += len(uncached_texts)
            if self._model is not None:
                vecs = self._model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                ).astype(np.float32)
            else:
                vecs = np.array([self._hash_encode(t) for t in uncached_texts])
            
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                full_text = prefixed[idx]
                cache_key = f"{mode}:{full_text[:500]}"
                self._cache[cache_key] = vec.copy()
                self._cache_order.append(cache_key)
        
        while len(self._cache_order) > self._cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)
        
        return np.array(results, dtype=np.float32)
    
    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(vec_a, vec_b))
    
    def batch_similarity(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute similarity between a query vector and multiple candidates."""
        if len(candidates) == 0:
            return np.array([], dtype=np.float32)
        return candidates @ query
    
    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based encoding (FALLBACK ONLY)."""
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], 'little')
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
    
    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            'model_loaded': self._model is not None,
            'degraded': self.is_degraded,
            'tier': self._tier,
            'model_name': self._config['name'],
            'embedding_dim': self.embedding_dim,
            'approx_ram_mb': self._config['approx_ram_mb'],
            'uses_asymmetric_prefixes': bool(
                self._config['query_prefix'] or self._config['passage_prefix']
            ),
            'total_encodes': self._encode_count,
            'cache_hits': self._cache_hits,
            'cache_size': len(self._cache),
            'cache_hit_rate': (
                self._cache_hits / (self._encode_count + self._cache_hits)
                if (self._encode_count + self._cache_hits) > 0 else 0.0
            ),
        }
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()


# =========================================================
# MODULE-LEVEL CONSTANTS (backwards compatible)
# =========================================================

# Legacy constant — kept for backwards compat, but code should query
# engine.embedding_dim since it varies by tier now
EMBEDDING_DIM = MODEL_CONFIGS[DEFAULT_TIER]['dim']

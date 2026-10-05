"""Read-only preflight and fusion sweep for hybrid search on Qdrant."""
from .preflight import Finding, preflight_hybrid_search
from .sweep import ndcg, paired_interval, run_sweep, validate_queries

__version__ = '0.1.0'

__all__ = ['Finding', 'preflight_hybrid_search', 'run_sweep', 'validate_queries',
           'ndcg', 'paired_interval', '__version__']

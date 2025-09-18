"""
OMNI ALPHA 5.0 - OBSERVABILITY MODULE
====================================
"""

try:
    from .tracing import get_distributed_tracing, initialize_tracing
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

__all__ = []

if TRACING_AVAILABLE:
    __all__.extend(['get_distributed_tracing', 'initialize_tracing'])

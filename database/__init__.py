"""
OMNI ALPHA 5.0 - DATABASE MODULE
===============================
"""

try:
    from .connection_pool import get_production_database_pool, initialize_production_db
    PRODUCTION_DB_AVAILABLE = True
except ImportError:
    PRODUCTION_DB_AVAILABLE = False

__all__ = []

if PRODUCTION_DB_AVAILABLE:
    __all__.extend(['get_production_database_pool', 'initialize_production_db'])

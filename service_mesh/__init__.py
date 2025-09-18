"""
OMNI ALPHA 5.0 - SERVICE MESH MODULE
====================================
"""

try:
    from .consul_registry import get_service_registry, initialize_service_registry
    SERVICE_MESH_AVAILABLE = True
except ImportError:
    SERVICE_MESH_AVAILABLE = False

__all__ = []

if SERVICE_MESH_AVAILABLE:
    __all__.extend(['get_service_registry', 'initialize_service_registry'])

"""
OMNI ALPHA 5.0 - ENTERPRISE SECURITY MODULE
===========================================
"""

try:
    from .security_manager import get_enterprise_security_manager, initialize_enterprise_security
    ENTERPRISE_SECURITY_AVAILABLE = True
except ImportError:
    ENTERPRISE_SECURITY_AVAILABLE = False

__all__ = []

if ENTERPRISE_SECURITY_AVAILABLE:
    __all__.extend(['get_enterprise_security_manager', 'initialize_enterprise_security'])

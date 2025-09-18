"""
OMNI ALPHA 5.0 - TESTING MODULE
===============================
"""

try:
    from .load_tests.load_test_framework import get_load_test_runner
    LOAD_TESTING_AVAILABLE = True
except ImportError:
    LOAD_TESTING_AVAILABLE = False

__all__ = []

if LOAD_TESTING_AVAILABLE:
    __all__.extend(['get_load_test_runner'])

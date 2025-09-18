"""
OMNI ALPHA 5.0 - MESSAGING MODULE
=================================
"""

try:
    from .queue_manager import get_message_queue_manager, initialize_message_queue
    MESSAGE_QUEUE_AVAILABLE = True
except ImportError:
    MESSAGE_QUEUE_AVAILABLE = False

__all__ = []

if MESSAGE_QUEUE_AVAILABLE:
    __all__.extend(['get_message_queue_manager', 'initialize_message_queue'])

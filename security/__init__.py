"""
Omni Alpha Cybersecurity Fortress
Military-grade security implementation
"""

__version__ = "1.0.0"
__author__ = "Omni Alpha Security Team"

# Security modules - avoid circular imports
try:
    from .zero_trust_framework import ZeroTrustSecurityFramework
    from .threat_detection_ai import AIThreatDetectionSystem
    from .advanced_encryption import AdvancedEncryption
    from .application_security import ApplicationSecurityLayer
    from .security_manager import ComprehensiveSecurityManager
    SECURITY_AVAILABLE = True
except ImportError as e:
    SECURITY_AVAILABLE = False

__all__ = [
    'ZeroTrustSecurityFramework',
    'AIThreatDetectionSystem', 
    'AdvancedEncryption',
    'ApplicationSecurityLayer',
    'ComprehensiveSecurityManager',
    'SECURITY_AVAILABLE'
]

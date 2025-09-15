"""
Global Financial Ecosystem - Step 12
The ultimate market dominance framework
"""

from .controller import (
    GlobalEcosystemController,
    EcosystemConfig,
    EcosystemState,
    HealthStatus,
    RevenueStream
)

from .roles import (
    SystemRole,
    ComponentRole,
    ECOSYSTEM_COMPONENTS
)

__all__ = [
    "GlobalEcosystemController",
    "EcosystemConfig", 
    "EcosystemState",
    "HealthStatus",
    "RevenueStream",
    "SystemRole",
    "ComponentRole",
    "ECOSYSTEM_COMPONENTS"
]

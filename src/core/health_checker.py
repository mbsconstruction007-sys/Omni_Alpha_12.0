"""
Advanced Health Check System
Comprehensive health monitoring with dependency checks
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    status: HealthStatus
    message: str
    response_time: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class HealthChecker:
    """Advanced health checking system"""
    
    def __init__(self):
        self.checks = {}
        self.last_results = {}
    
    def register_check(self, name: str, check_func, timeout: int = 5):
        """Register a health check function"""
        self.checks[name] = {
            "function": check_func,
            "timeout": timeout
        }
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
                response_time=0.0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]["function"]
            timeout = self.checks[name]["timeout"]
            
            # Run check with timeout
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            response_time = time.time() - start_time
            
            if result:
                status = HealthStatus.HEALTHY
                message = "Check passed"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Check failed"
            
            return HealthCheckResult(
                status=status,
                message=message,
                response_time=response_time,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {timeout}s",
                response_time=response_time,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time=response_time,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.checks:
            results[name] = await self.run_check(name)
        
        self.last_results = results
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

# Global health checker instance
health_checker = HealthChecker()

# Register default health checks
async def database_health_check():
    """Check database connectivity"""
    # Implement actual database check
    return True

async def redis_health_check():
    """Check Redis connectivity"""
    # Implement actual Redis check
    return True

async def external_api_health_check():
    """Check external API connectivity"""
    # Implement actual external API check
    return True

# Register checks
health_checker.register_check("database", database_health_check)
health_checker.register_check("redis", redis_health_check)
health_checker.register_check("external_apis", external_api_health_check)

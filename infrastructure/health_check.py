import asyncio
from typing import Dict, Any, List
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthCheck:
    """Simple health check system"""
    
    def __init__(self):
        self.components = {}
        self.last_check = None
        
    def register_component(self, name: str, health_func):
        """Register a component for health checking"""
        self.components[name] = health_func
        
    async def check_all(self) -> Dict[str, Any]:
        """Check health of all registered components"""
        results = {}
        overall_status = 'healthy'
        
        for name, health_func in self.components.items():
            try:
                if asyncio.iscoroutinefunction(health_func):
                    health = await health_func()
                else:
                    health = health_func()
                    
                results[name] = health
                
                # Determine overall status
                if health.get('status') == 'unhealthy':
                    overall_status = 'unhealthy'
                elif health.get('status') == 'degraded' and overall_status == 'healthy':
                    overall_status = 'degraded'
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                overall_status = 'unhealthy'
        
        self.last_check = datetime.now()
        
        return {
            'overall_status': overall_status,
            'components': results,
            'last_check': self.last_check.isoformat(),
            'healthy_count': len([r for r in results.values() if r.get('status') == 'healthy']),
            'total_count': len(results)
        }

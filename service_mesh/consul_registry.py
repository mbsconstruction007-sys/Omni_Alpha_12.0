"""
OMNI ALPHA 5.0 - SERVICE DISCOVERY & MESH
=========================================
Production-ready service discovery with Consul and health monitoring
"""

import asyncio
import json
import socket
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

try:
    import consul.aio
    import consul
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'service_mesh')

# Metrics (if available)
if PROMETHEUS_AVAILABLE:
    service_registrations = Counter('service_registrations_total', 'Total service registrations', ['service', 'status'])
    service_discoveries = Counter('service_discoveries_total', 'Total service discoveries', ['service', 'status'])
    health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['service'])
    active_services = Gauge('active_services_total', 'Number of active services', ['datacenter'])

class ServiceRegistry:
    """Service discovery and health monitoring with Consul"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'service_registry')
        
        # Consul configuration
        self.consul_host = os.getenv('CONSUL_HOST', 'localhost')
        self.consul_port = int(os.getenv('CONSUL_PORT', '8500'))
        self.consul_token = os.getenv('CONSUL_TOKEN', '')
        self.consul_datacenter = os.getenv('CONSUL_DATACENTER', 'dc1')
        
        # Service configuration
        self.service_name = os.getenv('SERVICE_NAME', 'omni-alpha-trading')
        self.service_version = os.getenv('SERVICE_VERSION', '5.0.0')
        self.service_host = os.getenv('SERVICE_HOST', self._get_local_ip())
        self.service_port = int(os.getenv('SERVICE_PORT', '8080'))
        self.instance_id = os.getenv('INSTANCE_ID', f'{self.service_name}-{socket.gethostname()}-{os.getpid()}')
        
        # Health check configuration
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '10'))
        self.health_check_timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '5'))
        self.health_check_deregister_critical = os.getenv('HEALTH_CHECK_DEREGISTER_CRITICAL', '30s')
        
        # Service tags
        self.service_tags = os.getenv('SERVICE_TAGS', '').split(',') if os.getenv('SERVICE_TAGS') else []
        self.service_tags.extend([
            f'version-{self.service_version}',
            f'environment-{os.getenv("ENVIRONMENT", "production")}',
            'trading-system',
            'omni-alpha'
        ])
        
        # State
        self.consul: Optional[consul.aio.Consul] = None
        self.is_registered = False
        self.health_check_task: Optional[asyncio.Task] = None
        self.service_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_ttl = int(os.getenv('SERVICE_CACHE_TTL', '30'))
        self.last_cache_update: Dict[str, datetime] = {}
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def initialize(self) -> bool:
        """Initialize Consul client and register service"""
        if not CONSUL_AVAILABLE:
            self.logger.warning("Consul client not available - service discovery disabled")
            return False
        
        try:
            # Initialize Consul client
            self.consul = consul.aio.Consul(
                host=self.consul_host,
                port=self.consul_port,
                token=self.consul_token or None,
                datacenter=self.consul_datacenter
            )
            
            # Test connection
            await self.consul.status.leader()
            
            # Register service
            await self.register_service()
            
            # Start health check updates
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info(f"Service registry initialized with Consul at {self.consul_host}:{self.consul_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service registry: {e}")
            return False
    
    async def register_service(self) -> bool:
        """Register service with Consul"""
        if not self.consul:
            return False
        
        try:
            # Service definition
            service_definition = {
                'ID': self.instance_id,
                'Name': self.service_name,
                'Tags': self.service_tags,
                'Address': self.service_host,
                'Port': self.service_port,
                'Meta': {
                    'version': self.service_version,
                    'environment': os.getenv('ENVIRONMENT', 'production'),
                    'started_at': datetime.utcnow().isoformat(),
                    'pid': str(os.getpid()),
                    'hostname': socket.gethostname()
                },
                'Checks': [
                    {
                        'CheckID': f'{self.instance_id}-health',
                        'Name': f'{self.service_name} Health Check',
                        'HTTP': f'http://{self.service_host}:{self.service_port}/health',
                        'Method': 'GET',
                        'Interval': f'{self.health_check_interval}s',
                        'Timeout': f'{self.health_check_timeout}s',
                        'DeregisterCriticalServiceAfter': self.health_check_deregister_critical,
                        'TLSSkipVerify': True
                    },
                    {
                        'CheckID': f'{self.instance_id}-ttl',
                        'Name': f'{self.service_name} TTL Check',
                        'TTL': f'{self.health_check_interval * 2}s',
                        'DeregisterCriticalServiceAfter': self.health_check_deregister_critical
                    }
                ]
            }
            
            # Add TCP check if HTTP health endpoint is not available
            if not await self._check_http_health():
                service_definition['Checks'] = [
                    {
                        'CheckID': f'{self.instance_id}-tcp',
                        'Name': f'{self.service_name} TCP Check',
                        'TCP': f'{self.service_host}:{self.service_port}',
                        'Interval': f'{self.health_check_interval}s',
                        'Timeout': f'{self.health_check_timeout}s',
                        'DeregisterCriticalServiceAfter': self.health_check_deregister_critical
                    }
                ]
            
            # Register with Consul
            await self.consul.agent.service.register(**service_definition)
            
            self.is_registered = True
            
            if PROMETHEUS_AVAILABLE:
                service_registrations.labels(service=self.service_name, status='success').inc()
            
            self.logger.info(f"Service registered: {self.instance_id} at {self.service_host}:{self.service_port}")
            return True
            
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                service_registrations.labels(service=self.service_name, status='error').inc()
            
            self.logger.error(f"Failed to register service: {e}")
            return False
    
    async def _check_http_health(self) -> bool:
        """Check if HTTP health endpoint is available"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'http://{self.service_host}:{self.service_port}/health',
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def discover_service(self, service_name: str, healthy_only: bool = True,
                              tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover healthy instances of a service"""
        if not self.consul:
            return []
        
        # Check cache first
        cache_key = f"{service_name}:{healthy_only}:{tag or 'all'}"
        if (cache_key in self.service_cache and 
            cache_key in self.last_cache_update and
            datetime.now() - self.last_cache_update[cache_key] < timedelta(seconds=self.cache_ttl)):
            return self.service_cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Query Consul
            if healthy_only:
                _, services = await self.consul.health.service(
                    service_name,
                    passing=True,
                    tag=tag
                )
            else:
                _, services = await self.consul.catalog.service(
                    service_name,
                    tag=tag
                )
            
            instances = []
            for service in services:
                service_info = service.get('Service', service)
                
                instance = {
                    'id': service_info['ID'],
                    'name': service_info['Service'],
                    'address': service_info['Address'],
                    'port': service_info['Port'],
                    'tags': service_info.get('Tags', []),
                    'meta': service_info.get('Meta', {}),
                    'datacenter': service.get('Node', {}).get('Datacenter', 'unknown')
                }
                
                # Add health status if available
                if 'Checks' in service:
                    checks = service['Checks']
                    passing_checks = [c for c in checks if c['Status'] == 'passing']
                    instance['health'] = {
                        'status': 'passing' if len(passing_checks) == len(checks) else 'failing',
                        'checks': len(checks),
                        'passing': len(passing_checks)
                    }
                
                instances.append(instance)
            
            # Update cache
            self.service_cache[cache_key] = instances
            self.last_cache_update[cache_key] = datetime.now()
            
            # Update metrics
            duration = time.time() - start_time
            if PROMETHEUS_AVAILABLE:
                service_discoveries.labels(service=service_name, status='success').inc()
                health_check_duration.labels(service=service_name).observe(duration)
            
            self.logger.debug(f"Discovered {len(instances)} instances of {service_name}")
            return instances
            
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                service_discoveries.labels(service=service_name, status='error').inc()
            
            self.logger.error(f"Failed to discover service {service_name}: {e}")
            return []
    
    async def get_service_endpoint(self, service_name: str, tag: Optional[str] = None) -> Optional[str]:
        """Get a single service endpoint with load balancing"""
        instances = await self.discover_service(service_name, healthy_only=True, tag=tag)
        
        if not instances:
            return None
        
        # Simple round-robin load balancing
        import random
        instance = random.choice(instances)
        return f"http://{instance['address']}:{instance['port']}"
    
    async def watch_service(self, service_name: str, callback: callable, tag: Optional[str] = None):
        """Watch service for changes and call callback"""
        last_instances = []
        
        while True:
            try:
                current_instances = await self.discover_service(service_name, tag=tag)
                
                # Check for changes
                if current_instances != last_instances:
                    await callback(service_name, current_instances, last_instances)
                    last_instances = current_instances.copy()
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error watching service {service_name}: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _health_check_loop(self):
        """Periodic health check updates"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.is_registered and self.consul:
                    # Update TTL check
                    try:
                        await self.consul.agent.check.ttl_pass(
                            f'{self.instance_id}-ttl',
                            f'Service healthy at {datetime.utcnow().isoformat()}'
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update TTL check: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def update_service_meta(self, meta: Dict[str, str]):
        """Update service metadata"""
        if not self.consul or not self.is_registered:
            return False
        
        try:
            # Get current service
            services = await self.consul.agent.services()
            current_service = services.get(self.instance_id)
            
            if not current_service:
                return False
            
            # Update metadata
            current_service['Meta'].update(meta)
            
            # Re-register with updated metadata
            await self.consul.agent.service.register(
                service_id=self.instance_id,
                name=self.service_name,
                tags=current_service['Tags'],
                address=current_service['Address'],
                port=current_service['Port'],
                meta=current_service['Meta']
            )
            
            self.logger.info(f"Updated service metadata: {meta}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update service metadata: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get Consul cluster information"""
        if not self.consul:
            return {}
        
        try:
            # Get leader
            leader = await self.consul.status.leader()
            
            # Get peers
            peers = await self.consul.status.peers()
            
            # Get datacenters
            datacenters = await self.consul.catalog.datacenters()
            
            # Get all services
            _, services = await self.consul.catalog.services()
            
            return {
                'leader': leader,
                'peers': peers,
                'datacenters': datacenters,
                'services_count': len(services),
                'services': list(services.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster info: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Service registry health check"""
        if not self.consul:
            return {
                'status': 'critical',
                'message': 'Consul client not initialized',
                'metrics': {}
            }
        
        try:
            start_time = time.time()
            
            # Test Consul connectivity
            leader = await self.consul.status.leader()
            
            # Check service registration status
            services = await self.consul.agent.services()
            is_registered = self.instance_id in services
            
            response_time = (time.time() - start_time) * 1000
            
            # Get cluster info
            cluster_info = await self.get_cluster_info()
            
            status = 'healthy' if is_registered and leader else 'degraded'
            
            return {
                'status': status,
                'message': f'Consul leader: {leader}, Service registered: {is_registered}',
                'metrics': {
                    'response_time_ms': response_time,
                    'is_registered': is_registered,
                    'consul_leader': leader,
                    'cluster_peers': len(cluster_info.get('peers', [])),
                    'total_services': cluster_info.get('services_count', 0),
                    'cached_services': len(self.service_cache)
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Service registry health check failed: {str(e)}',
                'metrics': {'error': str(e)}
            }
    
    async def deregister(self):
        """Deregister service from Consul"""
        if not self.consul or not self.is_registered:
            return
        
        try:
            # Cancel health check task
            if self.health_check_task:
                self.health_check_task.cancel()
            
            # Deregister service
            await self.consul.agent.service.deregister(self.instance_id)
            
            self.is_registered = False
            self.logger.info(f"Service deregistered: {self.instance_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service: {e}")

# ===================== SERVICE MESH UTILITIES =====================

class ServiceLoadBalancer:
    """Simple load balancer for service instances"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
        self.round_robin_counters: Dict[str, int] = {}
    
    async def get_instance(self, service_name: str, strategy: str = 'round_robin',
                          tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get service instance using specified load balancing strategy"""
        instances = await self.registry.discover_service(service_name, healthy_only=True, tag=tag)
        
        if not instances:
            return None
        
        if strategy == 'round_robin':
            counter = self.round_robin_counters.get(service_name, 0)
            instance = instances[counter % len(instances)]
            self.round_robin_counters[service_name] = counter + 1
            return instance
        
        elif strategy == 'random':
            import random
            return random.choice(instances)
        
        elif strategy == 'least_connections':
            # For now, just return random (would need connection tracking)
            import random
            return random.choice(instances)
        
        else:
            # Default to first instance
            return instances[0]

# ===================== GLOBAL INSTANCE =====================

_service_registry = None

def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance"""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry

async def initialize_service_registry():
    """Initialize service registry"""
    registry = get_service_registry()
    success = await registry.initialize()
    
    if success:
        # Register health check
        from infrastructure.monitoring import get_health_monitor
        health_monitor = get_health_monitor()
        health_monitor.register_health_check('service_registry', registry.health_check)
    
    return success

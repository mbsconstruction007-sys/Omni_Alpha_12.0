"""
OMNI ALPHA 5.0 - PRODUCTION DATABASE CONNECTION POOL
====================================================
Enterprise-grade database connection pool with failover, monitoring, and high availability
"""

import asyncio
import asyncpg
import os
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager
import time
import threading
from dataclasses import dataclass
from enum import Enum
import json

try:
    import redis.asyncio as redis
    from redis.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Histogram, Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'database_pool')

# Metrics (if available)
if PROMETHEUS_AVAILABLE:
    db_connection_time = Histogram('db_connection_duration_seconds', 'Database connection time')
    db_query_time = Histogram('db_query_duration_seconds', 'Database query execution time', ['query_type', 'database'])
    db_connection_pool_size = Gauge('db_connection_pool_size', 'Current connection pool size', ['database', 'pool_type'])
    db_connection_errors = Counter('db_connection_errors_total', 'Total database connection errors', ['database', 'error_type'])
    db_query_errors = Counter('db_query_errors_total', 'Total database query errors', ['database', 'error_type'])
    db_failover_events = Counter('db_failover_events_total', 'Total database failover events', ['from_db', 'to_db'])

class DatabaseType(Enum):
    """Database types"""
    PRIMARY = "primary"
    REPLICA = "replica"
    REDIS = "redis"

class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_enabled: bool = True
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_timeout: int = 30
    command_timeout: int = 60
    max_queries: int = 50000
    max_inactive_lifetime: int = 300

@dataclass
class PoolMetrics:
    """Connection pool metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    avg_query_time_ms: float = 0.0
    total_queries: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0

class ProductionDatabasePool:
    """Enterprise-grade database connection pool with failover"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'database_pool')
        
        # Connection pools
        self.primary_pool: Optional[asyncpg.Pool] = None
        self.replica_pools: List[asyncpg.Pool] = []
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_sentinel: Optional[Sentinel] = None
        
        # Configuration
        self.primary_config = self._create_primary_config()
        self.replica_configs = self._create_replica_configs()
        self.redis_config = self._create_redis_config()
        
        # Monitoring
        self.metrics = PoolMetrics()
        self.start_time = time.time()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Failover state
        self.primary_healthy = True
        self.replica_health = {}
        self.redis_healthy = True
        self._lock = threading.Lock()
    
    def _create_primary_config(self) -> DatabaseConfig:
        """Create primary database configuration"""
        return DatabaseConfig(
            host=os.getenv('DB_PRIMARY_HOST', 'localhost'),
            port=int(os.getenv('DB_PRIMARY_PORT', '5432')),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'omni_alpha'),
            ssl_enabled=os.getenv('DB_SSL_ENABLED', 'true').lower() == 'true',
            pool_min_size=int(os.getenv('DB_POOL_MIN_SIZE', '5')),
            pool_max_size=int(os.getenv('DB_POOL_MAX_SIZE', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', '60')),
            max_queries=int(os.getenv('DB_MAX_QUERIES', '50000')),
            max_inactive_lifetime=int(os.getenv('DB_MAX_INACTIVE_LIFETIME', '300'))
        )
    
    def _create_replica_configs(self) -> List[DatabaseConfig]:
        """Create replica database configurations"""
        replica_hosts = os.getenv('DB_REPLICA_HOSTS', '').split(',')
        configs = []
        
        for host in replica_hosts:
            if host.strip():
                configs.append(DatabaseConfig(
                    host=host.strip(),
                    port=int(os.getenv('DB_REPLICA_PORT', '5432')),
                    user=os.getenv('DB_REPLICA_USER', 'readonly'),
                    password=os.getenv('DB_REPLICA_PASSWORD', ''),
                    database=os.getenv('DB_NAME', 'omni_alpha'),
                    ssl_enabled=os.getenv('DB_SSL_ENABLED', 'true').lower() == 'true',
                    pool_min_size=2,
                    pool_max_size=10,
                    pool_timeout=10,
                    command_timeout=30
                ))
        
        return configs
    
    def _create_redis_config(self) -> Dict[str, Any]:
        """Create Redis configuration"""
        return {
            'sentinel_enabled': os.getenv('REDIS_SENTINEL_ENABLED', 'false').lower() == 'true',
            'sentinel_hosts': os.getenv('REDIS_SENTINEL_HOSTS', '').split(','),
            'master_name': os.getenv('REDIS_MASTER_NAME', 'mymaster'),
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD', ''),
            'db': int(os.getenv('REDIS_DB', '0')),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '50')),
            'socket_timeout': float(os.getenv('REDIS_SOCKET_TIMEOUT', '5.0')),
            'socket_connect_timeout': float(os.getenv('REDIS_CONNECT_TIMEOUT', '5.0'))
        }
    
    async def initialize(self) -> bool:
        """Initialize all database connections with failover"""
        try:
            self.logger.info("Initializing production database pools...")
            
            # Initialize primary database
            await self._init_primary_db()
            
            # Initialize read replicas
            await self._init_replicas()
            
            # Initialize Redis
            if REDIS_AVAILABLE:
                await self._init_redis()
            
            # Start monitoring tasks
            self._health_check_task = asyncio.create_task(self._health_monitor())
            self._metrics_task = asyncio.create_task(self._metrics_collector())
            
            self.logger.info("Production database pools initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database pools: {e}")
            return False
    
    async def _init_primary_db(self):
        """Initialize primary database with retry logic"""
        config = self.primary_config
        
        try:
            start_time = time.time()
            
            self.primary_pool = await asyncpg.create_pool(
                host=config.host,
                port=config.port,
                user=config.user,
                password=config.password,
                database=config.database,
                min_size=config.pool_min_size,
                max_size=config.pool_max_size,
                max_queries=config.max_queries,
                max_inactive_connection_lifetime=config.max_inactive_lifetime,
                timeout=config.pool_timeout,
                command_timeout=config.command_timeout,
                ssl='require' if config.ssl_enabled else None,
                server_settings={
                    'application_name': f'omni_alpha_pool_{os.getpid()}',
                    'timezone': 'UTC'
                }
            )
            
            connection_time = time.time() - start_time
            
            if PROMETHEUS_AVAILABLE:
                db_connection_time.observe(connection_time)
                db_connection_pool_size.labels(database='primary', pool_type='main').set(self.primary_pool._size)
            
            self.primary_healthy = True
            self.logger.info(f"Primary database pool initialized: {config.host}:{config.port} ({self.primary_pool._size} connections)")
            
        except Exception as e:
            self.primary_healthy = False
            if PROMETHEUS_AVAILABLE:
                db_connection_errors.labels(database='primary', error_type='init').inc()
            
            self.logger.error(f"Failed to initialize primary database: {e}")
            raise
    
    async def _init_replicas(self):
        """Initialize read replica pools"""
        for i, config in enumerate(self.replica_configs):
            try:
                pool = await asyncpg.create_pool(
                    host=config.host,
                    port=config.port,
                    user=config.user,
                    password=config.password,
                    database=config.database,
                    min_size=config.pool_min_size,
                    max_size=config.pool_max_size,
                    max_queries=config.max_queries,
                    timeout=config.pool_timeout,
                    command_timeout=config.command_timeout,
                    ssl='require' if config.ssl_enabled else None,
                    server_settings={
                        'application_name': f'omni_alpha_replica_{i}_{os.getpid()}',
                        'timezone': 'UTC'
                    }
                )
                
                self.replica_pools.append(pool)
                self.replica_health[config.host] = True
                
                if PROMETHEUS_AVAILABLE:
                    db_connection_pool_size.labels(database=f'replica_{config.host}', pool_type='replica').set(pool._size)
                
                self.logger.info(f"Replica pool initialized: {config.host}:{config.port}")
                
            except Exception as e:
                self.replica_health[config.host] = False
                if PROMETHEUS_AVAILABLE:
                    db_connection_errors.labels(database=f'replica_{config.host}', error_type='init').inc()
                
                self.logger.error(f"Failed to initialize replica {config.host}: {e}")
    
    async def _init_redis(self):
        """Initialize Redis with Sentinel for HA"""
        config = self.redis_config
        
        try:
            if config['sentinel_enabled'] and config['sentinel_hosts']:
                # Use Redis Sentinel for HA
                sentinel_hosts = [
                    tuple(host.strip().split(':')) 
                    for host in config['sentinel_hosts'] 
                    if host.strip()
                ]
                
                self.redis_sentinel = Sentinel(
                    sentinel_hosts,
                    socket_timeout=config['socket_timeout'],
                    password=config['password']
                )
                
                # Get master connection pool
                master = self.redis_sentinel.master_for(
                    config['master_name'],
                    socket_timeout=config['socket_timeout'],
                    password=config['password'],
                    decode_responses=True,
                    max_connections=config['max_connections']
                )
                self.redis_pool = master.connection_pool
                
                self.logger.info(f"Redis Sentinel initialized with master: {config['master_name']}")
                
            else:
                # Standard Redis connection
                self.redis_pool = redis.ConnectionPool(
                    host=config['host'],
                    port=config['port'],
                    password=config['password'],
                    db=config['db'],
                    max_connections=config['max_connections'],
                    socket_timeout=config['socket_timeout'],
                    socket_connect_timeout=config['socket_connect_timeout'],
                    socket_keepalive=True,
                    socket_keepalive_options={
                        1: 1,  # TCP_KEEPIDLE
                        2: 2,  # TCP_KEEPINTVL
                        3: 2,  # TCP_KEEPCNT
                    },
                    decode_responses=True
                )
                
                self.logger.info(f"Redis pool initialized: {config['host']}:{config['port']}")
            
            self.redis_healthy = True
            
        except Exception as e:
            self.redis_healthy = False
            if PROMETHEUS_AVAILABLE:
                db_connection_errors.labels(database='redis', error_type='init').inc()
            
            self.logger.error(f"Failed to initialize Redis: {e}")
    
    async def _health_monitor(self):
        """Continuous health monitoring of all connections"""
        health_check_interval = int(os.getenv('DB_HEALTH_CHECK_INTERVAL', '30'))
        
        while True:
            try:
                await asyncio.sleep(health_check_interval)
                
                # Check primary database
                if self.primary_pool:
                    try:
                        async with self.primary_pool.acquire() as conn:
                            await conn.fetchval('SELECT 1')
                        
                        if not self.primary_healthy:
                            self.logger.info("Primary database recovered")
                            self.primary_healthy = True
                            
                    except Exception as e:
                        if self.primary_healthy:
                            self.logger.error(f"Primary database health check failed: {e}")
                            if PROMETHEUS_AVAILABLE:
                                db_connection_errors.labels(database='primary', error_type='health_check').inc()
                        self.primary_healthy = False
                
                # Check replicas
                for i, pool in enumerate(self.replica_pools):
                    config = self.replica_configs[i]
                    try:
                        async with pool.acquire() as conn:
                            await conn.fetchval('SELECT 1')
                        
                        if not self.replica_health.get(config.host, False):
                            self.logger.info(f"Replica {config.host} recovered")
                            self.replica_health[config.host] = True
                            
                    except Exception as e:
                        if self.replica_health.get(config.host, True):
                            self.logger.error(f"Replica {config.host} health check failed: {e}")
                            if PROMETHEUS_AVAILABLE:
                                db_connection_errors.labels(database=f'replica_{config.host}', error_type='health_check').inc()
                        self.replica_health[config.host] = False
                
                # Check Redis
                if self.redis_pool and REDIS_AVAILABLE:
                    try:
                        async with redis.Redis(connection_pool=self.redis_pool) as r:
                            await r.ping()
                        
                        if not self.redis_healthy:
                            self.logger.info("Redis recovered")
                            self.redis_healthy = True
                            
                    except Exception as e:
                        if self.redis_healthy:
                            self.logger.error(f"Redis health check failed: {e}")
                            if PROMETHEUS_AVAILABLE:
                                db_connection_errors.labels(database='redis', error_type='health_check').inc()
                        self.redis_healthy = False
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    async def _metrics_collector(self):
        """Collect and update pool metrics"""
        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                with self._lock:
                    self.metrics.uptime_seconds = time.time() - self.start_time
                    
                    if self.primary_pool:
                        self.metrics.total_connections = self.primary_pool._size
                        self.metrics.active_connections = len([c for c in self.primary_pool._holders if c._con is not None])
                        self.metrics.idle_connections = self.primary_pool._size - self.metrics.active_connections
                    
                    # Update Prometheus metrics
                    if PROMETHEUS_AVAILABLE and self.primary_pool:
                        db_connection_pool_size.labels(database='primary', pool_type='active').set(self.metrics.active_connections)
                        db_connection_pool_size.labels(database='primary', pool_type='idle').set(self.metrics.idle_connections)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
    
    @asynccontextmanager
    async def get_connection(self, read_only: bool = False, timeout: Optional[float] = None):
        """Get database connection with automatic failover"""
        pool = None
        database_type = "primary"
        
        if read_only and self.replica_pools:
            # Load balance across healthy replicas
            healthy_replicas = [
                (pool, config.host) for pool, config in zip(self.replica_pools, self.replica_configs)
                if self.replica_health.get(config.host, False)
            ]
            
            if healthy_replicas:
                import random
                pool, host = random.choice(healthy_replicas)
                database_type = f"replica_{host}"
            else:
                # Fall back to primary if no healthy replicas
                pool = self.primary_pool if self.primary_healthy else None
                database_type = "primary"
        else:
            pool = self.primary_pool if self.primary_healthy else None
        
        if not pool:
            # Try to failover to any available database
            if self.replica_pools and any(self.replica_health.values()):
                healthy_replicas = [
                    (pool, config.host) for pool, config in zip(self.replica_pools, self.replica_configs)
                    if self.replica_health.get(config.host, False)
                ]
                if healthy_replicas:
                    import random
                    pool, host = random.choice(healthy_replicas)
                    database_type = f"replica_{host}_failover"
                    
                    if PROMETHEUS_AVAILABLE:
                        db_failover_events.labels(from_db='primary', to_db=f'replica_{host}').inc()
                    
                    self.logger.warning(f"Failed over to replica: {host}")
        
        if not pool:
            raise RuntimeError("No healthy database connections available")
        
        start_time = time.time()
        try:
            async with pool.acquire(timeout=timeout) as conn:
                async with conn.transaction():
                    yield conn, database_type
                    
        except Exception as e:
            query_time = (time.time() - start_time) * 1000
            if PROMETHEUS_AVAILABLE:
                db_query_errors.labels(database=database_type, error_type=str(type(e).__name__)).inc()
            raise
        finally:
            query_time = (time.time() - start_time) * 1000
            if PROMETHEUS_AVAILABLE:
                db_query_time.labels(query_type='transaction', database=database_type).observe(query_time / 1000)
    
    async def execute_query(self, query: str, *args, read_only: bool = False, timeout: Optional[float] = None):
        """Execute query with metrics and error handling"""
        start_time = time.time()
        
        try:
            async with self.get_connection(read_only=read_only, timeout=timeout) as (conn, db_type):
                result = await conn.fetch(query, *args)
                
                query_time = (time.time() - start_time) * 1000
                
                if PROMETHEUS_AVAILABLE:
                    db_query_time.labels(
                        query_type='read' if read_only else 'write', 
                        database=db_type
                    ).observe(query_time / 1000)
                
                with self._lock:
                    self.metrics.total_queries += 1
                    # Update rolling average
                    alpha = 0.1
                    if self.metrics.avg_query_time_ms == 0:
                        self.metrics.avg_query_time_ms = query_time
                    else:
                        self.metrics.avg_query_time_ms = (
                            alpha * query_time + 
                            (1 - alpha) * self.metrics.avg_query_time_ms
                        )
                
                return result
                
        except Exception as e:
            with self._lock:
                self.metrics.failed_connections += 1
                self.metrics.last_error = str(e)
            raise
    
    async def get_redis_connection(self):
        """Get Redis connection"""
        if not self.redis_pool or not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available")
        
        return redis.Redis(connection_pool=self.redis_pool)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status"""
        with self._lock:
            status = {
                'primary': {
                    'healthy': self.primary_healthy,
                    'pool_size': self.primary_pool._size if self.primary_pool else 0,
                    'active_connections': self.metrics.active_connections,
                    'idle_connections': self.metrics.idle_connections
                },
                'replicas': {},
                'redis': {
                    'healthy': self.redis_healthy,
                    'available': REDIS_AVAILABLE
                },
                'metrics': {
                    'total_queries': self.metrics.total_queries,
                    'failed_connections': self.metrics.failed_connections,
                    'avg_query_time_ms': self.metrics.avg_query_time_ms,
                    'uptime_seconds': self.metrics.uptime_seconds,
                    'last_error': self.metrics.last_error
                }
            }
            
            for config in self.replica_configs:
                status['replicas'][config.host] = {
                    'healthy': self.replica_health.get(config.host, False)
                }
            
            return status
    
    async def close(self):
        """Gracefully close all connections"""
        self.logger.info("Closing database pools...")
        
        # Cancel monitoring tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Close primary pool
        if self.primary_pool:
            await self.primary_pool.close()
            self.logger.info("Primary pool closed")
        
        # Close replica pools
        for i, pool in enumerate(self.replica_pools):
            await pool.close()
            self.logger.info(f"Replica pool {i} closed")
        
        self.logger.info("All database pools closed")

# ===================== GLOBAL INSTANCE =====================

_production_db_pool = None

def get_production_database_pool() -> ProductionDatabasePool:
    """Get global production database pool instance"""
    global _production_db_pool
    if _production_db_pool is None:
        _production_db_pool = ProductionDatabasePool()
    return _production_db_pool

async def initialize_production_db():
    """Initialize production database pool"""
    pool = get_production_database_pool()
    success = await pool.initialize()
    
    if success:
        # Register health check
        from infrastructure.monitoring import get_health_monitor
        health_monitor = get_health_monitor()
        
        async def db_health_check():
            status = pool.get_pool_status()
            
            if status['primary']['healthy']:
                return {
                    'status': 'healthy',
                    'message': 'Database pools operational',
                    'metrics': status['metrics']
                }
            else:
                return {
                    'status': 'critical',
                    'message': 'Primary database unhealthy',
                    'metrics': status['metrics']
                }
        
        health_monitor.register_health_check('production_database', db_health_check)
    
    return success

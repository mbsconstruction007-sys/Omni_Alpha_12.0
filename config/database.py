"""
OMNI ALPHA 5.0 - DATABASE CONNECTION MANAGER
============================================
Production-ready database connections with pooling, failover, and monitoring
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import time
import threading
from dataclasses import dataclass

try:
    from sqlalchemy import create_engine, text, event
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool, StaticPool
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ===================== DATABASE MODELS BASE =====================

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
else:
    Base = None

# ===================== CONNECTION MANAGERS =====================

@dataclass
class ConnectionStats:
    """Database connection statistics"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    avg_response_time_ms: float = 0.0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0

class PostgreSQLManager:
    """PostgreSQL connection manager with pooling"""
    
    def __init__(self, settings):
        self.settings = settings
        self.engine = None
        self.Session = None
        self.stats = ConnectionStats()
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        if SQLALCHEMY_AVAILABLE:
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize PostgreSQL engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.settings.database.postgres_url,
                poolclass=QueuePool,
                pool_size=self.settings.database.postgres_pool_size,
                max_overflow=self.settings.database.postgres_max_overflow,
                pool_timeout=self.settings.database.postgres_pool_timeout,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
                echo=self.settings.debug,
                connect_args={
                    "application_name": "omni_alpha_5.0",
                    "connect_timeout": 10
                }
            )
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Add event listeners for monitoring
            event.listen(self.engine, "connect", self._on_connect)
            event.listen(self.engine, "checkout", self._on_checkout)
            event.listen(self.engine, "checkin", self._on_checkin)
            
            logger.info("PostgreSQL engine initialized with connection pooling")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL engine: {e}")
            self.stats.last_error = str(e)
            raise
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Handle new database connection"""
        with self._lock:
            self.stats.total_connections += 1
        logger.debug("New PostgreSQL connection established")
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout from pool"""
        with self._lock:
            self.stats.active_connections += 1
    
    def _on_checkin(self, dbapi_connection, connection_record):
        """Handle connection checkin to pool"""
        with self._lock:
            self.stats.active_connections = max(0, self.stats.active_connections - 1)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Session, None]:
        """Get database session with automatic cleanup"""
        if not self.Session:
            raise RuntimeError("Database not initialized")
        
        session = self.Session()
        start_time = time.time()
        
        try:
            yield session
            session.commit()
            
            # Update response time stats
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
        except Exception as e:
            session.rollback()
            with self._lock:
                self.stats.failed_connections += 1
                self.stats.last_error = str(e)
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time"""
        with self._lock:
            if self.stats.avg_response_time_ms == 0:
                self.stats.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.stats.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * self.stats.avg_response_time_ms
                )
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
    
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics"""
        with self._lock:
            self.stats.uptime_seconds = time.time() - self.start_time
            return self.stats
    
    def close(self):
        """Close all connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connections closed")

class RedisManager:
    """Redis connection manager for caching"""
    
    def __init__(self, settings):
        self.settings = settings
        self.client = None
        self.stats = ConnectionStats()
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        if REDIS_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client"""
        try:
            # Parse Redis URL
            redis_config = {
                'decode_responses': True,
                'socket_connect_timeout': 5,
                'socket_timeout': 5,
                'retry_on_timeout': True,
                'health_check_interval': 30
            }
            
            if self.settings.database.redis_password:
                redis_config['password'] = self.settings.database.redis_password
            
            self.client = redis.from_url(
                self.settings.database.redis_url,
                **redis_config
            )
            
            # Test connection
            self.client.ping()
            
            with self._lock:
                self.stats.total_connections += 1
            
            logger.info("Redis client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.stats.last_error = str(e)
            self.client = None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis with error handling"""
        if not self.client:
            return None
        
        start_time = time.time()
        
        try:
            result = self.client.get(key)
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return result
            
        except RedisError as e:
            with self._lock:
                self.stats.failed_connections += 1
                self.stats.last_error = str(e)
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with TTL"""
        if not self.client:
            return False
        
        start_time = time.time()
        
        try:
            if ttl:
                result = self.client.setex(key, ttl, value)
            else:
                result = self.client.set(key, value)
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return bool(result)
            
        except RedisError as e:
            with self._lock:
                self.stats.failed_connections += 1
                self.stats.last_error = str(e)
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.client:
            return False
        
        try:
            result = self.client.delete(key)
            return bool(result)
        except RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time"""
        with self._lock:
            if self.stats.avg_response_time_ms == 0:
                self.stats.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.stats.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * self.stats.avg_response_time_ms
                )
    
    async def health_check(self) -> bool:
        """Check Redis connectivity"""
        try:
            if self.client:
                self.client.ping()
                return True
            return False
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics"""
        with self._lock:
            self.stats.uptime_seconds = time.time() - self.start_time
            return self.stats
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")

class InfluxDBManager:
    """InfluxDB manager for time-series data"""
    
    def __init__(self, settings):
        self.settings = settings
        self.client = None
        self.write_api = None
        self.query_api = None
        self.stats = ConnectionStats()
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        if INFLUXDB_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize InfluxDB client"""
        try:
            self.client = InfluxDBClient(
                url=self.settings.database.influxdb_url,
                token=self.settings.database.influxdb_token,
                org=self.settings.database.influxdb_org,
                timeout=30000
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            self.client.ping()
            
            with self._lock:
                self.stats.total_connections += 1
            
            logger.info("InfluxDB client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            self.stats.last_error = str(e)
            self.client = None
    
    async def write_point(self, measurement: str, tags: Dict[str, str], 
                         fields: Dict[str, Any], timestamp: Optional[int] = None) -> bool:
        """Write data point to InfluxDB"""
        if not self.write_api:
            return False
        
        start_time = time.time()
        
        try:
            point = Point(measurement)
            
            # Add tags
            for key, value in tags.items():
                point = point.tag(key, value)
            
            # Add fields
            for key, value in fields.items():
                point = point.field(key, value)
            
            # Add timestamp if provided
            if timestamp:
                point = point.time(timestamp, WritePrecision.NS)
            
            # Write point
            self.write_api.write(
                bucket=self.settings.database.influxdb_bucket,
                record=point
            )
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return True
            
        except Exception as e:
            with self._lock:
                self.stats.failed_connections += 1
                self.stats.last_error = str(e)
            logger.error(f"InfluxDB write error: {e}")
            return False
    
    async def query(self, flux_query: str) -> Optional[Any]:
        """Query data from InfluxDB"""
        if not self.query_api:
            return None
        
        start_time = time.time()
        
        try:
            result = self.query_api.query(flux_query)
            
            response_time = (time.time() - start_time) * 1000
            self._update_response_time(response_time)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.stats.failed_connections += 1
                self.stats.last_error = str(e)
            logger.error(f"InfluxDB query error: {e}")
            return None
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time"""
        with self._lock:
            if self.stats.avg_response_time_ms == 0:
                self.stats.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.stats.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * self.stats.avg_response_time_ms
                )
    
    async def health_check(self) -> bool:
        """Check InfluxDB connectivity"""
        try:
            if self.client:
                self.client.ping()
                return True
            return False
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            return False
    
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics"""
        with self._lock:
            self.stats.uptime_seconds = time.time() - self.start_time
            return self.stats
    
    def close(self):
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()
            logger.info("InfluxDB connection closed")

class DatabaseManager:
    """Main database manager orchestrating all database connections"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.postgres = PostgreSQLManager(settings)
        self.redis = RedisManager(settings)
        self.influxdb = InfluxDBManager(settings)
        
        self.is_initialized = False
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize all database connections"""
        try:
            self.logger.info("Initializing database connections...")
            
            # Test all connections
            postgres_ok = await self.postgres.health_check()
            redis_ok = await self.redis.health_check()
            influxdb_ok = await self.influxdb.health_check()
            
            # Log status
            self.logger.info(f"PostgreSQL: {'✅' if postgres_ok else '❌'}")
            self.logger.info(f"Redis: {'✅' if redis_ok else '❌'}")
            self.logger.info(f"InfluxDB: {'✅' if influxdb_ok else '❌'}")
            
            # Require at least PostgreSQL/SQLite
            if not postgres_ok:
                self.logger.warning("PostgreSQL not available, using SQLite fallback")
                # Initialize SQLite fallback
                self._initialize_sqlite_fallback()
            
            self.is_initialized = True
            self.logger.info("Database manager initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def _initialize_sqlite_fallback(self):
        """Initialize SQLite as fallback database"""
        try:
            if SQLALCHEMY_AVAILABLE:
                self.postgres.engine = create_engine(
                    self.settings.database.sqlite_url,
                    poolclass=StaticPool,
                    connect_args={'check_same_thread': False},
                    echo=self.settings.debug
                )
                self.postgres.Session = sessionmaker(bind=self.postgres.engine)
                
                # Create tables if using Base
                if Base:
                    Base.metadata.create_all(self.postgres.engine)
                
                self.logger.info("SQLite fallback initialized")
        except Exception as e:
            self.logger.error(f"SQLite fallback initialization failed: {e}")
    
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[Session, None]:
        """Get PostgreSQL session"""
        async with self.postgres.get_session() as session:
            yield session
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        return await self.redis.get(key)
    
    async def cache_set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = self.settings.cache_ttl_seconds
        return await self.redis.set(key, value, ttl)
    
    async def write_timeseries(self, measurement: str, tags: Dict[str, str], 
                              fields: Dict[str, Any], timestamp: Optional[int] = None) -> bool:
        """Write time-series data to InfluxDB"""
        return await self.influxdb.write_point(measurement, tags, fields, timestamp)
    
    async def query_timeseries(self, flux_query: str) -> Optional[Any]:
        """Query time-series data from InfluxDB"""
        return await self.influxdb.query(flux_query)
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all databases"""
        postgres_health = await self.postgres.health_check()
        redis_health = await self.redis.health_check()
        influxdb_health = await self.influxdb.health_check()
        
        # Calculate overall health score
        healthy_count = sum([postgres_health, redis_health, influxdb_health])
        health_score = healthy_count / 3
        
        # Determine status
        if health_score >= 0.8:
            status = "HEALTHY"
        elif health_score >= 0.5:
            status = "DEGRADED"
        else:
            status = "CRITICAL"
        
        return {
            'status': status,
            'health_score': health_score,
            'databases': {
                'postgres': {
                    'healthy': postgres_health,
                    'stats': self.postgres.get_stats().__dict__
                },
                'redis': {
                    'healthy': redis_health,
                    'stats': self.redis.get_stats().__dict__
                },
                'influxdb': {
                    'healthy': influxdb_health,
                    'stats': self.influxdb.get_stats().__dict__
                }
            }
        }
    
    async def shutdown(self):
        """Shutdown all database connections"""
        self.logger.info("Shutting down database connections...")
        
        self.postgres.close()
        self.redis.close()
        self.influxdb.close()
        
        self.logger.info("All database connections closed")

# ===================== GLOBAL DATABASE INSTANCE =====================

_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

async def initialize_databases():
    """Initialize global database manager"""
    db_manager = get_database_manager()
    return await db_manager.initialize()

async def shutdown_databases():
    """Shutdown global database manager"""
    global _db_manager
    if _db_manager:
        await _db_manager.shutdown()
        _db_manager = None

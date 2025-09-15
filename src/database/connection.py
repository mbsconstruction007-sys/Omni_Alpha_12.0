"""
High-performance database connection manager with pooling
"""

import asyncio
import asyncpg
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient
import structlog
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import os
from urllib.parse import urlparse

logger = structlog.get_logger()

class DatabaseManager:
    """
    World-class database connection manager with:
    - Connection pooling
    - Automatic reconnection
    - Health monitoring
    - Performance tracking
    """
    
    def __init__(self):
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.timescale_pool: Optional[asyncpg.Pool] = None
        
        # Connection settings
        self.pg_dsn = os.getenv("DATABASE_URL", "postgresql://omni:omni@localhost:5432/omni_alpha")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017/omni_alpha")
        self.timescale_dsn = os.getenv("TIMESCALE_URL", "postgresql://omni:omni@localhost:5433/market_data")
        
        # Pool settings
        self.pool_min_size = int(os.getenv("DB_POOL_MIN", "10"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX", "50"))
        self.pool_max_queries = int(os.getenv("DB_POOL_MAX_QUERIES", "50000"))
        self.pool_max_inactive_connection_lifetime = float(os.getenv("DB_POOL_MAX_INACTIVE", "300"))
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # PostgreSQL connection pool
            logger.info("Initializing PostgreSQL connection pool...")
            self.postgres_pool = await asyncpg.create_pool(
                self.pg_dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                max_queries=self.pool_max_queries,
                max_inactive_connection_lifetime=self.pool_max_inactive_connection_lifetime,
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # Disable JIT for consistent performance
                    'application_name': 'omni_alpha_trading'
                },
            )
            
            # TimescaleDB connection pool (for time-series data)
            logger.info("Initializing TimescaleDB connection pool...")
            self.timescale_pool = await asyncpg.create_pool(
                self.timescale_dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size * 2,  # More connections for high-frequency data
                max_queries=self.pool_max_queries * 2,
                command_timeout=30,
                server_settings={
                    'jit': 'off',
                    'application_name': 'omni_alpha_market_data'
                },
            )
            
            # Redis connection
            logger.info("Initializing Redis connection...")
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.pool_max_size,
                health_check_interval=30,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                }
            )
            
            # MongoDB connection (for unstructured data)
            logger.info("Initializing MongoDB connection...")
            self.mongo_client = AsyncIOMotorClient(
                self.mongo_url,
                maxPoolSize=self.pool_max_size,
                minPoolSize=self.pool_min_size,
                maxIdleTimeMS=300000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )
            
            # Verify connections
            await self._verify_connections()
            
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
            
    async def _verify_connections(self):
        """Verify all database connections are working"""
        # Test PostgreSQL
        async with self.postgres_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            logger.info("PostgreSQL connection verified")
            
        # Test TimescaleDB
        async with self.timescale_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            logger.info("TimescaleDB connection verified")
            
        # Test Redis
        pong = await self.redis_client.ping()
        assert pong
        logger.info("Redis connection verified")
        
        # Test MongoDB
        result = await self.mongo_client.admin.command('ping')
        assert result['ok'] == 1
        logger.info("MongoDB connection verified")
        
    async def close(self):
        """Close all database connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()
            
        if self.timescale_pool:
            await self.timescale_pool.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.mongo_client:
            self.mongo_client.close()
            
        logger.info("All database connections closed")
        
    @asynccontextmanager
    async def postgres_connection(self):
        """Get PostgreSQL connection from pool"""
        async with self.postgres_pool.acquire() as connection:
            yield connection
            
    @asynccontextmanager
    async def timescale_connection(self):
        """Get TimescaleDB connection from pool"""
        async with self.timescale_pool.acquire() as connection:
            yield connection
            
    @asynccontextmanager
    async def transaction(self):
        """PostgreSQL transaction context manager"""
        async with self.postgres_pool.acquire() as connection:
            async with connection.transaction():
                yield connection
                
    async def execute(self, query: str, *args, timeout: float = None):
        """Execute a query with automatic connection management"""
        async with self.postgres_connection() as conn:
            return await conn.execute(query, *args, timeout=timeout)
            
    async def fetch(self, query: str, *args, timeout: float = None):
        """Fetch results with automatic connection management"""
        async with self.postgres_connection() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
            
    async def fetchval(self, query: str, *args, timeout: float = None):
        """Fetch single value with automatic connection management"""
        async with self.postgres_connection() as conn:
            return await conn.fetchval(query, *args, timeout=timeout)
            
    async def fetchrow(self, query: str, *args, timeout: float = None):
        """Fetch single row with automatic connection management"""
        async with self.postgres_connection() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
            
    # Redis operations
    async def cache_set(self, key: str, value: Any, ttl: int = None):
        """Set cache value with optional TTL"""
        if ttl:
            await self.redis_client.setex(key, ttl, value)
        else:
            await self.redis_client.set(key, value)
            
    async def cache_get(self, key: str):
        """Get cache value"""
        return await self.redis_client.get(key)
        
    async def cache_delete(self, key: str):
        """Delete cache value"""
        return await self.redis_client.delete(key)
        
    async def cache_exists(self, key: str):
        """Check if cache key exists"""
        return await self.redis_client.exists(key)
        
    # Performance monitoring
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'postgres': {
                'size': self.postgres_pool.get_size() if self.postgres_pool else 0,
                'free': self.postgres_pool.get_idle_size() if self.postgres_pool else 0,
                'used': self.postgres_pool.get_size() - self.postgres_pool.get_idle_size() if self.postgres_pool else 0,
            },
            'timescale': {
                'size': self.timescale_pool.get_size() if self.timescale_pool else 0,
                'free': self.timescale_pool.get_idle_size() if self.timescale_pool else 0,
                'used': self.timescale_pool.get_size() - self.timescale_pool.get_idle_size() if self.timescale_pool else 0,
            },
            'redis': {
                'connected': await self.redis_client.ping() if self.redis_client else False,
            },
            'mongo': {
                'connected': (await self.mongo_client.admin.command('ping'))['ok'] == 1 if self.mongo_client else False,
            }
        }

# Global database manager instance
db_manager = DatabaseManager()

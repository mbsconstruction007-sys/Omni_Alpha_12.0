import asyncpg
import redis
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from typing import Optional, Dict, Any
import backoff

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simplified database manager that actually works"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.influx_client: Optional[InfluxDBClient] = None
        self.connected = False
        
    async def initialize(self) -> bool:
        """Initialize all database connections with fallbacks"""
        success = True
        
        # PostgreSQL with fallback to SQLite
        try:
            await self._connect_postgres()
            logger.info("PostgreSQL connected successfully")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}, using SQLite fallback")
            self._setup_sqlite_fallback()
            
        # Redis with fallback to in-memory cache
        try:
            self._connect_redis()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self._setup_memory_cache()
            
        # InfluxDB - optional
        try:
            self._connect_influxdb()
            logger.info("InfluxDB connected successfully")
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {e}, continuing without metrics DB")
            
        self.connected = True
        return success
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _connect_postgres(self):
        """Connect to PostgreSQL with retry logic"""
        self.pg_pool = await asyncpg.create_pool(
            host=self.config.get('DB_HOST', 'localhost'),
            port=self.config.get('DB_PORT', 5432),
            user=self.config.get('DB_USER', 'postgres'),
            password=self.config.get('DB_PASSWORD', 'postgres'),
            database=self.config.get('DB_NAME', 'omni_alpha'),
            min_size=2,
            max_size=10,
            timeout=10
        )
        # Test connection
        async with self.pg_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
            
    def _setup_sqlite_fallback(self):
        """Setup SQLite as fallback database"""
        import sqlite3
        self.sqlite_conn = sqlite3.connect('omni_alpha.db')
        self.sqlite_conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.sqlite_conn.commit()
        
    def _connect_redis(self):
        """Connect to Redis"""
        self.redis_client = redis.Redis(
            host=self.config.get('REDIS_HOST', 'localhost'),
            port=self.config.get('REDIS_PORT', 6379),
            password=self.config.get('REDIS_PASSWORD', None),
            decode_responses=True,
            socket_connect_timeout=5
        )
        self.redis_client.ping()  # Test connection
        
    def _setup_memory_cache(self):
        """Setup in-memory cache as Redis fallback"""
        self.memory_cache = {}
        
    def _connect_influxdb(self):
        """Connect to InfluxDB for metrics"""
        url = self.config.get('INFLUXDB_URL', 'http://localhost:8086')
        token = self.config.get('INFLUXDB_TOKEN', 'my-token')
        org = self.config.get('INFLUXDB_ORG', 'omni-alpha')
        
        self.influx_client = InfluxDBClient(url=url, token=token, org=org)
        self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
    async def close(self):
        """Close all connections gracefully"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            self.redis_client.close()
        if self.influx_client:
            self.influx_client.close()

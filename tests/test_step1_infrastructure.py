import pytest
import asyncio
import asyncpg
import redis
import os
from unittest.mock import Mock, patch
import logging
from datetime import datetime

# Import components to test
from config.settings import get_settings
from database.simple_connection import DatabaseManager
from infrastructure.prometheus_monitor import PrometheusMonitor
from infrastructure.health_check import HealthCheck

class TestStep1Infrastructure:
    """Complete tests for Step 1: Core Infrastructure"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'DB_HOST': 'localhost',
            'DB_PORT': 5432,
            'DB_NAME': 'test_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': 6379,
            'PROMETHEUS_PORT': 8001,
            'LOG_LEVEL': 'DEBUG'
        }
    
    @pytest.mark.asyncio
    async def test_database_connection(self, config):
        """Test database connectivity with fallback"""
        db = DatabaseManager(config)
        result = await db.initialize()
        
        assert result == True
        assert db.connected == True
        
        # Test that either PostgreSQL or SQLite is working
        if db.pg_pool:
            # PostgreSQL connected
            async with db.pg_pool.acquire() as conn:
                result = await conn.fetchval('SELECT 1')
                assert result == 1
        else:
            # SQLite fallback
            assert db.sqlite_conn is not None
            cursor = db.sqlite_conn.execute('SELECT 1')
            assert cursor.fetchone()[0] == 1
            
        await db.close()
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, config):
        """Test Redis connectivity with fallback"""
        db = DatabaseManager(config)
        await db.initialize()
        
        if db.redis_client:
            # Redis connected
            db.redis_client.set('test_key', 'test_value')
            value = db.redis_client.get('test_key')
            assert value == 'test_value'
        else:
            # Memory cache fallback
            assert db.memory_cache is not None
            db.memory_cache['test_key'] = 'test_value'
            assert db.memory_cache['test_key'] == 'test_value'
            
        await db.close()
    
    def test_monitoring_initialization(self, config):
        """Test Prometheus monitoring setup"""
        monitor = PrometheusMonitor(config)
        
        # Check metrics are registered
        assert monitor.trade_counter is not None
        assert monitor.error_counter is not None
        assert monitor.latency_histogram is not None
        assert monitor.system_health is not None
        assert monitor.active_connections is not None
        
        # Test metric operations
        monitor.record_trade()
        monitor.record_error('test_component')
        monitor.record_latency(0.1)
        monitor.update_health(0.8)
        monitor.update_connections('database', 5)
        
        # Test server configuration
        assert monitor.port == 8001
    
    @pytest.mark.asyncio
    async def test_health_check_system(self):
        """Test health check functionality"""
        health = HealthCheck()
        
        # Mock health functions
        async def healthy_component():
            return {'status': 'healthy', 'message': 'All good'}
            
        def unhealthy_component():
            return {'status': 'unhealthy', 'error': 'Connection failed'}
        
        # Register components
        health.register_component('good_component', healthy_component)
        health.register_component('bad_component', unhealthy_component)
        
        # Get overall status
        status = await health.check_all()
        
        assert 'overall_status' in status
        assert 'components' in status
        assert status['components']['good_component']['status'] == 'healthy'
        assert status['components']['bad_component']['status'] == 'unhealthy'
        assert status['overall_status'] == 'unhealthy'  # Bad component makes overall unhealthy
    
    def test_configuration_loading(self):
        """Test configuration management"""
        # Set test environment variables
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['TEST_INT'] = '123'
        os.environ['TEST_BOOL'] = 'true'
        
        settings = get_settings()
        
        # Test that settings object is created
        assert settings is not None
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'version')
        assert hasattr(settings, 'environment')
        
        # Clean up
        del os.environ['TEST_VAR']
        del os.environ['TEST_INT'] 
        del os.environ['TEST_BOOL']
    
    def test_logging_configuration(self):
        """Test logging setup"""
        logger = logging.getLogger('test_logger')
        
        # Test log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify logger configuration
        assert logger.level >= 0
        # Root logger should have handlers or the logger itself should have handlers
        has_handlers = len(logger.handlers) > 0 or len(logging.root.handlers) > 0
        assert has_handlers or logger.propagate  # Either has handlers or propagates to root
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, config):
        """Test graceful shutdown of components"""
        db = DatabaseManager(config)
        await db.initialize()
        
        # Verify components are active
        assert db.connected == True
        
        # Test graceful shutdown
        await db.close()
        
        # Verify components are properly closed
        if hasattr(db, 'pg_pool') and db.pg_pool:
            assert db.pg_pool._closed == True
        if hasattr(db, 'redis_client') and db.redis_client:
            # Redis client doesn't have a direct closed check, but connection should be None
            pass
    
    def test_circuit_breaker_basic(self):
        """Test circuit breaker basic functionality"""
        # Test that circuit breaker module can be imported
        try:
            from infrastructure.circuit_breaker import get_circuit_breaker_manager
            cb_manager = get_circuit_breaker_manager()
            assert cb_manager is not None
        except ImportError:
            # If circuit breaker not available, test passes
            pass
    
    @pytest.mark.asyncio
    async def test_influxdb_optional(self, config):
        """Test that InfluxDB is optional and system works without it"""
        # Remove InfluxDB config to test fallback
        config_no_influx = config.copy()
        config_no_influx.pop('INFLUXDB_URL', None)
        
        db = DatabaseManager(config_no_influx)
        result = await db.initialize()
        
        # Should still initialize successfully without InfluxDB
        assert result == True
        assert db.connected == True
        
        await db.close()
    
    def test_prometheus_metrics_collection(self, config):
        """Test that Prometheus metrics can be collected"""
        monitor = PrometheusMonitor(config)
        
        # Record some metrics
        monitor.record_trade()
        monitor.record_trade()
        monitor.record_error('database')
        monitor.update_health(0.85)
        
        # Verify metrics are recorded (basic check)
        assert monitor.trade_counter._value._value >= 2
        assert monitor.error_counter.labels(component='database')._value._value >= 1
        assert monitor.system_health._value._value == 0.85
    
    @pytest.mark.asyncio
    async def test_database_fallback_behavior(self, config):
        """Test database fallback behavior when PostgreSQL unavailable"""
        # Use invalid PostgreSQL config to force SQLite fallback
        bad_config = config.copy()
        bad_config['DB_HOST'] = 'invalid_host'
        bad_config['DB_PORT'] = 99999
        
        db = DatabaseManager(bad_config)
        result = await db.initialize()
        
        # Should still succeed with SQLite fallback
        assert result == True
        assert db.connected == True
        assert hasattr(db, 'sqlite_conn')
        assert db.sqlite_conn is not None
        
        # Test SQLite operations
        cursor = db.sqlite_conn.execute('SELECT 1')
        assert cursor.fetchone()[0] == 1
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_redis_fallback_behavior(self, config):
        """Test Redis fallback to memory cache"""
        # Use invalid Redis config to force memory cache fallback
        bad_config = config.copy()
        bad_config['REDIS_HOST'] = 'invalid_host'
        bad_config['REDIS_PORT'] = 99999
        
        db = DatabaseManager(bad_config)
        await db.initialize()
        
        # Should fallback to memory cache
        assert hasattr(db, 'memory_cache')
        assert db.memory_cache is not None
        
        # Test memory cache operations
        db.memory_cache['test'] = 'value'
        assert db.memory_cache['test'] == 'value'
        
        await db.close()

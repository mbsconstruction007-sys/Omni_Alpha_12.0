"""
TEST SUITE FOR STEP 1: CORE INFRASTRUCTURE
Comprehensive testing of all infrastructure components
"""

import os
import sys
import asyncio
import pytest
import tempfile
import sqlite3
from datetime import datetime
from pathlib import Path
import json

# Import the Step 1 components
from step_1_core_infrastructure import (
    CoreInfrastructure,
    OmniAlphaConfig,
    DatabaseManager,
    LoggingManager,
    MetricsCollector,
    HealthChecker
)

class TestOmniAlphaConfig:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test configuration loads with defaults"""
        config = OmniAlphaConfig()
        
        assert config.APP_NAME == "Omni Alpha Enhanced"
        assert config.APP_VERSION == "12.0+"
        assert config.ENV in ["production", "development", "test"]
        assert config.MAX_POSITIONS > 0
        assert 0 < config.MAX_POSITION_SIZE_PERCENT <= 1
    
    def test_config_validation_missing_keys(self):
        """Test configuration validation with missing API keys"""
        config = OmniAlphaConfig()
        
        # Temporarily clear API keys
        original_alpaca_key = config.ALPACA_API_KEY
        original_telegram_token = config.TELEGRAM_BOT_TOKEN
        
        config.ALPACA_API_KEY = None
        config.TELEGRAM_BOT_TOKEN = None
        
        errors = config.validate_config()
        
        assert len(errors) >= 2
        assert any("ALPACA_API_KEY" in error for error in errors)
        assert any("TELEGRAM_BOT_TOKEN" in error for error in errors)
        
        # Restore original values
        config.ALPACA_API_KEY = original_alpaca_key
        config.TELEGRAM_BOT_TOKEN = original_telegram_token
    
    def test_config_validation_invalid_values(self):
        """Test configuration validation with invalid values"""
        config = OmniAlphaConfig()
        
        # Set invalid values
        config.MAX_POSITION_SIZE_PERCENT = 1.5  # > 1
        config.MAX_POSITIONS = -1  # < 0
        
        errors = config.validate_config()
        
        assert len(errors) >= 2
        assert any("MAX_POSITION_SIZE_PERCENT" in error for error in errors)
        assert any("MAX_POSITIONS" in error for error in errors)
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = OmniAlphaConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "app_name" in config_dict
        assert "app_version" in config_dict
        assert "env" in config_dict
        assert "max_positions" in config_dict

class TestDatabaseManager:
    """Test database management"""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        config = OmniAlphaConfig()
        config.DATABASE_URL = f"sqlite:///{db_path}"
        
        yield config
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_database_connection(self, temp_db_config):
        """Test database connection"""
        db_manager = DatabaseManager(temp_db_config)
        
        await db_manager.connect()
        
        assert db_manager.connected
        assert await db_manager.health_check()
    
    @pytest.mark.asyncio
    async def test_database_tables_creation(self, temp_db_config):
        """Test database table creation"""
        db_manager = DatabaseManager(temp_db_config)
        await db_manager.connect()
        
        # Check if tables exist (SQLite fallback mode)
        if hasattr(db_manager, 'connection'):
            cursor = db_manager.connection.cursor()
            
            # Check trades table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            assert cursor.fetchone() is not None
            
            # Check positions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
            assert cursor.fetchone() is not None
            
            # Check system_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_metrics'")
            assert cursor.fetchone() is not None
    
    @pytest.mark.asyncio
    async def test_database_session_context_manager(self, temp_db_config):
        """Test database session context manager"""
        db_manager = DatabaseManager(temp_db_config)
        await db_manager.connect()
        
        async with db_manager.get_session() as session:
            assert session is not None
            # In SQLite fallback mode, session is the connection
            if hasattr(session, 'cursor'):
                cursor = session.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

class TestLoggingManager:
    """Test logging management"""
    
    def test_logging_setup(self):
        """Test logging system setup"""
        config = OmniAlphaConfig()
        config.LOG_LEVEL = "INFO"
        config.LOG_FILE = "test_omni_alpha.log"
        
        logging_manager = LoggingManager(config)
        logger = logging_manager.setup_logging()
        
        assert logger is not None
        assert logger.name == "omni_alpha"
        
        # Test logging
        logger.info("Test log message")
        
        # Check if log file was created
        log_path = Path("logs") / config.LOG_FILE
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Test log message" in content
            
            # Cleanup
            log_path.unlink()

class TestMetricsCollector:
    """Test metrics collection"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        config = OmniAlphaConfig()
        config.ENABLE_METRICS = True
        config.METRICS_PORT = None  # Don't start server in tests
        
        metrics = MetricsCollector(config)
        
        if metrics.enabled:
            assert hasattr(metrics, 'trades_total')
            assert hasattr(metrics, 'portfolio_value')
            assert hasattr(metrics, 'system_health')
    
    def test_metrics_recording(self):
        """Test metrics recording"""
        config = OmniAlphaConfig()
        config.ENABLE_METRICS = True
        config.METRICS_PORT = None
        
        metrics = MetricsCollector(config)
        
        if metrics.enabled:
            # Test trade recording
            metrics.record_trade("BUY", "AAPL", 1.5)
            
            # Test portfolio value update
            metrics.update_portfolio_value(100000.0)
            
            # Test positions count update
            metrics.update_positions_count(5)
            
            # Test error recording
            metrics.record_error("connection_error")

class TestHealthChecker:
    """Test health monitoring"""
    
    @pytest.fixture
    def health_checker_setup(self):
        """Setup health checker with temporary database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        config = OmniAlphaConfig()
        config.DATABASE_URL = f"sqlite:///{db_path}"
        
        db_manager = DatabaseManager(config)
        health_checker = HealthChecker(config, db_manager)
        
        yield config, db_manager, health_checker
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_database_health_check(self, health_checker_setup):
        """Test database health check"""
        config, db_manager, health_checker = health_checker_setup
        
        await db_manager.connect()
        
        db_health = await health_checker.check_database()
        
        assert isinstance(db_health, dict)
        assert 'healthy' in db_health
        assert 'response_time_ms' in db_health
        assert 'connected' in db_health
        assert db_health['healthy'] is True
    
    @pytest.mark.asyncio
    async def test_system_resources_check(self, health_checker_setup):
        """Test system resources check"""
        config, db_manager, health_checker = health_checker_setup
        
        system_health = await health_checker.check_system_resources()
        
        assert isinstance(system_health, dict)
        
        if 'error' not in system_health:
            assert 'memory' in system_health
            assert 'disk' in system_health
            assert 'cpu' in system_health
            
            assert isinstance(system_health['memory']['percent'], (int, float))
            assert isinstance(system_health['cpu']['percent'], (int, float))
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, health_checker_setup):
        """Test comprehensive health check"""
        config, db_manager, health_checker = health_checker_setup
        
        await db_manager.connect()
        
        health_status = await health_checker.comprehensive_health_check()
        
        assert isinstance(health_status, dict)
        assert 'timestamp' in health_status
        assert 'overall_health_score' in health_status
        assert 'status' in health_status
        assert 'database' in health_status
        assert 'apis' in health_status
        
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 0 <= health_status['overall_health_score'] <= 1

class TestCoreInfrastructure:
    """Test core infrastructure orchestrator"""
    
    @pytest.fixture
    def temp_core_setup(self):
        """Setup temporary core infrastructure"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        # Set environment variables for testing
        os.environ['DATABASE_URL'] = f"sqlite:///{db_path}"
        os.environ['ALPACA_API_KEY'] = "test_key"
        os.environ['ALPACA_SECRET_KEY'] = "test_secret"
        os.environ['TELEGRAM_BOT_TOKEN'] = "test_token"
        os.environ['ENABLE_METRICS'] = "false"  # Disable metrics server in tests
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        
        # Remove test environment variables
        for key in ['DATABASE_URL', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'ENABLE_METRICS']:
            if key in os.environ:
                del os.environ[key]
    
    @pytest.mark.asyncio
    async def test_core_infrastructure_initialization(self, temp_core_setup):
        """Test full core infrastructure initialization"""
        core = CoreInfrastructure()
        
        try:
            await core.initialize()
            
            assert core.initialized
            assert core.config is not None
            assert core.db_manager is not None
            assert core.logging_manager is not None
            assert core.health_checker is not None
            assert core.logger is not None
            
        finally:
            await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_core_infrastructure_status(self, temp_core_setup):
        """Test core infrastructure status reporting"""
        core = CoreInfrastructure()
        
        try:
            await core.initialize()
            
            status = core.get_status()
            
            assert isinstance(status, dict)
            assert 'initialized' in status
            assert 'config' in status
            assert 'database_connected' in status
            assert 'metrics_enabled' in status
            
            assert status['initialized'] is True
            assert status['database_connected'] is True
            
        finally:
            await core.shutdown()
    
    @pytest.mark.asyncio
    async def test_core_infrastructure_health_check(self, temp_core_setup):
        """Test core infrastructure health check integration"""
        core = CoreInfrastructure()
        
        try:
            await core.initialize()
            
            # Perform health check
            health_status = await core.health_checker.comprehensive_health_check()
            
            assert health_status is not None
            assert health_status['overall_health_score'] > 0
            assert health_status['database']['healthy'] is True
            
        finally:
            await core.shutdown()

# ===================== INTEGRATION TESTS =====================

class TestIntegration:
    """Integration tests for complete Step 1 functionality"""
    
    @pytest.mark.asyncio
    async def test_full_step1_workflow(self):
        """Test complete Step 1 workflow"""
        
        # Set up test environment
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        os.environ['DATABASE_URL'] = f"sqlite:///{db_path}"
        os.environ['ALPACA_API_KEY'] = "test_key"
        os.environ['ALPACA_SECRET_KEY'] = "test_secret"
        os.environ['TELEGRAM_BOT_TOKEN'] = "test_token"
        os.environ['ENABLE_METRICS'] = "false"
        os.environ['LOG_LEVEL'] = "INFO"
        
        try:
            # Initialize core infrastructure
            core = CoreInfrastructure()
            await core.initialize()
            
            # Verify all components are working
            assert core.initialized
            
            # Test configuration
            config_dict = core.config.to_dict()
            assert config_dict['app_name'] == "Omni Alpha Enhanced"
            
            # Test database
            assert core.db_manager.connected
            db_health = await core.health_checker.check_database()
            assert db_health['healthy']
            
            # Test logging
            core.logger.info("Integration test log message")
            
            # Test health check
            health_status = await core.health_checker.comprehensive_health_check()
            assert health_status['overall_health_score'] > 0.5
            
            # Test status reporting
            status = core.get_status()
            assert status['initialized']
            assert status['database_connected']
            
            print("✅ Step 1 Integration Test: ALL COMPONENTS WORKING!")
            
        finally:
            await core.shutdown()
            
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
            
            # Remove test environment variables
            for key in ['DATABASE_URL', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'ENABLE_METRICS', 'LOG_LEVEL']:
                if key in os.environ:
                    del os.environ[key]

# ===================== PERFORMANCE TESTS =====================

class TestPerformance:
    """Performance tests for Step 1 components"""
    
    @pytest.mark.asyncio
    async def test_database_performance(self):
        """Test database operation performance"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        config = OmniAlphaConfig()
        config.DATABASE_URL = f"sqlite:///{db_path}"
        
        db_manager = DatabaseManager(config)
        
        try:
            # Measure connection time
            start_time = asyncio.get_event_loop().time()
            await db_manager.connect()
            connection_time = asyncio.get_event_loop().time() - start_time
            
            assert connection_time < 5.0  # Should connect in under 5 seconds
            
            # Measure health check time
            start_time = asyncio.get_event_loop().time()
            health_result = await db_manager.health_check()
            health_check_time = asyncio.get_event_loop().time() - start_time
            
            assert health_check_time < 1.0  # Health check should be fast
            assert health_result is True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self):
        """Test core infrastructure initialization performance"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        os.environ['DATABASE_URL'] = f"sqlite:///{db_path}"
        os.environ['ALPACA_API_KEY'] = "test_key"
        os.environ['ALPACA_SECRET_KEY'] = "test_secret"
        os.environ['TELEGRAM_BOT_TOKEN'] = "test_token"
        os.environ['ENABLE_METRICS'] = "false"
        
        try:
            core = CoreInfrastructure()
            
            # Measure initialization time
            start_time = asyncio.get_event_loop().time()
            await core.initialize()
            init_time = asyncio.get_event_loop().time() - start_time
            
            assert init_time < 10.0  # Should initialize in under 10 seconds
            assert core.initialized
            
            await core.shutdown()
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            
            for key in ['DATABASE_URL', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'ENABLE_METRICS']:
                if key in os.environ:
                    del os.environ[key]

# ===================== MAIN TEST EXECUTION =====================

def run_step1_tests():
    """Run all Step 1 tests"""
    print("🧪 RUNNING STEP 1 CORE INFRASTRUCTURE TESTS")
    print("=" * 60)
    
    # Run pytest with verbose output
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--color=yes'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_step1_tests()
    if success:
        print("\n✅ ALL STEP 1 TESTS PASSED!")
    else:
        print("\n❌ SOME STEP 1 TESTS FAILED!")
        sys.exit(1)

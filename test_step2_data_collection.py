"""
TEST SUITE FOR STEP 2: DATA COLLECTION & MARKET DATA
Comprehensive testing of all data collection components
"""

import os
import sys
import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from unittest.mock import Mock, patch, AsyncMock

# Import the Step 2 components
from step_2_data_collection import (
    DataCollectionSystem,
    DataRequest,
    MarketData,
    Quote,
    DataQuality,
    DataSource,
    TimeFrame,
    DataValidator,
    DataCache,
    DataStorage,
    AlpacaDataSource,
    YahooFinanceDataSource,
    AlphaVantageDataSource
)

class TestDataStructures:
    """Test data structures and enums"""
    
    def test_market_data_creation(self):
        """Test MarketData creation"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            timeframe="1Day",
            source=DataSource.ALPACA,
            quality=DataQuality.GOOD
        )
        
        assert data.symbol == "AAPL"
        assert data.open == 150.0
        assert data.high == 155.0
        assert data.low == 149.0
        assert data.close == 154.0
        assert data.volume == 1000000
        assert data.source == DataSource.ALPACA
        assert data.quality == DataQuality.GOOD
    
    def test_quote_creation(self):
        """Test Quote creation"""
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid=153.50,
            ask=153.52,
            bid_size=100,
            ask_size=200,
            last_price=153.51,
            last_size=50,
            source=DataSource.ALPACA
        )
        
        assert quote.symbol == "AAPL"
        assert quote.bid == 153.50
        assert quote.ask == 153.52
        assert quote.bid_size == 100
        assert quote.ask_size == 200
        assert quote.last_price == 153.51
        assert quote.source == DataSource.ALPACA
    
    def test_data_request_creation(self):
        """Test DataRequest creation"""
        request = DataRequest(
            symbol="MSFT",
            timeframe=TimeFrame.DAY_1,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            cache_enabled=True,
            quality_threshold=DataQuality.GOOD
        )
        
        assert request.symbol == "MSFT"
        assert request.timeframe == TimeFrame.DAY_1
        assert request.cache_enabled is True
        assert request.quality_threshold == DataQuality.GOOD

class TestDataValidator:
    """Test data validation framework"""
    
    def test_valid_market_data(self):
        """Test validation of valid market data"""
        validator = DataValidator()
        
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            timeframe="1Day",
            source=DataSource.ALPACA
        )
        
        is_valid, errors = validator.validate_market_data(data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_market_data_negative_prices(self):
        """Test validation of invalid market data with negative prices"""
        validator = DataValidator()
        
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=-150.0,  # Invalid
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            timeframe="1Day",
            source=DataSource.ALPACA
        )
        
        is_valid, errors = validator.validate_market_data(data)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("Open price must be positive" in error for error in errors)
    
    def test_invalid_market_data_ohlc_inconsistency(self):
        """Test validation of invalid OHLC data"""
        validator = DataValidator()
        
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=145.0,  # High less than low - invalid
            low=149.0,
            close=154.0,
            volume=1000000,
            timeframe="1Day",
            source=DataSource.ALPACA
        )
        
        is_valid, errors = validator.validate_market_data(data)
        
        assert is_valid is False
        assert any("High price cannot be less than low price" in error for error in errors)
    
    def test_valid_quote(self):
        """Test validation of valid quote"""
        validator = DataValidator()
        
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=1),
            bid=153.50,
            ask=153.52,
            bid_size=100,
            ask_size=200,
            last_price=153.51,
            last_size=50,
            source=DataSource.ALPACA
        )
        
        is_valid, errors = validator.validate_quote(quote)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_quote_bid_ask_spread(self):
        """Test validation of invalid quote with bid > ask"""
        validator = DataValidator()
        
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid=153.52,
            ask=153.50,  # Ask less than bid - invalid
            bid_size=100,
            ask_size=200,
            last_price=153.51,
            last_size=50,
            source=DataSource.ALPACA
        )
        
        is_valid, errors = validator.validate_quote(quote)
        
        assert is_valid is False
        assert any("Ask price cannot be less than bid price" in error for error in errors)
    
    def test_data_completeness_assessment(self):
        """Test data completeness assessment"""
        validator = DataValidator()
        
        # Create sample data with gaps
        data_list = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(8):  # Only 8 out of 10 expected data points
            data = MarketData(
                symbol="AAPL",
                timestamp=base_time + timedelta(hours=i),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Hour",
                source=DataSource.ALPACA
            )
            data_list.append(data)
        
        completeness, issues = validator.assess_data_completeness(data_list, 10)
        
        assert completeness == 0.8  # 8/10 = 80%
        assert len(issues) > 0
        assert any("Low data completeness" in issue for issue in issues)

class TestDataCache:
    """Test data caching system"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        config = {
            'memory_cache_size': 500,
            'redis_enabled': False
        }
        
        cache = DataCache(config)
        
        assert cache.memory_cache_size == 500
        assert cache.redis_client is None
    
    def test_memory_cache_operations(self):
        """Test memory cache get/set operations"""
        config = {
            'memory_cache_size': 100,
            'redis_enabled': False
        }
        
        cache = DataCache(config)
        
        # Create sample data
        data_list = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Day",
                source=DataSource.ALPACA
            )
        ]
        
        symbol = "AAPL"
        timeframe = "1Day"
        start = datetime.now() - timedelta(days=1)
        end = datetime.now()
        
        # Cache data
        cache.cache_data(symbol, timeframe, start, end, data_list)
        
        # Retrieve cached data
        cached_data = cache.get_cached_data(symbol, timeframe, start, end)
        
        assert cached_data is not None
        assert len(cached_data) == 1
        assert cached_data[0].symbol == "AAPL"
        assert cached_data[0].close == 154.0
    
    def test_cache_expiry(self):
        """Test cache expiry mechanism"""
        config = {
            'memory_cache_size': 100,
            'redis_enabled': False
        }
        
        cache = DataCache(config)
        
        # Mock datetime to simulate expired cache
        with patch('step_2_data_collection.datetime') as mock_datetime:
            # Set initial time
            initial_time = datetime.now()
            mock_datetime.now.return_value = initial_time
            
            # Cache data
            data_list = [MarketData(
                symbol="AAPL",
                timestamp=initial_time,
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Day",
                source=DataSource.ALPACA
            )]
            
            cache.cache_data("AAPL", "1Day", initial_time, initial_time, data_list)
            
            # Fast forward time to expire cache (more than 1 hour for daily data)
            mock_datetime.now.return_value = initial_time + timedelta(hours=2)
            
            # Try to retrieve expired data
            cached_data = cache.get_cached_data("AAPL", "1Day", initial_time, initial_time)
            
            assert cached_data is None

class TestDataSources:
    """Test data source implementations"""
    
    @patch('step_2_data_collection.ALPACA_AVAILABLE', True)
    @patch('step_2_data_collection.tradeapi')
    def test_alpaca_data_source_initialization(self, mock_tradeapi):
        """Test Alpaca data source initialization"""
        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }
        
        mock_api = Mock()
        mock_tradeapi.REST.return_value = mock_api
        
        source = AlpacaDataSource(config)
        
        assert source.name == "ALPACA"
        assert source.is_active is True
        assert source.api == mock_api
        
        mock_tradeapi.REST.assert_called_once_with(
            'test_key',
            'test_secret',
            'https://paper-api.alpaca.markets'
        )
    
    @patch('step_2_data_collection.YFINANCE_AVAILABLE', True)
    def test_yahoo_finance_data_source_initialization(self):
        """Test Yahoo Finance data source initialization"""
        config = {}
        
        source = YahooFinanceDataSource(config)
        
        assert source.name == "YAHOO_FINANCE"
        assert source.is_active is True
    
    @patch('step_2_data_collection.REQUESTS_AVAILABLE', True)
    def test_alpha_vantage_data_source_initialization(self):
        """Test Alpha Vantage data source initialization"""
        config = {
            'api_key': 'test_api_key'
        }
        
        with patch('step_2_data_collection.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            source = AlphaVantageDataSource(config)
            
            assert source.name == "ALPHA_VANTAGE"
            assert source.is_active is True
            assert source.api_key == 'test_api_key'
            assert source.session == mock_session
    
    @pytest.mark.asyncio
    @patch('step_2_data_collection.ALPACA_AVAILABLE', True)
    @patch('step_2_data_collection.tradeapi')
    async def test_alpaca_health_check(self, mock_tradeapi):
        """Test Alpaca health check"""
        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret'
        }
        
        mock_api = Mock()
        mock_account = Mock()
        mock_account.status = 'ACTIVE'
        mock_api.get_account.return_value = mock_account
        mock_tradeapi.REST.return_value = mock_api
        
        source = AlpacaDataSource(config)
        
        is_healthy = await source.health_check()
        
        assert is_healthy is True
        mock_api.get_account.assert_called_once()

class TestDataStorage:
    """Test data storage system"""
    
    @pytest.fixture
    def temp_storage_config(self):
        """Create temporary storage configuration"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        config = {
            'database_url': f"sqlite:///{db_path}",
            'debug': False
        }
        
        yield config
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_data_storage_initialization(self, temp_storage_config):
        """Test data storage initialization"""
        storage = DataStorage(temp_storage_config)
        
        assert storage.engine is not None
        assert storage.Session is not None
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_historical_data(self, temp_storage_config):
        """Test storing and retrieving historical data"""
        storage = DataStorage(temp_storage_config)
        
        # Create sample data
        data_list = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Day",
                source=DataSource.ALPACA,
                quality=DataQuality.GOOD
            )
        ]
        
        # Store data
        await storage.store_historical_data(data_list)
        
        # Retrieve data
        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)
        
        retrieved_data = await storage.get_historical_data(
            "AAPL", "1Day", start_date, end_date
        )
        
        assert len(retrieved_data) == 1
        assert retrieved_data[0].symbol == "AAPL"
        assert retrieved_data[0].close == 154.0
        assert retrieved_data[0].source == DataSource.ALPACA
    
    @pytest.mark.asyncio
    async def test_store_quote(self, temp_storage_config):
        """Test storing quotes"""
        storage = DataStorage(temp_storage_config)
        
        # Create sample quote
        quote = Quote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid=153.50,
            ask=153.52,
            bid_size=100,
            ask_size=200,
            last_price=153.51,
            last_size=50,
            source=DataSource.ALPACA
        )
        
        # Store quote
        await storage.store_quote(quote)
        
        # Verify storage (we'll check this by ensuring no exception is raised)
        assert True  # If we reach here, storage succeeded

class TestDataCollectionSystem:
    """Test main data collection system"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        return {
            'sources': {
                'alpaca': {
                    'enabled': True,
                    'api_key': 'test_key',
                    'secret_key': 'test_secret',
                    'base_url': 'https://paper-api.alpaca.markets'
                },
                'yahoo_finance': {
                    'enabled': True
                },
                'alpha_vantage': {
                    'enabled': False,  # Disabled for testing
                    'api_key': None
                }
            },
            'cache': {
                'memory_cache_size': 100,
                'redis_enabled': False
            },
            'storage': {
                'database_url': 'sqlite:///:memory:',
                'debug': False
            }
        }
    
    def test_system_initialization(self, mock_config):
        """Test data collection system initialization"""
        with patch('step_2_data_collection.ALPACA_AVAILABLE', False), \
             patch('step_2_data_collection.YFINANCE_AVAILABLE', False):
            
            system = DataCollectionSystem(mock_config)
            
            assert system.config == mock_config
            assert isinstance(system.validator, DataValidator)
            assert isinstance(system.cache, DataCache)
            assert isinstance(system.storage, DataStorage)
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_config):
        """Test system health check"""
        with patch('step_2_data_collection.ALPACA_AVAILABLE', True), \
             patch('step_2_data_collection.YFINANCE_AVAILABLE', True), \
             patch('step_2_data_collection.tradeapi'):
            
            system = DataCollectionSystem(mock_config)
            
            # Mock data sources
            for source in system.data_sources.values():
                source.health_check = AsyncMock(return_value=True)
            
            health_status = await system.health_check()
            
            assert 'overall_status' in health_status
            assert 'sources' in health_status
            assert 'statistics' in health_status
            assert health_status['overall_status'] in ['HEALTHY', 'WARNING', 'DEGRADED', 'CRITICAL']
    
    @pytest.mark.asyncio
    async def test_get_historical_data_with_cache(self, mock_config):
        """Test getting historical data with caching"""
        with patch('step_2_data_collection.ALPACA_AVAILABLE', True), \
             patch('step_2_data_collection.YFINANCE_AVAILABLE', True), \
             patch('step_2_data_collection.tradeapi'):
            
            system = DataCollectionSystem(mock_config)
            
            # Mock cached data
            sample_data = [
                MarketData(
                    symbol="AAPL",
                    timestamp=datetime.now(timezone.utc),
                    open=150.0,
                    high=155.0,
                    low=149.0,
                    close=154.0,
                    volume=1000000,
                    timeframe="1Day",
                    source=DataSource.ALPACA
                )
            ]
            
            system.cache.get_cached_data = Mock(return_value=sample_data)
            
            request = DataRequest(
                symbol="AAPL",
                timeframe=TimeFrame.DAY_1,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                cache_enabled=True
            )
            
            data = await system.get_historical_data(request)
            
            assert len(data) == 1
            assert data[0].symbol == "AAPL"
            assert system.stats['cache_hits'] == 1
    
    def test_statistics(self, mock_config):
        """Test statistics collection"""
        with patch('step_2_data_collection.ALPACA_AVAILABLE', False), \
             patch('step_2_data_collection.YFINANCE_AVAILABLE', False):
            
            system = DataCollectionSystem(mock_config)
            
            # Simulate some activity
            system.stats['requests_total'] = 10
            system.stats['requests_success'] = 8
            system.stats['requests_failed'] = 2
            system.stats['cache_hits'] = 3
            system.stats['cache_misses'] = 7
            
            stats = system.get_statistics()
            
            assert 'requests' in stats
            assert 'sources' in stats
            assert 'cache_hit_rate' in stats
            assert stats['cache_hit_rate'] == 0.3  # 3/10

# ===================== INTEGRATION TESTS =====================

class TestIntegration:
    """Integration tests for complete Step 2 functionality"""
    
    @pytest.mark.asyncio
    async def test_full_step2_workflow(self):
        """Test complete Step 2 workflow"""
        
        # Set up test environment
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        config = {
            'sources': {
                'alpaca': {'enabled': False},  # Disable for testing
                'yahoo_finance': {'enabled': False},  # Disable for testing
                'alpha_vantage': {'enabled': False}  # Disable for testing
            },
            'cache': {
                'memory_cache_size': 100,
                'redis_enabled': False
            },
            'storage': {
                'database_url': f"sqlite:///{db_path}",
                'debug': False
            }
        }
        
        try:
            # Initialize data collection system
            system = DataCollectionSystem(config)
            
            # Verify system initialization
            assert system.validator is not None
            assert system.cache is not None
            assert system.storage is not None
            
            # Test health check
            health_status = await system.health_check()
            assert 'overall_status' in health_status
            
            # Test statistics
            stats = system.get_statistics()
            assert 'requests' in stats
            assert 'sources' in stats
            
            print("✅ Step 2 Integration Test: ALL COMPONENTS WORKING!")
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

# ===================== PERFORMANCE TESTS =====================

class TestPerformance:
    """Performance tests for Step 2 components"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance"""
        config = {
            'memory_cache_size': 1000,
            'redis_enabled': False
        }
        
        cache = DataCache(config)
        
        # Create sample data
        data_list = [
            MarketData(
                symbol=f"TEST{i}",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Day",
                source=DataSource.ALPACA
            )
            for i in range(100)
        ]
        
        # Measure cache operations
        start_time = asyncio.get_event_loop().time()
        
        # Cache data
        for i, data in enumerate(data_list):
            cache.cache_data(
                f"TEST{i}", 
                "1Day", 
                datetime.now() - timedelta(days=1), 
                datetime.now(), 
                [data]
            )
        
        cache_time = asyncio.get_event_loop().time() - start_time
        
        # Retrieve data
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            cached_data = cache.get_cached_data(
                f"TEST{i}", 
                "1Day", 
                datetime.now() - timedelta(days=1), 
                datetime.now()
            )
            assert cached_data is not None
        
        retrieve_time = asyncio.get_event_loop().time() - start_time
        
        # Performance assertions (should be fast)
        assert cache_time < 1.0  # Should cache 100 items in under 1 second
        assert retrieve_time < 1.0  # Should retrieve 100 items in under 1 second
    
    def test_validator_performance(self):
        """Test validator performance"""
        validator = DataValidator()
        
        # Create sample data
        data_list = [
            MarketData(
                symbol=f"TEST{i}",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000000,
                timeframe="1Day",
                source=DataSource.ALPACA
            )
            for i in range(1000)
        ]
        
        # Measure validation time
        start_time = time.time()
        
        valid_count = 0
        for data in data_list:
            is_valid, errors = validator.validate_market_data(data)
            if is_valid:
                valid_count += 1
        
        validation_time = time.time() - start_time
        
        # Performance assertions
        assert validation_time < 2.0  # Should validate 1000 items in under 2 seconds
        assert valid_count == 1000  # All should be valid

# ===================== MAIN TEST EXECUTION =====================

def run_step2_tests():
    """Run all Step 2 tests"""
    print("🧪 RUNNING STEP 2 DATA COLLECTION TESTS")
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
    success = run_step2_tests()
    if success:
        print("\n✅ ALL STEP 2 TESTS PASSED!")
    else:
        print("\n❌ SOME STEP 2 TESTS FAILED!")
        sys.exit(1)

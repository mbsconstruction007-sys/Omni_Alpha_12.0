import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_collection.fixed_alpaca_collector import FixedAlpacaCollector

class TestStep2DataCollection:
    """Complete tests for Step 2: Data Collection"""
    
    @pytest.fixture
    def config(self):
        return {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }
    
    @pytest.mark.asyncio
    async def test_alpaca_initialization(self, config):
        """Test Alpaca collector initialization"""
        with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
            mock_account = Mock()
            mock_account.cash = '100000'
            mock_client.return_value.get_account.return_value = mock_account
            
            collector = FixedAlpacaCollector(config)
            result = await collector.initialize()
            
            assert result == True
            assert collector.is_connected == True
            assert collector.api_key == 'test_key'
            assert collector.secret_key == 'test_secret'
    
    @pytest.mark.asyncio
    async def test_alpaca_initialization_failure(self, config):
        """Test Alpaca collector initialization failure handling"""
        with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
            mock_client.return_value.get_account.side_effect = Exception("API Error")
            
            collector = FixedAlpacaCollector(config)
            result = await collector.initialize()
            
            assert result == False
            assert collector.is_connected == False
    
    @pytest.mark.asyncio
    async def test_streaming_subscription(self, config):
        """Test market data streaming"""
        with patch('data_collection.fixed_alpaca_collector.StockDataStream') as mock_stream:
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                collector = FixedAlpacaCollector(config)
                await collector.initialize()
                
                symbols = ['AAPL', 'GOOGL', 'MSFT']
                result = await collector.start_streaming(symbols)
                
                assert result == True
                assert collector.subscribed_symbols == symbols
                
                # Verify stream subscriptions were called
                mock_stream.return_value.subscribe_bars.assert_called()
                mock_stream.return_value.subscribe_quotes.assert_called()
    
    @pytest.mark.asyncio
    async def test_streaming_failure(self, config):
        """Test streaming failure handling"""
        with patch('data_collection.fixed_alpaca_collector.StockDataStream') as mock_stream:
            mock_stream.return_value.subscribe_bars.side_effect = Exception("Stream error")
            
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                collector = FixedAlpacaCollector(config)
                await collector.initialize()
                
                symbols = ['AAPL']
                result = await collector.start_streaming(symbols)
                
                assert result == False
    
    @pytest.mark.asyncio
    async def test_data_handlers(self, config):
        """Test data handler functionality"""
        with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
            mock_account = Mock()
            mock_account.cash = '100000'
            mock_client.return_value.get_account.return_value = mock_account
            
            collector = FixedAlpacaCollector(config)
            await collector.initialize()
            
            # Test data handler registration
            received_data = []
            
            async def test_handler(data_type, data):
                received_data.append((data_type, data))
            
            collector.add_data_handler(test_handler)
            assert len(collector.data_handlers) == 1
            
            # Test bar handling
            test_bar = {'symbol': 'AAPL', 'close': 150.0}
            await collector._handle_bar(test_bar)
            
            assert len(received_data) == 1
            assert received_data[0][0] == 'bar'
            assert received_data[0][1] == test_bar
            
            # Test quote handling
            test_quote = {'symbol': 'AAPL', 'bid': 149.99, 'ask': 150.01}
            await collector._handle_quote(test_quote)
            
            assert len(received_data) == 2
            assert received_data[1][0] == 'quote'
            assert received_data[1][1] == test_quote
    
    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, config):
        """Test historical data fetching"""
        with patch('data_collection.fixed_alpaca_collector.StockHistoricalDataClient') as mock_client:
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_trading:
                # Mock historical data response
                mock_df = pd.DataFrame({
                    'close': [100, 101, 102],
                    'volume': [1000, 1100, 1200],
                    'timestamp': [datetime.now() - timedelta(days=i) for i in range(3)]
                })
                mock_bars = Mock()
                mock_bars.df = mock_df
                mock_client.return_value.get_stock_bars.return_value = mock_bars
                
                # Mock trading client for initialization
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_trading.return_value.get_account.return_value = mock_account
                
                collector = FixedAlpacaCollector(config)
                await collector.initialize()
                
                data = await collector.get_historical_data('AAPL', days=30)
                
                assert data is not None
                assert len(data) == 3
                assert 'close' in data.columns
                assert 'volume' in data.columns
    
    @pytest.mark.asyncio
    async def test_historical_data_failure(self, config):
        """Test historical data failure handling"""
        with patch('data_collection.fixed_alpaca_collector.StockHistoricalDataClient') as mock_client:
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_trading:
                mock_client.return_value.get_stock_bars.side_effect = Exception("API Error")
                
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_trading.return_value.get_account.return_value = mock_account
                
                collector = FixedAlpacaCollector(config)
                await collector.initialize()
                
                data = await collector.get_historical_data('AAPL', days=30)
                
                assert data is None
    
    def test_health_status(self, config):
        """Test health status reporting"""
        collector = FixedAlpacaCollector(config)
        
        # Test disconnected state
        health = collector.get_health_status()
        assert health['connected'] == False
        assert health['status'] == 'degraded'
        assert health['streaming_symbols'] == 0
        
        # Test connected state
        collector.is_connected = True
        collector.subscribed_symbols = ['AAPL', 'GOOGL']
        
        health = collector.get_health_status()
        assert health['connected'] == True
        assert health['status'] == 'healthy'
        assert health['streaming_symbols'] == 2
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self, config):
        """Test WebSocket auto-reconnection"""
        with patch('data_collection.fixed_alpaca_collector.StockDataStream') as mock_stream:
            collector = FixedAlpacaCollector(config)
            collector.is_connected = True
            collector.subscribed_symbols = ['AAPL', 'GOOGL']
            
            # Test reconnection
            await collector._reconnect()
            
            # Verify new stream client was created
            assert collector.stream_client is not None
            
            # Verify subscriptions were restored
            mock_stream.return_value.subscribe_bars.assert_called()
            mock_stream.return_value.subscribe_quotes.assert_called()
    
    @pytest.mark.asyncio
    async def test_collector_close(self, config):
        """Test collector cleanup"""
        with patch('data_collection.fixed_alpaca_collector.StockDataStream') as mock_stream:
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                collector = FixedAlpacaCollector(config)
                await collector.initialize()
                
                # Close collector
                await collector.close()
                
                assert collector.is_connected == False
                mock_stream.return_value.stop.assert_called()
    
    def test_original_alpaca_collector_exists(self):
        """Test that original enhanced Alpaca collector still exists"""
        try:
            from data_collection.providers.alpaca_collector import initialize_alpaca_collector
            # If we can import it, the original enhanced version exists
            assert initialize_alpaca_collector is not None
        except ImportError:
            pytest.fail("Original enhanced Alpaca collector missing")
    
    def test_data_collection_structure(self):
        """Test that data collection structure is intact"""
        import os
        
        # Check that all data collection modules exist
        base_path = 'data_collection'
        expected_dirs = [
            'providers',
            'streams', 
            'orderbook',
            'storage',
            'validation',
            'news_sentiment'
        ]
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(base_path, dir_name)
            assert os.path.exists(dir_path), f"Missing directory: {dir_path}"
            
            init_file = os.path.join(dir_path, '__init__.py')
            assert os.path.exists(init_file), f"Missing __init__.py in: {dir_path}"
    
    def test_enhanced_alpaca_collector_preserved(self):
        """Test that enhanced Alpaca collector features are preserved"""
        # Check that the original enhanced file exists
        import os
        enhanced_file = 'data_collection/providers/alpaca_collector.py'
        assert os.path.exists(enhanced_file), "Enhanced Alpaca collector file missing"
        
        # Check file size to ensure it's the enhanced version
        file_size = os.path.getsize(enhanced_file)
        assert file_size > 10000, "Enhanced Alpaca collector appears to be missing content"
    
    @pytest.mark.asyncio
    async def test_multiple_data_handlers(self, config):
        """Test multiple data handlers"""
        with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
            mock_account = Mock()
            mock_account.cash = '100000'
            mock_client.return_value.get_account.return_value = mock_account
            
            collector = FixedAlpacaCollector(config)
            await collector.initialize()
            
            # Add multiple handlers
            handler1_data = []
            handler2_data = []
            
            async def handler1(data_type, data):
                handler1_data.append((data_type, data))
            
            async def handler2(data_type, data):
                handler2_data.append((data_type, data))
            
            collector.add_data_handler(handler1)
            collector.add_data_handler(handler2)
            
            # Test that both handlers receive data
            test_data = {'symbol': 'AAPL', 'price': 150.0}
            await collector._handle_bar(test_data)
            
            assert len(handler1_data) == 1
            assert len(handler2_data) == 1
            assert handler1_data[0][1] == test_data
            assert handler2_data[0][1] == test_data
    
    def test_config_validation(self, config):
        """Test configuration validation"""
        # Test with valid config
        collector = FixedAlpacaCollector(config)
        assert collector.api_key == 'test_key'
        assert collector.secret_key == 'test_secret'
        
        # Test with missing keys
        invalid_config = {}
        try:
            collector = FixedAlpacaCollector(invalid_config)
            # Should handle missing keys gracefully
            assert collector.api_key is None or collector.api_key == ''
        except KeyError:
            # KeyError is acceptable for missing required config
            pass

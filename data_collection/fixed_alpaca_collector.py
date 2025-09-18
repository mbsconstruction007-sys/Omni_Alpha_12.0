import asyncio
from typing import Optional, Dict, Any, Callable
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
import logging
from datetime import datetime, timedelta
import backoff

logger = logging.getLogger(__name__)

class FixedAlpacaCollector:
    """Fixed Alpaca collector that actually works"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config['ALPACA_API_KEY']
        self.secret_key = config['ALPACA_SECRET_KEY']
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.stream_client = StockDataStream(self.api_key, self.secret_key)
        
        self.is_connected = False
        self.subscribed_symbols = []
        self.data_handlers = []
        
    async def initialize(self) -> bool:
        """Initialize and verify connection"""
        try:
            # Test connection
            account = self.trading_client.get_account()
            logger.info(f"Alpaca connected: Balance=${account.cash}")
            
            # Setup stream handlers
            self.stream_client.subscribe_bars(self._handle_bar, *['SPY'])
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            self.is_connected = False
            return False
    
    async def start_streaming(self, symbols: list):
        """Start streaming market data"""
        try:
            self.subscribed_symbols = symbols
            
            # Subscribe to data streams
            if symbols:
                self.stream_client.subscribe_bars(self._handle_bar, *symbols)
                self.stream_client.subscribe_quotes(self._handle_quote, *symbols)
                
            # Run stream in background
            asyncio.create_task(self._run_stream())
            logger.info(f"Streaming started for {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    async def _run_stream(self):
        """Run the stream with error handling"""
        while self.is_connected:
            try:
                self.stream_client.run()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)
                # Reconnect
                if self.is_connected:
                    await self._reconnect()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def _reconnect(self):
        """Reconnect to stream"""
        logger.info("Attempting to reconnect to Alpaca stream...")
        self.stream_client = StockDataStream(self.api_key, self.secret_key)
        if self.subscribed_symbols:
            self.stream_client.subscribe_bars(self._handle_bar, *self.subscribed_symbols)
            self.stream_client.subscribe_quotes(self._handle_quote, *self.subscribed_symbols)
            
    async def _handle_bar(self, bar):
        """Handle bar data"""
        for handler in self.data_handlers:
            await handler('bar', bar)
            
    async def _handle_quote(self, quote):
        """Handle quote data"""
        for handler in self.data_handlers:
            await handler('quote', quote)
            
    def add_data_handler(self, handler: Callable):
        """Add a data handler"""
        self.data_handlers.append(handler)
        
    async def get_historical_data(self, symbol: str, days: int = 30):
        """Get historical data"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=datetime.now() - timedelta(days=days),
                timeframe=TimeFrame.Day
            )
            bars = self.data_client.get_stock_bars(request)
            return bars.df if hasattr(bars, 'df') else None
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None
            
    def get_health_status(self) -> Dict:
        """Get health status of the collector"""
        return {
            'connected': self.is_connected,
            'streaming_symbols': len(self.subscribed_symbols),
            'status': 'healthy' if self.is_connected else 'degraded'
        }
        
    async def close(self):
        """Close connections"""
        self.is_connected = False
        if self.stream_client:
            self.stream_client.stop()

"""
OMNI ALPHA 5.0 - ALPACA DATA COLLECTOR
======================================
Production-ready Alpaca API integration with WebSocket streaming and error handling
"""

import asyncio
import json
import time
import websockets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
import logging

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger
from infrastructure.circuit_breaker import circuit_breaker, ErrorSeverity
from infrastructure.monitoring import get_metrics_collector

# ===================== DATA STRUCTURES =====================

class AlpacaDataCollector:
    """Alpaca Markets data collector with streaming support"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'alpaca_collector')
        self.metrics = get_metrics_collector()
        
        # API configuration
        api_key, secret_key = settings.api.get_decrypted_alpaca_credentials()
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = settings.api.alpaca_base_url
        self.stream_url = settings.api.alpaca_stream_url
        
        # Initialize REST API
        if ALPACA_AVAILABLE and self.api_key and self.secret_key:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
        else:
            self.api = None
            self.logger.warning("Alpaca API not available or credentials missing")
        
        # WebSocket state
        self.websocket = None
        self.is_streaming = False
        self.subscribed_symbols = set()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = settings.data_collection.ws_max_reconnects
        
        # Data callbacks
        self.tick_callbacks: List[Callable] = []
        self.quote_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        
        # Performance tracking
        self.connection_start_time = None
        self.last_message_time = None
        self.message_count = 0
    
    def register_tick_callback(self, callback: Callable):
        """Register callback for tick data"""
        self.tick_callbacks.append(callback)
    
    def register_quote_callback(self, callback: Callable):
        """Register callback for quote data"""
        self.quote_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        """Register callback for trade data"""
        self.trade_callbacks.append(callback)
    
    @circuit_breaker('alpaca_api')
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if not self.api:
            return None
        
        try:
            start_time = time.time_ns()
            account = self.api.get_account()
            latency_us = (time.time_ns() - start_time) / 1000
            
            self.metrics.record_api_request('alpaca', 'get_account', 'success')
            self.metrics.record_data_latency('alpaca', 'account_info', latency_us)
            
            return {
                'account_id': account.id,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'currency': account.currency
            }
            
        except Exception as e:
            self.metrics.record_api_request('alpaca', 'get_account', 'error')
            self.logger.error(f"Failed to get account info: {e}")
            raise
    
    @circuit_breaker('alpaca_api')
    async def get_historical_bars(self, symbol: str, timeframe: str, 
                                 start: datetime, end: datetime,
                                 limit: int = 10000) -> List[Dict[str, Any]]:
        """Get historical bar data"""
        if not self.api:
            return []
        
        try:
            start_time = time.time_ns()
            
            # Map timeframe
            alpaca_timeframe = self._map_timeframe(timeframe)
            
            # Get bars
            bars = self.api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit,
                adjustment='raw'
            ).df
            
            latency_us = (time.time_ns() - start_time) / 1000
            
            self.metrics.record_api_request('alpaca', 'get_bars', 'success')
            self.metrics.record_data_latency('alpaca', 'historical_bars', latency_us)
            self.metrics.record_data_point('alpaca', symbol, 'historical_bars')
            
            if bars.empty:
                return []
            
            # Convert to standard format
            result = []
            for idx, row in bars.iterrows():
                bar_data = {
                    'symbol': symbol,
                    'timestamp': idx.to_pydatetime(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'trade_count': int(row.get('trade_count', 0)),
                    'vwap': float(row.get('vwap', 0)),
                    'source': 'alpaca',
                    'timeframe': timeframe
                }
                result.append(bar_data)
            
            self.logger.info(f"Retrieved {len(result)} bars for {symbol}")
            return result
            
        except Exception as e:
            self.metrics.record_api_request('alpaca', 'get_bars', 'error')
            self.logger.error(f"Failed to get historical bars for {symbol}: {e}")
            raise
    
    @circuit_breaker('alpaca_api')
    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote"""
        if not self.api:
            return None
        
        try:
            start_time = time.time_ns()
            quote = self.api.get_latest_quote(symbol)
            latency_us = (time.time_ns() - start_time) / 1000
            
            self.metrics.record_api_request('alpaca', 'get_quote', 'success')
            self.metrics.record_data_latency('alpaca', 'quote', latency_us)
            
            return {
                'symbol': symbol,
                'timestamp': quote.timestamp.replace(tzinfo=timezone.utc),
                'bid': float(quote.bp),
                'ask': float(quote.ap),
                'bid_size': int(quote.bs),
                'ask_size': int(quote.as_),
                'spread': float(quote.ap - quote.bp),
                'mid': float((quote.ap + quote.bp) / 2),
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.metrics.record_api_request('alpaca', 'get_quote', 'error')
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            raise
    
    @circuit_breaker('alpaca_api')
    async def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest trade"""
        if not self.api:
            return None
        
        try:
            start_time = time.time_ns()
            trade = self.api.get_latest_trade(symbol)
            latency_us = (time.time_ns() - start_time) / 1000
            
            self.metrics.record_api_request('alpaca', 'get_trade', 'success')
            self.metrics.record_data_latency('alpaca', 'trade', latency_us)
            
            return {
                'symbol': symbol,
                'timestamp': trade.timestamp.replace(tzinfo=timezone.utc),
                'price': float(trade.p),
                'size': int(trade.s),
                'conditions': trade.c if hasattr(trade, 'c') else [],
                'exchange': trade.x if hasattr(trade, 'x') else '',
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.metrics.record_api_request('alpaca', 'get_trade', 'error')
            self.logger.error(f"Failed to get trade for {symbol}: {e}")
            raise
    
    async def start_websocket_stream(self, symbols: List[str], 
                                   data_types: List[str] = None):
        """Start WebSocket streaming for symbols"""
        if not self.api_key or not self.secret_key:
            self.logger.error("Cannot start WebSocket: missing credentials")
            return False
        
        if data_types is None:
            data_types = ['trades', 'quotes']
        
        try:
            self.logger.info(f"Starting WebSocket stream for {symbols}")
            
            # Connect to WebSocket
            async with websockets.connect(
                self.stream_url,
                ping_interval=self.settings.data_collection.ws_ping_interval,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                self.websocket = websocket
                self.connection_start_time = time.time()
                
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.secret_key
                }
                await websocket.send(json.dumps(auth_msg))
                
                # Wait for auth response
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if auth_data[0].get('T') != 'success':
                    raise Exception(f"Authentication failed: {auth_data}")
                
                # Subscribe to data
                subscribe_msg = {"action": "subscribe"}
                for data_type in data_types:
                    subscribe_msg[data_type] = symbols
                
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for subscription confirmation
                sub_response = await websocket.recv()
                sub_data = json.loads(sub_response)
                
                self.subscribed_symbols.update(symbols)
                self.is_streaming = True
                self.reconnect_attempts = 0
                
                self.logger.info(f"WebSocket streaming started for {symbols}")
                
                # Start message processing loop
                await self._process_websocket_messages()
                
        except Exception as e:
            self.is_streaming = False
            self.logger.error(f"WebSocket streaming failed: {e}")
            
            # Attempt reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                await asyncio.sleep(self.settings.data_collection.ws_reconnect_delay)
                self.logger.info(f"Reconnection attempt {self.reconnect_attempts}")
                return await self.start_websocket_stream(symbols, data_types)
            
            return False
        
        return True
    
    async def _process_websocket_messages(self):
        """Process incoming WebSocket messages"""
        while self.is_streaming and self.websocket:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=30
                )
                
                receive_time = time.time_ns()
                self.last_message_time = receive_time
                self.message_count += 1
                
                # Parse and process message
                data = json.loads(message)
                await self._handle_stream_data(data, receive_time)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self.websocket.ping()
                except:
                    self.logger.warning("WebSocket ping failed, connection may be lost")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                break
                
            except Exception as e:
                self.logger.error(f"Error processing WebSocket message: {e}")
                self.metrics.record_error('alpaca_stream', 'message_processing')
    
    async def _handle_stream_data(self, data: List[Dict], receive_time: int):
        """Handle incoming stream data"""
        for item in data:
            msg_type = item.get('T')
            
            if msg_type == 't':  # Trade
                await self._process_trade_message(item, receive_time)
            elif msg_type == 'q':  # Quote
                await self._process_quote_message(item, receive_time)
            elif msg_type == 'b':  # Bar
                await self._process_bar_message(item, receive_time)
    
    async def _process_trade_message(self, trade_data: Dict, receive_time: int):
        """Process trade message"""
        try:
            trade = {
                'symbol': trade_data['S'],
                'timestamp_ns': trade_data['t'],
                'price': Decimal(str(trade_data['p'])),
                'size': trade_data['s'],
                'conditions': trade_data.get('c', []),
                'exchange': trade_data.get('x', ''),
                'tape': trade_data.get('z', ''),
                'receive_time_ns': receive_time,
                'source': 'alpaca_stream'
            }
            
            # Calculate latency
            latency_ns = receive_time - trade_data['t']
            latency_us = latency_ns / 1000
            
            self.metrics.record_data_latency('alpaca', 'trade_stream', latency_us)
            self.metrics.record_data_point('alpaca', trade['symbol'], 'trade')
            
            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    await callback(trade)
                except Exception as e:
                    self.logger.error(f"Trade callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade message: {e}")
            self.metrics.record_error('alpaca_stream', 'trade_processing')
    
    async def _process_quote_message(self, quote_data: Dict, receive_time: int):
        """Process quote message"""
        try:
            quote = {
                'symbol': quote_data['S'],
                'timestamp_ns': quote_data['t'],
                'bid': Decimal(str(quote_data['bp'])),
                'ask': Decimal(str(quote_data['ap'])),
                'bid_size': quote_data['bs'],
                'ask_size': quote_data['as'],
                'conditions': quote_data.get('c', []),
                'exchange': quote_data.get('x', ''),
                'tape': quote_data.get('z', ''),
                'receive_time_ns': receive_time,
                'source': 'alpaca_stream'
            }
            
            # Calculate spread and mid
            quote['spread'] = quote['ask'] - quote['bid']
            quote['mid'] = (quote['ask'] + quote['bid']) / 2
            
            # Calculate latency
            latency_ns = receive_time - quote_data['t']
            latency_us = latency_ns / 1000
            
            self.metrics.record_data_latency('alpaca', 'quote_stream', latency_us)
            self.metrics.record_data_point('alpaca', quote['symbol'], 'quote')
            
            # Trigger callbacks
            for callback in self.quote_callbacks:
                try:
                    await callback(quote)
                except Exception as e:
                    self.logger.error(f"Quote callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing quote message: {e}")
            self.metrics.record_error('alpaca_stream', 'quote_processing')
    
    async def _process_bar_message(self, bar_data: Dict, receive_time: int):
        """Process bar message"""
        try:
            bar = {
                'symbol': bar_data['S'],
                'timestamp': datetime.fromtimestamp(bar_data['t'] / 1000000000, tz=timezone.utc),
                'open': float(bar_data['o']),
                'high': float(bar_data['h']),
                'low': float(bar_data['l']),
                'close': float(bar_data['c']),
                'volume': bar_data['v'],
                'trade_count': bar_data.get('n', 0),
                'vwap': float(bar_data.get('vw', 0)),
                'receive_time_ns': receive_time,
                'source': 'alpaca_stream'
            }
            
            # Calculate latency
            latency_ns = receive_time - bar_data['t']
            latency_us = latency_ns / 1000
            
            self.metrics.record_data_latency('alpaca', 'bar_stream', latency_us)
            self.metrics.record_data_point('alpaca', bar['symbol'], 'bar')
            
            self.logger.debug(f"Received bar for {bar['symbol']}: ${bar['close']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing bar message: {e}")
            self.metrics.record_error('alpaca_stream', 'bar_processing')
    
    def _map_timeframe(self, timeframe: str):
        """Map timeframe string to Alpaca TimeFrame"""
        mapping = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day,
            '1Week': TimeFrame.Week,
            '1Month': TimeFrame.Month
        }
        return mapping.get(timeframe, TimeFrame.Day)
    
    async def stop_websocket_stream(self):
        """Stop WebSocket streaming"""
        self.is_streaming = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.subscribed_symbols.clear()
        self.logger.info("WebSocket streaming stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        if not self.api:
            return {
                'status': 'critical',
                'message': 'API not initialized',
                'metrics': {}
            }
        
        try:
            # Test API connectivity
            start_time = time.time()
            account = self.api.get_account()
            response_time = (time.time() - start_time) * 1000
            
            # Check WebSocket status
            ws_status = 'connected' if self.is_streaming else 'disconnected'
            ws_uptime = (time.time() - self.connection_start_time) if self.connection_start_time else 0
            
            return {
                'status': 'healthy',
                'message': f'Account status: {account.status}',
                'metrics': {
                    'api_response_time_ms': response_time,
                    'websocket_status': ws_status,
                    'websocket_uptime_seconds': ws_uptime,
                    'subscribed_symbols': len(self.subscribed_symbols),
                    'messages_received': self.message_count,
                    'reconnect_attempts': self.reconnect_attempts
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Health check failed: {str(e)}',
                'metrics': {
                    'error': str(e),
                    'websocket_status': 'disconnected' if not self.is_streaming else 'unknown'
                }
            }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'is_streaming': self.is_streaming,
            'subscribed_symbols': list(self.subscribed_symbols),
            'connection_uptime': (time.time() - self.connection_start_time) if self.connection_start_time else 0,
            'message_count': self.message_count,
            'reconnect_attempts': self.reconnect_attempts,
            'last_message_time': self.last_message_time
        }

# ===================== GLOBAL INSTANCE =====================

_alpaca_collector = None

def get_alpaca_collector() -> AlpacaDataCollector:
    """Get global Alpaca collector instance"""
    global _alpaca_collector
    if _alpaca_collector is None:
        _alpaca_collector = AlpacaDataCollector()
    return _alpaca_collector

async def initialize_alpaca_collector():
    """Initialize Alpaca collector"""
    collector = get_alpaca_collector()
    
    # Register health check
    from infrastructure.monitoring import get_health_monitor
    health_monitor = get_health_monitor()
    health_monitor.register_health_check('alpaca_collector', collector.health_check)
    
    return collector

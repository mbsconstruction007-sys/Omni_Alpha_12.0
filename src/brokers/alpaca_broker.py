"""
Alpaca broker implementation for US markets
Supports both paper and live trading
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import structlog
from urllib.parse import urlencode
import time

from src.brokers.base import BaseBroker, BrokerConfig, BrokerStatus
from src.database.models import (
    Order, Trade, Position, Account,
    OrderStatus, OrderType, OrderSide, TimeInForce, AssetType, ExchangeType
)

logger = structlog.get_logger()

class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation with WebSocket support
    Production-ready with comprehensive error handling
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Set WebSocket URLs based on environment
        if config.paper_trading:
            self.ws_url = "wss://stream.data.sandbox.alpaca.markets/v2/test"
            self.trade_ws_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            self.trade_ws_url = "wss://api.alpaca.markets/stream"
            
        self._trade_ws = None
        self._data_ws = None
        self._ws_tasks = []
        
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            self.status = BrokerStatus.INITIALIZING
            
            # Create HTTP session with authentication headers
            self._session = aiohttp.ClientSession(
                headers={
                    'APCA-API-KEY-ID': self.config.api_key,
                    'APCA-API-SECRET-KEY': self.config.secret_key,
                    'Content-Type': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(limit=self.config.max_connections)
            )
            
            # Test connection by getting account info
            async with self._session.get(f"{self.config.base_url}/v2/account") as response:
                if response.status == 200:
                    account_data = await response.json()
                    self.config.account_id = account_data.get('id')
                    
                    self.status = BrokerStatus.CONNECTED
                    self.metrics.uptime_start = datetime.now()
                    
                    logger.info(f"Connected to Alpaca {'Paper' if self.config.paper_trading else 'Live'} API",
                              account_id=self.config.account_id,
                              buying_power=account_data.get('buying_power'))
                    
                    # Start WebSocket connections
                    self._ws_tasks.append(asyncio.create_task(self._connect_trade_stream()))
                    self._ws_tasks.append(asyncio.create_task(self._connect_data_stream()))
                    
                    # Start heartbeat
                    await self._start_heartbeat()
                    
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to connect to Alpaca: {error}")
                    self.status = BrokerStatus.ERROR
                    self.metrics.last_error = error
                    self.metrics.last_error_time = datetime.now()
                    return False
                    
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            self.status = BrokerStatus.ERROR
            self.metrics.connection_errors += 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca"""
        try:
            self._is_closing = True
            
            # Cancel WebSocket tasks
            for task in self._ws_tasks:
                task.cancel()
                
            # Close WebSocket connections
            if self._trade_ws:
                await self._trade_ws.close()
            if self._data_ws:
                await self._data_ws.close()
                
            # Close HTTP session
            if self._session:
                await self._session.close()
                
            self.status = BrokerStatus.DISCONNECTED
            logger.info("Disconnected from Alpaca")
            return True
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            return False
    
    async def place_order(self, order: Order) -> Order:
        """Place order on Alpaca"""
        start_time = time.time()
        
        try:
            # Validate order
            await self.validate_order(order)
            
            # Rate limiting
            await self._rate_limiter.wait_for_token()
            
            # Generate client order ID if not provided
            if not order.client_order_id:
                order.client_order_id = f"OMNI_{uuid.uuid4().hex[:8]}"
            
            # Prepare order payload
            payload = {
                'symbol': order.symbol,
                'qty': str(order.quantity),
                'side': order.side.value,
                'type': self._map_order_type(order.order_type),
                'time_in_force': self._map_time_in_force(order.time_in_force),
                'client_order_id': order.client_order_id,
            }
            
            # Add price fields based on order type
            if order.order_type == OrderType.LIMIT:
                payload['limit_price'] = str(order.limit_price)
            elif order.order_type == OrderType.STOP:
                payload['stop_price'] = str(order.stop_price)
            elif order.order_type == OrderType.STOP_LIMIT:
                payload['limit_price'] = str(order.limit_price)
                payload['stop_price'] = str(order.stop_price)
            elif order.order_type == OrderType.TRAILING_STOP:
                if order.trail_price:
                    payload['trail_price'] = str(order.trail_price)
                elif order.trail_percent:
                    payload['trail_percent'] = str(order.trail_percent)
                    
            # Add extended hours flag if configured
            if self.config.environment == "production" and order.extended_hours:
                payload['extended_hours'] = True
                
            # Submit order
            async with self._session.post(
                f"{self.config.base_url}/v2/orders",
                json=payload
            ) as response:
                response_time = (time.time() - start_time) * 1000
                self.metrics.average_latency_ms = (
                    (self.metrics.average_latency_ms * self.metrics.total_orders + response_time) /
                    (self.metrics.total_orders + 1)
                )
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    # Update order with broker response
                    order.order_id = data['id']
                    order.status = self._map_order_status(data['status'])
                    order.submitted_at = datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00'))
                    
                    # Track order
                    self._pending_orders[order.order_id] = order
                    
                    # Update metrics
                    self.metrics.total_orders += 1
                    self.metrics.successful_orders += 1
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('order_placed', order)
                    
                    logger.info(f"Order placed successfully",
                              order_id=order.order_id,
                              symbol=order.symbol,
                              side=order.side.value,
                              quantity=float(order.quantity),
                              latency_ms=response_time)
                    
                    return order
                    
                else:
                    error = await response.text()
                    self.metrics.total_orders += 1
                    self.metrics.failed_orders += 1
                    self.metrics.api_errors += 1
                    
                    # Parse Alpaca error
                    try:
                        error_data = json.loads(error)
                        error_message = error_data.get('message', error)
                    except:
                        error_message = error
                        
                    raise Exception(f"Order placement failed: {error_message}")
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca"""
        try:
            await self._rate_limiter.wait_for_token()
            
            async with self._session.delete(
                f"{self.config.base_url}/v2/orders/{order_id}"
            ) as response:
                if response.status in [200, 204]:
                    # Remove from pending orders
                    self._pending_orders.pop(order_id, None)
                    
                    # Update metrics
                    self.metrics.cancelled_orders += 1
                    
                    await self._trigger_callbacks('order_cancelled', order_id)
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to cancel order {order_id}: {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def modify_order(self, order_id: str, modifications: Dict) -> Order:
        """Modify existing order"""
        try:
            await self._rate_limiter.wait_for_token()
            
            payload = {}
            
            # Map modifications to Alpaca format
            if 'quantity' in modifications:
                payload['qty'] = str(modifications['quantity'])
            if 'limit_price' in modifications:
                payload['limit_price'] = str(modifications['limit_price'])
            if 'stop_price' in modifications:
                payload['stop_price'] = str(modifications['stop_price'])
            if 'time_in_force' in modifications:
                payload['time_in_force'] = self._map_time_in_force(modifications['time_in_force'])
                
            async with self._session.patch(
                f"{self.config.base_url}/v2/orders/{order_id}",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    order = self._parse_order(data)
                    
                    # Update pending order
                    if order_id in self._pending_orders:
                        self._pending_orders[order_id] = order
                        
                    await self._trigger_callbacks('order_modified', order)
                    logger.info(f"Order modified: {order_id}")
                    return order
                else:
                    error = await response.text()
                    raise Exception(f"Order modification failed: {error}")
                    
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            raise
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        try:
            await self._rate_limiter.wait_for_token()
            
            async with self._session.get(
                f"{self.config.base_url}/v2/orders/{order_id}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_order(data)
                elif response.status == 404:
                    return None
                else:
                    error = await response.text()
                    logger.error(f"Error getting order {order_id}: {error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get list of orders"""
        try:
            await self._rate_limiter.wait_for_token()
            
            params = {'limit': 500}
            if status:
                # Map to Alpaca status
                alpaca_status = self._map_order_status_to_alpaca(status)
                params['status'] = alpaca_status
                
            async with self._session.get(
                f"{self.config.base_url}/v2/orders",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_order(order_data) for order_data in data]
                else:
                    error = await response.text()
                    logger.error(f"Error getting orders: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            await self._rate_limiter.wait_for_token()
            
            async with self._session.get(
                f"{self.config.base_url}/v2/positions"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for pos_data in data:
                        position = Position(
                            position_id=pos_data['asset_id'],
                            account_id=self.config.account_id or pos_data.get('account_id', ''),
                            symbol=pos_data['symbol'],
                            asset_type=AssetType.STOCK,
                            quantity=Decimal(pos_data['qty']),
                            available_quantity=Decimal(pos_data['qty_available']),
                            average_entry_price=Decimal(pos_data['avg_entry_price']),
                            current_price=Decimal(pos_data.get('current_price', 0)) if pos_data.get('current_price') else Decimal('0'),
                            market_value=Decimal(pos_data['market_value']),
                            unrealized_pnl=Decimal(pos_data['unrealized_pl']),
                            realized_pnl=Decimal(pos_data.get('realized_pl', 0)),
                            total_pnl=Decimal(pos_data['unrealized_pl']) + Decimal(pos_data.get('realized_pl', 0)),
                            pnl_percentage=Decimal(str(float(pos_data.get('unrealized_plpc', 0)) * 100)),
                            opened_at=datetime.now(),
                            last_modified=datetime.now()
                        )
                        positions.append(position)
                        
                    return positions
                else:
                    error = await response.text()
                    logger.error(f"Error getting positions: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_account(self) -> Account:
        """Get account information"""
        try:
            await self._rate_limiter.wait_for_token()
            
            async with self._session.get(
                f"{self.config.base_url}/v2/account"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return Account(
                        account_id=data['id'],
                        account_type='margin' if data.get('account_type') == 'margin' else 'cash',
                        cash_balance=Decimal(data['cash']),
                        buying_power=Decimal(data['buying_power']),
                        portfolio_value=Decimal(data['portfolio_value']),
                        margin_used=Decimal(data.get('initial_margin', 0)),
                        margin_available=Decimal(data.get('regt_buying_power', 0)),
                        maintenance_margin=Decimal(data.get('maintenance_margin', 0)),
                        daily_pnl=Decimal('0'),  # Calculate from positions
                        total_pnl=Decimal('0'),  # Calculate from positions
                        risk_score=0.0,
                        day_trade_count=data.get('daytrade_count', 0),
                        pattern_day_trader=data.get('pattern_day_trader', False),
                        active=data['status'] == 'ACTIVE',
                        restricted=data.get('trading_blocked', False) or data.get('account_blocked', False),
                        restriction_reason='Trading blocked' if data.get('trading_blocked') else None
                    )
                else:
                    error = await response.text()
                    logger.error(f"Error getting account: {error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    async def get_trades(self, start_date: datetime = None) -> List[Trade]:
        """Get trade history"""
        try:
            await self._rate_limiter.wait_for_token()
            
            params = {'activity_types': 'FILL', 'page_size': 100}
            if start_date:
                params['after'] = start_date.isoformat() + 'Z'
                
            async with self._session.get(
                f"{self.config.base_url}/v2/account/activities",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    trades = []
                    
                    for trade_data in data:
                        if trade_data.get('activity_type') == 'FILL':
                            trade = Trade(
                                trade_id=trade_data['id'],
                                order_id=trade_data.get('order_id', ''),
                                symbol=trade_data['symbol'],
                                side=OrderSide.BUY if trade_data['side'] == 'buy' else OrderSide.SELL,
                                quantity=Decimal(trade_data['qty']),
                                price=Decimal(trade_data['price']),
                                commission=Decimal('0'),  # Alpaca has no commissions
                                fees=Decimal('0'),
                                executed_at=datetime.fromisoformat(trade_data['transaction_time'].replace('Z', '+00:00')),
                                exchange=ExchangeType.NASDAQ,  # Default
                                execution_id=trade_data['id']
                            )
                            trades.append(trade)
                            
                            # Update metrics
                            self.metrics.total_trades += 1
                            self.metrics.total_volume += trade.quantity * trade.price
                            
                    return trades
                else:
                    error = await response.text()
                    logger.error(f"Error getting trades: {error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    async def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to real-time market data"""
        try:
            if self._data_ws and not self._data_ws.closed:
                await self._data_ws.send_json({
                    'action': 'subscribe',
                    'trades': symbols,
                    'quotes': symbols,
                    'bars': symbols
                })
                logger.info(f"Subscribed to market data for {symbols}")
                return True
            else:
                logger.warning("Data WebSocket not connected")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            return False
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data"""
        try:
            if self._data_ws and not self._data_ws.closed:
                await self._data_ws.send_json({
                    'action': 'unsubscribe',
                    'trades': symbols,
                    'quotes': symbols,
                    'bars': symbols
                })
                logger.info(f"Unsubscribed from market data for {symbols}")
                return True
            else:
                logger.warning("Data WebSocket not connected")
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing from market data: {e}")
            return False
    
    # ============================================
    # WebSocket connections
    # ============================================
    
    async def _connect_trade_stream(self):
        """Connect to trade updates WebSocket"""
        while not self._is_closing:
            try:
                session = aiohttp.ClientSession()
                ws = await session.ws_connect(self.trade_ws_url)
                self._trade_ws = ws
                
                # Authenticate
                await ws.send_json({
                    'action': 'auth',
                    'key': self.config.api_key,
                    'secret': self.config.secret_key
                })
                
                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if isinstance(data, list):
                            for item in data:
                                if item.get('stream') == 'authorization':
                                    if item['data']['status'] == 'authorized':
                                        logger.info("Trade stream authorized")
                                        await ws.send_json({
                                            'action': 'listen',
                                            'data': {'streams': ['trade_updates']}
                                        })
                                elif item.get('stream') == 'trade_updates':
                                    await self._handle_trade_update(item['data'])
                                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Trade WebSocket error: {ws.exception()}")
                        break
                        
            except Exception as e:
                logger.error(f"Trade stream error: {e}")
                
            # Cleanup
            if ws:
                await ws.close()
            if session:
                await session.close()
                
            # Reconnect delay
            if not self._is_closing:
                await asyncio.sleep(self.config.reconnect_delay)
    
    async def _connect_data_stream(self):
        """Connect to market data WebSocket"""
        while not self._is_closing:
            try:
                session = aiohttp.ClientSession()
                ws = await session.ws_connect(self.ws_url)
                self._data_ws = ws
                
                # Authenticate
                await ws.send_json({
                    'action': 'auth',
                    'key': self.config.api_key,
                    'secret': self.config.secret_key
                })
                
                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if isinstance(data, list):
                            for item in data:
                                if item.get('msg') == 'authenticated':
                                    logger.info("Data stream authenticated")
                                elif item.get('T'):
                                    await self._handle_market_data(item)
                                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Data WebSocket error: {ws.exception()}")
                        break
                        
            except Exception as e:
                logger.error(f"Data stream error: {e}")
                
            # Cleanup
            if ws:
                await ws.close()
            if session:
                await session.close()
                
            # Reconnect delay
            if not self._is_closing:
                await asyncio.sleep(self.config.reconnect_delay)
    
    async def _handle_trade_update(self, data: Dict):
        """Handle trade update from WebSocket"""
        event_type = data.get('event')
        order_data = data.get('order', {})
        
        # Update pending order
        order_id = order_data.get('id')
        if order_id in self._pending_orders:
            order = self._pending_orders[order_id]
            order.status = self._map_order_status(order_data.get('status'))
            
            if event_type == 'fill':
                order.filled_quantity = Decimal(order_data.get('filled_qty', 0))
                order.average_fill_price = Decimal(order_data.get('filled_avg_price', 0))
                order.filled_at = datetime.now()
                
                # Update metrics
                self.metrics.total_trades += 1
                
                await self._trigger_callbacks('order_filled', order)
                logger.info(f"Order filled: {order_id}")
                
            elif event_type == 'partial_fill':
                order.filled_quantity = Decimal(order_data.get('filled_qty', 0))
                order.average_fill_price = Decimal(order_data.get('filled_avg_price', 0))
                
                await self._trigger_callbacks('order_partial_fill', order)
                logger.info(f"Order partially filled: {order_id}")
                
            elif event_type == 'canceled':
                order.cancelled_at = datetime.now()
                self._pending_orders.pop(order_id, None)
                
                await self._trigger_callbacks('order_cancelled', order)
                logger.info(f"Order cancelled: {order_id}")
                
            elif event_type == 'rejected':
                self._pending_orders.pop(order_id, None)
                
                await self._trigger_callbacks('order_rejected', order)
                logger.warning(f"Order rejected: {order_id}")
    
    async def _handle_market_data(self, data: Dict):
        """Handle market data from WebSocket"""
        msg_type = data.get('T')
        
        if msg_type == 't':  # Trade
            await self._trigger_callbacks('trade', data)
        elif msg_type == 'q':  # Quote
            await self._trigger_callbacks('quote', data)
        elif msg_type == 'b':  # Bar
            await self._trigger_callbacks('bar', data)
    
    # ============================================
    # Mapping functions
    # ============================================
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to Alpaca"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.TRAILING_STOP: 'trailing_stop'
        }
        return mapping.get(order_type, 'market')
    
    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map internal TIF to Alpaca"""
        mapping = {
            TimeInForce.DAY: 'day',
            TimeInForce.GTC: 'gtc',
            TimeInForce.IOC: 'ioc',
            TimeInForce.FOK: 'fok',
            TimeInForce.GTD: 'gtd'
        }
        return mapping.get(tif, 'day')
    
    def _map_order_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca status to internal"""
        mapping = {
            'pending_new': OrderStatus.PENDING,
            'accepted': OrderStatus.SUBMITTED,
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'done_for_day': OrderStatus.CANCELLED,
            'stopped': OrderStatus.CANCELLED,
            'suspended': OrderStatus.PENDING,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)
    
    def _map_order_status_to_alpaca(self, status: OrderStatus) -> str:
        """Map internal status to Alpaca for filtering"""
        if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            return 'open'
        elif status == OrderStatus.FILLED:
            return 'closed'
        else:
            return 'all'
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse Alpaca order data to internal Order"""
        return Order(
            order_id=data['id'],
            client_order_id=data.get('client_order_id'),
            account_id=self.config.account_id,
            symbol=data['symbol'],
            asset_type=AssetType.STOCK,  # Alpaca is stocks only for now
            exchange=ExchangeType.NASDAQ,  # Default
            side=OrderSide.BUY if data['side'] == 'buy' else OrderSide.SELL,
            order_type=self._parse_order_type(data['order_type']),
            quantity=Decimal(data['qty']),
            filled_quantity=Decimal(data.get('filled_qty', 0)),
            remaining_quantity=Decimal(data['qty']) - Decimal(data.get('filled_qty', 0)),
            limit_price=Decimal(data['limit_price']) if data.get('limit_price') else None,
            stop_price=Decimal(data['stop_price']) if data.get('stop_price') else None,
            average_fill_price=Decimal(data['filled_avg_price']) if data.get('filled_avg_price') else None,
            time_in_force=self._parse_time_in_force(data['time_in_force']),
            status=self._map_order_status(data['status']),
            submitted_at=datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00')) if data.get('submitted_at') else None,
            filled_at=datetime.fromisoformat(data['filled_at'].replace('Z', '+00:00')) if data.get('filled_at') else None,
            cancelled_at=datetime.fromisoformat(data['canceled_at'].replace('Z', '+00:00')) if data.get('canceled_at') else None,
            expired_at=datetime.fromisoformat(data['expired_at'].replace('Z', '+00:00')) if data.get('expired_at') else None,
        )
    
    def _parse_order_type(self, alpaca_type: str) -> OrderType:
        """Parse Alpaca order type to internal"""
        mapping = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT,
            'trailing_stop': OrderType.TRAILING_STOP
        }
        return mapping.get(alpaca_type, OrderType.MARKET)
    
    def _parse_time_in_force(self, alpaca_tif: str) -> TimeInForce:
        """Parse Alpaca TIF to internal"""
        mapping = {
            'day': TimeInForce.DAY,
            'gtc': TimeInForce.GTC,
            'ioc': TimeInForce.IOC,
            'fok': TimeInForce.FOK,
            'gtd': TimeInForce.GTD,
            'opg': TimeInForce.DAY,  # Market on open
            'cls': TimeInForce.DAY,  # Market on close
        }
        return mapping.get(alpaca_tif.lower(), TimeInForce.DAY)

"""
Abstract base class for all broker implementations
World-class broker abstraction with failover support
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from dataclasses import dataclass, field
import structlog
from enum import Enum
import time
import uuid

from src.database.models import (
    Order, Trade, Position, Account,
    OrderStatus, OrderType, OrderSide, TimeInForce, AssetType
)

logger = structlog.get_logger()

class BrokerStatus(Enum):
    """Broker connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    INITIALIZING = "initializing"

@dataclass
class BrokerConfig:
    """Broker configuration with all necessary parameters"""
    name: str
    api_key: str
    secret_key: str
    base_url: str
    data_url: Optional[str] = None
    paper_trading: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_count: int = 3
    max_connections: int = 10
    heartbeat_interval: int = 30
    reconnect_delay: int = 5
    
    # Additional configurations
    access_token: Optional[str] = None  # For OAuth brokers
    account_id: Optional[str] = None
    environment: str = "development"  # development, staging, production
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.api_key or not self.secret_key:
            raise ValueError(f"API credentials required for {self.name}")
        if not self.base_url:
            raise ValueError(f"Base URL required for {self.name}")

@dataclass
class BrokerMetrics:
    """Broker performance metrics"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    cancelled_orders: int = 0
    total_trades: int = 0
    total_volume: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    connection_errors: int = 0
    api_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    uptime_start: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    average_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate order success rate"""
        if self.total_orders == 0:
            return 0.0
        return (self.successful_orders / self.total_orders) * 100
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds"""
        if not self.uptime_start:
            return 0.0
        return (datetime.now() - self.uptime_start).total_seconds()

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, period: int = 60):
        self.rate = rate  # requests per period
        self.period = period  # in seconds
        self.tokens = rate
        self.updated_at = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens for request"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.updated_at
            
            # Refill tokens
            self.tokens = min(
                self.rate,
                self.tokens + (elapsed * self.rate / self.period)
            )
            self.updated_at = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    async def wait_for_token(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)

class BaseBroker(ABC):
    """
    Abstract base class for broker implementations
    Provides common functionality and interface
    """
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.status = BrokerStatus.DISCONNECTED
        self.metrics = BrokerMetrics()
        self._session = None
        self._websocket = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._rate_limiter = RateLimiter(config.rate_limit)
        self._heartbeat_task = None
        self._reconnect_task = None
        self._is_closing = False
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._order_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(f"Initialized {self.config.name} broker", 
                   paper_trading=self.config.paper_trading)
    
    # ============================================
    # Abstract methods that must be implemented
    # ============================================
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def modify_order(self, order_id: str, modifications: Dict) -> Order:
        """Modify an existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get list of orders"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_trades(self, start_date: datetime = None) -> List[Trade]:
        """Get trade history"""
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to real-time market data"""
        pass
    
    @abstractmethod
    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data"""
        pass
    
    # ============================================
    # Common utility methods
    # ============================================
    
    async def validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        errors = []
        
        # Check required fields
        if not order.symbol:
            errors.append("Symbol is required")
        if not order.quantity or order.quantity <= 0:
            errors.append("Quantity must be positive")
        if not order.side:
            errors.append("Order side (buy/sell) is required")
        if not order.order_type:
            errors.append("Order type is required")
            
        # Check order type specific requirements
        if order.order_type == OrderType.LIMIT:
            if not order.limit_price or order.limit_price <= 0:
                errors.append("Limit price required for limit orders")
                
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                errors.append("Stop price required for stop orders")
                
        if order.order_type == OrderType.STOP_LIMIT:
            if not order.limit_price or order.limit_price <= 0:
                errors.append("Limit price required for stop limit orders")
                
        # Check time in force
        if not order.time_in_force:
            order.time_in_force = TimeInForce.DAY  # Default to DAY
            
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"Order validation failed: {error_msg}")
            raise ValueError(f"Invalid order: {error_msg}")
            
        return True
    
    async def is_market_open(self) -> bool:
        """Check if market is open for trading"""
        now = datetime.now()
        weekday = now.weekday()
        
        # Market closed on weekends
        if weekday >= 5:
            return False
        
        # Check US market hours (simplified)
        # Should be enhanced based on broker and market
        current_time = now.time()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Check regular trading hours
        if market_open <= current_time <= market_close:
            return True
            
        # Check extended hours if enabled
        if self.config.environment == "production":
            pre_market_open = datetime.strptime("04:00", "%H:%M").time()
            after_market_close = datetime.strptime("20:00", "%H:%M").time()
            
            if pre_market_open <= current_time <= after_market_close:
                return True
                
        return False
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for events"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")
    
    async def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error for {event}: {e}", exc_info=True)
    
    async def _start_heartbeat(self):
        """Start heartbeat task"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Heartbeat loop to maintain connection"""
        while not self._is_closing:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send heartbeat
                if self.status == BrokerStatus.CONNECTED:
                    # Implementation specific heartbeat
                    self.metrics.last_heartbeat = datetime.now()
                    await self._trigger_callbacks('heartbeat', self.metrics.last_heartbeat)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _handle_reconnect(self):
        """Handle automatic reconnection"""
        if self._reconnect_task:
            return  # Already reconnecting
            
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """Reconnection loop"""
        attempts = 0
        max_attempts = self.config.retry_count
        
        while attempts < max_attempts and not self._is_closing:
            attempts += 1
            logger.info(f"Reconnection attempt {attempts}/{max_attempts}")
            
            try:
                # Wait before reconnecting
                await asyncio.sleep(self.config.reconnect_delay * attempts)
                
                # Try to reconnect
                self.status = BrokerStatus.RECONNECTING
                if await self.connect():
                    logger.info("Reconnection successful")
                    self.status = BrokerStatus.CONNECTED
                    self.metrics.connection_errors = 0
                    break
                    
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self.metrics.connection_errors += 1
                
        if attempts >= max_attempts:
            logger.error(f"Max reconnection attempts reached for {self.config.name}")
            self.status = BrokerStatus.ERROR
            await self._trigger_callbacks('connection_lost', self.config.name)
            
        self._reconnect_task = None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get broker metrics"""
        return {
            'broker': self.config.name,
            'status': self.status.value,
            'paper_trading': self.config.paper_trading,
            'metrics': {
                'total_orders': self.metrics.total_orders,
                'successful_orders': self.metrics.successful_orders,
                'failed_orders': self.metrics.failed_orders,
                'cancelled_orders': self.metrics.cancelled_orders,
                'success_rate': f"{self.metrics.success_rate:.2f}%",
                'total_trades': self.metrics.total_trades,
                'total_volume': float(self.metrics.total_volume),
                'total_commission': float(self.metrics.total_commission),
                'connection_errors': self.metrics.connection_errors,
                'api_errors': self.metrics.api_errors,
                'uptime_seconds': self.metrics.uptime_seconds,
                'average_latency_ms': self.metrics.average_latency_ms,
                'last_error': self.metrics.last_error,
                'last_heartbeat': self.metrics.last_heartbeat.isoformat() if self.metrics.last_heartbeat else None,
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'broker': self.config.name,
            'status': self.status.value,
            'healthy': self.status == BrokerStatus.CONNECTED,
            'checks': {}
        }
        
        # Check connection
        health['checks']['connection'] = self.status == BrokerStatus.CONNECTED
        
        # Check heartbeat
        if self.metrics.last_heartbeat:
            heartbeat_age = (datetime.now() - self.metrics.last_heartbeat).total_seconds()
            health['checks']['heartbeat'] = heartbeat_age < (self.config.heartbeat_interval * 2)
        else:
            health['checks']['heartbeat'] = False
            
        # Check error rate
        if self.metrics.total_orders > 0:
            error_rate = (self.metrics.failed_orders / self.metrics.total_orders) * 100
            health['checks']['error_rate'] = error_rate < 5  # Less than 5% error rate
        else:
            health['checks']['error_rate'] = True
            
        # Overall health
        health['healthy'] = all(health['checks'].values())
        
        return health
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, status={self.status.value})"

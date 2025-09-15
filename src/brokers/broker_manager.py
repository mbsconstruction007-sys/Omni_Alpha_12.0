"""
Unified broker manager for seamless broker switching and failover
Production-ready with comprehensive monitoring
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from decimal import Decimal
import structlog
from enum import Enum
import random

from src.brokers.base import BaseBroker, BrokerConfig, BrokerStatus
from src.brokers.alpaca_broker import AlpacaBroker
from src.database.models import Order, Trade, Position, Account

logger = structlog.get_logger()

class BrokerType(Enum):
    """Supported broker types"""
    ALPACA = "alpaca"
    UPSTOX = "upstox"
    IBKR = "ibkr"
    ZERODHA = "zerodha"
    MOCK = "mock"  # For testing

class RoutingStrategy(Enum):
    """Order routing strategies"""
    PRIMARY_ONLY = "primary_only"
    ROUND_ROBIN = "round_robin"
    BEST_EXECUTION = "best_execution"
    LOAD_BALANCED = "load_balanced"
    FAILOVER = "failover"

class BrokerManager:
    """
    Manages multiple brokers with automatic failover and smart routing
    Production-ready with monitoring and health checks
    """
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        self.brokers: Dict[BrokerType, BaseBroker] = {}
        self.primary_broker: Optional[BrokerType] = None
        self.fallback_brokers: List[BrokerType] = []
        self.routing_strategy = routing_strategy
        self._initialized = False
        self._round_robin_index = 0
        self._health_check_task = None
        self._health_check_interval = 30  # seconds
        
        # Performance tracking
        self.broker_performance: Dict[BrokerType, Dict[str, Any]] = {}
        
        logger.info(f"Broker manager initialized with {routing_strategy.value} routing")
    
    async def initialize(self, configs: Dict[BrokerType, BrokerConfig]):
        """Initialize all configured brokers"""
        if self._initialized:
            logger.warning("Broker manager already initialized")
            return
            
        initialization_tasks = []
        
        for broker_type, config in configs.items():
            initialization_tasks.append(self._initialize_broker(broker_type, config))
        
        # Initialize all brokers concurrently
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Process results
        for i, (broker_type, config) in enumerate(configs.items()):
            if isinstance(results[i], Exception):
                logger.error(f"Failed to initialize {broker_type.value}: {results[i]}")
            elif results[i]:
                logger.info(f"Successfully initialized {broker_type.value}")
                
                # Set primary broker if not set
                if not self.primary_broker:
                    self.primary_broker = broker_type
                    logger.info(f"Set {broker_type.value} as primary broker")
                else:
                    self.fallback_brokers.append(broker_type)
                    
                # Initialize performance tracking
                self.broker_performance[broker_type] = {
                    'total_orders': 0,
                    'successful_orders': 0,
                    'failed_orders': 0,
                    'average_latency_ms': 0.0,
                    'last_used': None,
                    'health_score': 100.0,
                }
        
        if not self.brokers:
            raise Exception("No brokers successfully initialized")
        
        self._initialized = True
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Broker manager initialized with {len(self.brokers)} brokers")
        logger.info(f"Primary: {self.primary_broker.value if self.primary_broker else 'None'}")
        logger.info(f"Fallbacks: {[b.value for b in self.fallback_brokers]}")
    
    async def _initialize_broker(self, broker_type: BrokerType, config: BrokerConfig) -> bool:
        """Initialize a single broker"""
        try:
            broker = self._create_broker(broker_type, config)
            
            if await broker.connect():
                self.brokers[broker_type] = broker
                
                # Register callbacks
                broker.register_callback('order_filled', self._on_order_filled)
                broker.register_callback('order_rejected', self._on_order_rejected)
                broker.register_callback('connection_lost', self._on_connection_lost)
                
                return True
            else:
                logger.error(f"Failed to connect to {broker_type.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing {broker_type.value}: {e}")
            return False
    
    def _create_broker(self, broker_type: BrokerType, config: BrokerConfig) -> BaseBroker:
        """Create broker instance based on type"""
        if broker_type == BrokerType.ALPACA:
            from src.brokers.alpaca_broker import AlpacaBroker
            return AlpacaBroker(config)
        elif broker_type == BrokerType.UPSTOX:
            from src.brokers.upstox_broker import UpstoxBroker
            return UpstoxBroker(config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
    
    def _select_broker(self, preferred_broker: Optional[BrokerType] = None) -> Optional[BrokerType]:
        """Select broker based on routing strategy"""
        if preferred_broker and preferred_broker in self.brokers:
            broker = self.brokers[preferred_broker]
            if broker.status == BrokerStatus.CONNECTED:
                return preferred_broker
        
        available_brokers = [
            bt for bt, b in self.brokers.items()
            if b.status == BrokerStatus.CONNECTED
        ]
        
        if not available_brokers:
            return None
        
        if self.routing_strategy == RoutingStrategy.PRIMARY_ONLY:
            if self.primary_broker in available_brokers:
                return self.primary_broker
            return None
            
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            if available_brokers:
                broker_type = available_brokers[self._round_robin_index % len(available_brokers)]
                self._round_robin_index += 1
                return broker_type
                
        elif self.routing_strategy == RoutingStrategy.BEST_EXECUTION:
            # Select broker with best performance
            best_broker = None
            best_score = -1
            
            for broker_type in available_brokers:
                score = self.broker_performance[broker_type]['health_score']
                if score > best_score:
                    best_score = score
                    best_broker = broker_type
                    
            return best_broker
            
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            # Select broker with least recent usage
            least_used = None
            oldest_time = datetime.now()
            
            for broker_type in available_brokers:
                last_used = self.broker_performance[broker_type]['last_used']
                if last_used is None or last_used < oldest_time:
                    oldest_time = last_used or datetime.min
                    least_used = broker_type
                    
            return least_used
            
        elif self.routing_strategy == RoutingStrategy.FAILOVER:
            # Use primary if available, otherwise first fallback
            if self.primary_broker in available_brokers:
                return self.primary_broker
                
            for fallback in self.fallback_brokers:
                if fallback in available_brokers:
                    return fallback
        
        return None
    
    async def place_order(
        self,
        order: Order,
        broker_type: Optional[BrokerType] = None
    ) -> Order:
        """
        Place order with automatic broker selection and failover
        """
        if not self._initialized:
            raise Exception("Broker manager not initialized")
        
        # Select broker
        selected_broker = self._select_broker(broker_type)
        
        if not selected_broker:
            raise Exception("No available brokers for order placement")
        
        broker = self.brokers[selected_broker]
        
        # Track performance
        start_time = datetime.now()
        self.broker_performance[selected_broker]['last_used'] = start_time
        
        try:
            # Place order
            result = await broker.place_order(order)
            
            # Update performance metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            perf = self.broker_performance[selected_broker]
            perf['total_orders'] += 1
            perf['successful_orders'] += 1
            perf['average_latency_ms'] = (
                (perf['average_latency_ms'] * (perf['total_orders'] - 1) + latency_ms) /
                perf['total_orders']
            )
            
            logger.info(f"Order placed via {selected_broker.value}",
                       order_id=result.order_id,
                       latency_ms=latency_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Order placement failed on {selected_broker.value}: {e}")
            
            # Update performance metrics
            perf = self.broker_performance[selected_broker]
            perf['total_orders'] += 1
            perf['failed_orders'] += 1
            perf['health_score'] *= 0.95  # Reduce health score
            
            # Try failover if configured
            if self.routing_strategy == RoutingStrategy.FAILOVER:
                for fallback_type in self.fallback_brokers:
                    if fallback_type != selected_broker and fallback_type in self.brokers:
                        fallback_broker = self.brokers[fallback_type]
                        if fallback_broker.status == BrokerStatus.CONNECTED:
                            try:
                                logger.info(f"Attempting failover to {fallback_type.value}")
                                result = await fallback_broker.place_order(order)
                                
                                # Update failover broker performance
                                self.broker_performance[fallback_type]['total_orders'] += 1
                                self.broker_performance[fallback_type]['successful_orders'] += 1
                                
                                logger.info(f"Order placed via failover broker {fallback_type.value}")
                                return result
                                
                            except Exception as fallback_error:
                                logger.error(f"Failover to {fallback_type.value} failed: {fallback_error}")
            
            raise e
    
    async def cancel_order(
        self,
        order_id: str,
        broker_type: Optional[BrokerType] = None
    ) -> bool:
        """Cancel order"""
        broker_type = broker_type or self.primary_broker
        
        if broker_type not in self.brokers:
            logger.error(f"Broker {broker_type.value} not available")
            return False
        
        broker = self.brokers[broker_type]
        return await broker.cancel_order(order_id)
    
    async def get_positions(
        self,
        broker_type: Optional[BrokerType] = None
    ) -> List[Position]:
        """Get positions from specific or all brokers"""
        if broker_type:
            if broker_type not in self.brokers:
                return []
            broker = self.brokers[broker_type]
            if broker.status == BrokerStatus.CONNECTED:
                return await broker.get_positions()
            return []
        
        # Get positions from all connected brokers
        all_positions = []
        
        for broker_type, broker in self.brokers.items():
            if broker.status == BrokerStatus.CONNECTED:
                try:
                    positions = await broker.get_positions()
                    all_positions.extend(positions)
                except Exception as e:
                    logger.error(f"Error getting positions from {broker_type.value}: {e}")
        
        return all_positions
    
    async def get_account(
        self,
        broker_type: Optional[BrokerType] = None
    ) -> Optional[Account]:
        """Get account information"""
        broker_type = broker_type or self.primary_broker
        
        if broker_type not in self.brokers:
            return None
        
        broker = self.brokers[broker_type]
        if broker.status == BrokerStatus.CONNECTED:
            return await broker.get_account()
        return None
    
    async def get_consolidated_account(self) -> Account:
        """Get consolidated account across all brokers"""
        accounts = []
        
        for broker_type, broker in self.brokers.items():
            if broker.status == BrokerStatus.CONNECTED:
                try:
                    account = await broker.get_account()
                    if account:
                        accounts.append(account)
                except Exception as e:
                    logger.error(f"Error getting account from {broker_type.value}: {e}")
        
        if not accounts:
            return None
        
        # Consolidate accounts
        total_cash = sum(a.cash_balance for a in accounts)
        total_buying_power = sum(a.buying_power for a in accounts)
        total_portfolio_value = sum(a.portfolio_value for a in accounts)
        total_margin_used = sum(a.margin_used for a in accounts)
        total_daily_pnl = sum(a.daily_pnl for a in accounts)
        total_pnl = sum(a.total_pnl for a in accounts)
        
        return Account(
            account_id="CONSOLIDATED",
            account_type="multi-broker",
            cash_balance=total_cash,
            buying_power=total_buying_power,
            portfolio_value=total_portfolio_value,
            margin_used=total_margin_used,
            margin_available=Decimal("0"),
            maintenance_margin=Decimal("0"),
            daily_pnl=total_daily_pnl,
            total_pnl=total_pnl,
            risk_score=0.0,
            day_trade_count=0,
            pattern_day_trader=False,
            active=True,
            restricted=False
        )
    
    async def subscribe_market_data(
        self,
        symbols: List[str],
        broker_type: Optional[BrokerType] = None
    ) -> bool:
        """Subscribe to market data"""
        broker_type = broker_type or self.primary_broker
        
        if broker_type not in self.brokers:
            return False
        
        broker = self.brokers[broker_type]
        if broker.status == BrokerStatus.CONNECTED:
            return await broker.subscribe_market_data(symbols)
        return False
    
    async def get_broker_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all brokers"""
        status = {
            'initialized': self._initialized,
            'routing_strategy': self.routing_strategy.value,
            'primary': self.primary_broker.value if self.primary_broker else None,
            'fallbacks': [b.value for b in self.fallback_brokers],
            'brokers': {},
            'performance': self.broker_performance,
        }
        
        for broker_type, broker in self.brokers.items():
            # Get broker metrics
            metrics = await broker.get_metrics()
            
            # Get broker health
            health = await broker.health_check()
            
            status['brokers'][broker_type.value] = {
                'status': broker.status.value,
                'healthy': health['healthy'],
                'metrics': metrics['metrics'],
                'health_checks': health['checks'],
            }
        
        return status
    
    async def _health_check_loop(self):
        """Periodic health check for all brokers"""
        while self._initialized:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                for broker_type, broker in self.brokers.items():
                    try:
                        health = await broker.health_check()
                        
                        # Update health score
                        if health['healthy']:
                            self.broker_performance[broker_type]['health_score'] = min(
                                100.0,
                                self.broker_performance[broker_type]['health_score'] * 1.02
                            )
                        else:
                            self.broker_performance[broker_type]['health_score'] *= 0.9
                            
                            # Attempt reconnection if disconnected
                            if broker.status == BrokerStatus.DISCONNECTED:
                                logger.info(f"Attempting to reconnect {broker_type.value}")
                                await broker.connect()
                                
                    except Exception as e:
                        logger.error(f"Health check failed for {broker_type.value}: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _on_order_filled(self, order: Order):
        """Callback for order filled events"""
        logger.info(f"Order filled callback", order_id=order.order_id)
    
    async def _on_order_rejected(self, order: Order):
        """Callback for order rejected events"""
        logger.warning(f"Order rejected callback", order_id=order.order_id)
    
    async def _on_connection_lost(self, broker_name: str):
        """Callback for connection lost events"""
        logger.error(f"Connection lost for broker: {broker_name}")
        
        # Check if this was the primary broker
        if self.primary_broker and self.brokers[self.primary_broker].config.name == broker_name:
            # Promote first available fallback to primary
            for fallback in self.fallback_brokers:
                if self.brokers[fallback].status == BrokerStatus.CONNECTED:
                    logger.info(f"Promoting {fallback.value} to primary broker")
                    self.primary_broker = fallback
                    self.fallback_brokers.remove(fallback)
                    break
    
    async def shutdown(self):
        """Shutdown all brokers and cleanup"""
        logger.info("Shutting down broker manager")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all brokers
        disconnection_tasks = []
        for broker_type, broker in self.brokers.items():
            disconnection_tasks.append(broker.disconnect())
        
        # Wait for all disconnections
        await asyncio.gather(*disconnection_tasks, return_exceptions=True)
        
        # Clear brokers
        self.brokers.clear()
        self._initialized = False
        
        logger.info("Broker manager shutdown complete")

"""
Order Executor - Handles order submission and execution
Interfaces with brokers and manages execution quality
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import logging

from .models import Order, OrderType, OrderStatus, ExecutionVenue
from .router import SmartOrderRouter
try:
    from src.brokers import BrokerManager, BrokerType
    from src.database.models import Order as DBOrder
except ImportError:
    # For testing purposes, create mock classes
    class BrokerManager:
        pass
    class BrokerType:
        ALPACA = "alpaca"
        UPSTOX = "upstox"
    class DBOrder:
        pass

logger = logging.getLogger(__name__)

class OrderExecutor:
    """Order execution engine"""
    
    def __init__(self, broker_manager: BrokerManager, config: Dict[str, Any]):
        self.broker_manager = broker_manager
        self.config = config
        
        # Smart order router
        self.router = SmartOrderRouter(broker_manager)
        
        # Execution settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.execution_timeout = config.get('execution_timeout', 30)
        
        # Execution metrics
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_latency_ms': 0,
            'total_slippage': Decimal('0')
        }

    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to broker"""
        start_time = datetime.utcnow()
        
        try:
            # Select execution venue
            if order.venue == ExecutionVenue.SMART:
                venue = await self.router.select_venue(order)
            else:
                venue = order.venue
            
            # Get broker for venue
            broker = await self._get_broker_for_venue(venue)
            if not broker:
                raise ValueError(f"Broker not available for venue: {venue}")
            
            # Prepare order parameters
            params = self._prepare_order_params(order)
            
            # Submit with retries
            result = await self._submit_with_retry(broker, params)
            
            # Record execution metrics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_execution_stats('success', latency)
            
            logger.info(f"Order submitted successfully: {order.order_id} via {venue}")
            
            return {
                'order_id': result.get('id'),
                'venue': venue.value,
                'latency_ms': latency,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            self._update_execution_stats('failed', 0)
            raise

    async def modify_order(self, order: Order) -> Dict[str, Any]:
        """Modify existing order"""
        broker_order_id = order.metadata.get('broker_order_id')
        if not broker_order_id:
            raise ValueError("Broker order ID not found")
        
        venue = ExecutionVenue(order.metadata.get('venue', 'alpaca'))
        broker = await self._get_broker_for_venue(venue)
        
        if not broker:
            raise ValueError(f"Broker not available for venue: {venue}")
        
        try:
            # Prepare modification parameters
            params = {
                'qty': str(order.quantity),
                'limit_price': str(order.limit_price) if order.limit_price else None,
                'stop_price': str(order.stop_price) if order.stop_price else None,
                'time_in_force': order.time_in_force.value
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Submit modification
            result = await broker.modify_order(broker_order_id, params)
            
            logger.info(f"Order modified: {order.order_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to modify order {order.order_id}: {e}")
            raise

    async def cancel_order(self, order: Order) -> Dict[str, Any]:
        """Cancel order at broker"""
        broker_order_id = order.metadata.get('broker_order_id')
        if not broker_order_id:
            raise ValueError("Broker order ID not found")
        
        venue = ExecutionVenue(order.metadata.get('venue', 'alpaca'))
        broker = await self._get_broker_for_venue(venue)
        
        if not broker:
            raise ValueError(f"Broker not available for venue: {venue}")
        
        try:
            result = await broker.cancel_order(broker_order_id)
            logger.info(f"Order cancelled: {order.order_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            raise

    async def get_order_status(self, order: Order) -> Optional[OrderStatus]:
        """Get current order status from broker"""
        broker_order_id = order.metadata.get('broker_order_id')
        if not broker_order_id:
            return None
        
        venue = ExecutionVenue(order.metadata.get('venue', 'alpaca'))
        broker = await self._get_broker_for_venue(venue)
        
        if not broker:
            return None
        
        try:
            broker_order = await broker.get_order(broker_order_id)
            
            if broker_order:
                # Map broker status to our status
                status_map = {
                    'new': OrderStatus.NEW,
                    'accepted': OrderStatus.NEW,
                    'pending_new': OrderStatus.PENDING_NEW,
                    'partially_filled': OrderStatus.PARTIALLY_FILLED,
                    'filled': OrderStatus.FILLED,
                    'canceled': OrderStatus.CANCELLED,
                    'expired': OrderStatus.EXPIRED,
                    'rejected': OrderStatus.REJECTED,
                    'pending_cancel': OrderStatus.PENDING_CANCEL,
                    'pending_replace': OrderStatus.PENDING_REPLACE,
                    'replaced': OrderStatus.REPLACED
                }
                
                broker_status = broker_order.get('status', '').lower()
                return status_map.get(broker_status)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order status for {order.order_id}: {e}")
            return None

    async def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all orders from brokers"""
        all_orders = []
        
        try:
            # Get orders from all brokers
            brokers = await self.broker_manager.get_all_brokers()
            
            for broker_type, broker in brokers.items():
                if broker and broker.status.value == 'connected':
                    try:
                        orders = await broker.get_orders()
                        for order in orders:
                            order['venue'] = broker_type.value
                        all_orders.extend(orders)
                    except Exception as e:
                        logger.error(f"Error getting orders from {broker_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error getting all orders: {e}")
        
        return all_orders

    # Private methods
    
    async def _get_broker_for_venue(self, venue: ExecutionVenue):
        """Get broker instance for venue"""
        try:
            if venue == ExecutionVenue.ALPACA:
                return await self.broker_manager.get_broker(BrokerType.ALPACA)
            elif venue == ExecutionVenue.UPSTOX:
                return await self.broker_manager.get_broker(BrokerType.UPSTOX)
            else:
                # Default to primary broker
                return await self.broker_manager.get_primary_broker()
        except Exception as e:
            logger.error(f"Error getting broker for venue {venue}: {e}")
            return None
    
    def _prepare_order_params(self, order: Order) -> Dict[str, Any]:
        """Prepare order parameters for broker API"""
        params = {
            'symbol': order.symbol,
            'qty': str(order.quantity),
            'side': order.side.value,
            'type': order.order_type.value,
            'time_in_force': order.time_in_force.value,
            'extended_hours': order.extended_hours,
            'client_order_id': order.client_order_id
        }
        
        # Add price parameters based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            params['limit_price'] = str(order.limit_price)
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            params['stop_price'] = str(order.stop_price)
        
        if order.order_type == OrderType.TRAILING_STOP:
            if order.trail_percent:
                params['trail_percent'] = str(order.trail_percent)
            else:
                params['trail_price'] = str(order.trail_amount)
        
        # Handle bracket orders
        if order.order_type == OrderType.BRACKET:
            params['order_class'] = 'bracket'
            params['take_profit'] = {
                'limit_price': str(order.metadata.get('take_profit_price'))
            }
            params['stop_loss'] = {
                'stop_price': str(order.metadata.get('stop_loss_price'))
            }
        
        # Handle OCO orders
        if order.order_type == OrderType.OCO:
            params['order_class'] = 'oco'
            params['take_profit'] = {
                'limit_price': str(order.metadata.get('limit_price'))
            }
            params['stop_loss'] = {
                'stop_price': str(order.metadata.get('stop_price'))
            }
        
        return params

    async def _submit_with_retry(
        self,
        broker: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit order with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Submit order with timeout
                result = await asyncio.wait_for(
                    broker.place_order(params),
                    timeout=self.execution_timeout
                )
                return result
                
            except asyncio.TimeoutError:
                last_error = "Order submission timeout"
                logger.warning(f"Order submission timeout, attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order submission failed, attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"Order submission failed after {self.max_retries} attempts: {last_error}")

    def _update_execution_stats(self, result: str, latency: float):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if result == 'success':
            self.execution_stats['successful_orders'] += 1
        else:
            self.execution_stats['failed_orders'] += 1
        
        if latency > 0:
            self.execution_stats['total_latency_ms'] += latency

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = self.execution_stats['total_orders']
        
        if total == 0:
            return self.execution_stats
        
        return {
            **self.execution_stats,
            'success_rate': self.execution_stats['successful_orders'] / total,
            'avg_latency_ms': self.execution_stats['total_latency_ms'] / total,
            'avg_slippage': self.execution_stats['total_slippage'] / total
        }

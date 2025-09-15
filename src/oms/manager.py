"""
Order Manager - Core order lifecycle management
Handles order creation, modification, cancellation, and state transitions
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from collections import defaultdict

from .models import Order, OrderStatus, OrderRequest, OrderUpdate, Fill, OrderType, TimeInForce
from .executor import OrderExecutor
from .risk_checker import RiskChecker
from .position_manager import PositionManager
try:
    from src.database.repositories.order_repository import OrderRepository
    from src.brokers import BrokerManager
except ImportError:
    # For testing purposes, create mock classes
    class OrderRepository:
        pass
    class BrokerManager:
        pass

logger = logging.getLogger(__name__)

class OrderManager:
    """Central order management system"""
    
    def __init__(
        self,
        executor: OrderExecutor,
        risk_checker: RiskChecker,
        position_manager: PositionManager,
        repository: OrderRepository,
        broker_manager: BrokerManager
    ):
        self.executor = executor
        self.risk_checker = risk_checker
        self.position_manager = position_manager
        self.repository = repository
        self.broker_manager = broker_manager
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.pending_orders: asyncio.Queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'rejected': 0,
            'cancelled': 0,
            'avg_latency_ms': 0
        })
        
        # State machine
        self.valid_transitions = {
            OrderStatus.PENDING_NEW: [OrderStatus.NEW, OrderStatus.REJECTED],
            OrderStatus.NEW: [
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
                OrderStatus.PENDING_REPLACE
            ],
            OrderStatus.PARTIALLY_FILLED: [
                OrderStatus.FILLED,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.CANCELLED
            ],
            OrderStatus.PENDING_CANCEL: [OrderStatus.CANCELLED, OrderStatus.REJECTED],
            OrderStatus.PENDING_REPLACE: [OrderStatus.REPLACED, OrderStatus.REJECTED],
            # Terminal states
            OrderStatus.FILLED: [],
            OrderStatus.CANCELLED: [],
            OrderStatus.REJECTED: [],
            OrderStatus.EXPIRED: [],
            OrderStatus.REPLACED: []
        }
        
        self._running = False
        self._tasks = []

    async def start(self):
        """Start order manager"""
        logger.info("Starting Order Manager")
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._process_pending_orders()),
            asyncio.create_task(self._monitor_active_orders()),
            asyncio.create_task(self._check_order_expiry()),
            asyncio.create_task(self._reconcile_orders())
        ]
        
        # Load active orders from database
        await self._load_active_orders()
        
        logger.info("Order Manager started successfully")

    async def stop(self):
        """Stop order manager"""
        logger.info("Stopping Order Manager")
        self._running = False
        
        # Cancel all pending orders
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id, reason="System shutdown")
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Order Manager stopped")

    async def create_order(self, request: OrderRequest) -> Order:
        """Create and submit a new order"""
        start_time = datetime.utcnow()
        
        try:
            # Create order object
            order = Order(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_type=request.order_type,
                limit_price=request.limit_price,
                stop_price=request.stop_price,
                time_in_force=request.time_in_force,
                extended_hours=request.extended_hours,
                client_order_id=request.client_order_id or None,
                notes=request.notes,
                metadata=request.metadata
            )
            
            # Pre-trade risk checks
            risk_result = await self.risk_checker.check_order(order)
            order.risk_check_passed = risk_result.passed
            order.risk_check_details = risk_result.details
            
            if not risk_result.passed:
                order.status = OrderStatus.REJECTED
                await self._save_order(order)
                await self._emit_order_event(order, "rejected", risk_result.reason)
                raise ValueError(f"Order rejected: {risk_result.reason}")
            
            # Update position reservations
            await self.position_manager.reserve_for_order(order)
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Queue for execution
            await self.pending_orders.put(order)
            
            # Record metrics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_metrics(order.symbol, 'created', latency)
            
            # Emit event
            await self._emit_order_event(order, "created")
            
            logger.info(f"Order created: {order.order_id} - {order.symbol} {order.side} {order.quantity}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            self._update_metrics(request.symbol, 'failed', 0)
            raise

    async def modify_order(self, order_id: str, update: OrderUpdate) -> Order:
        """Modify an existing order"""
        order = self.active_orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        # Check if modification is allowed
        if order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot modify order in status {order.status}")
        
        # Validate state transition
        await self._transition_state(order, OrderStatus.PENDING_REPLACE)
        
        try:
            # Apply modifications
            if update.quantity:
                order.quantity = update.quantity
                order.remaining_quantity = order.quantity - order.filled_quantity
            
            if update.limit_price:
                order.limit_price = update.limit_price
            
            if update.stop_price:
                order.stop_price = update.stop_price
            
            if update.time_in_force:
                order.time_in_force = update.time_in_force
            
            # Re-run risk checks
            risk_result = await self.risk_checker.check_order(order)
            if not risk_result.passed:
                await self._transition_state(order, OrderStatus.REJECTED)
                raise ValueError(f"Modification rejected: {risk_result.reason}")
            
            # Submit modification to broker
            await self.executor.modify_order(order)
            
            # Update state
            await self._transition_state(order, OrderStatus.REPLACED)
            order.updated_at = datetime.utcnow()
            
            # Save changes
            await self._save_order(order)
            
            # Emit event
            await self._emit_order_event(order, "modified")
            
            logger.info(f"Order modified: {order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            await self._transition_state(order, OrderStatus.REJECTED)
            raise

    async def cancel_order(self, order_id: str, reason: str = "") -> Order:
        """Cancel an order"""
        order = self.active_orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        # Check if cancellation is allowed
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status {order.status}")
        
        try:
            # Update state
            await self._transition_state(order, OrderStatus.PENDING_CANCEL)
            
            # Submit cancellation to broker
            await self.executor.cancel_order(order)
            
            # Update order
            await self._transition_state(order, OrderStatus.CANCELLED)
            order.cancelled_at = datetime.utcnow()
            order.notes = f"{order.notes or ''} Cancelled: {reason}".strip()
            
            # Release position reservations
            await self.position_manager.release_reservation(order)
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Save changes
            await self._save_order(order)
            
            # Emit event
            await self._emit_order_event(order, "cancelled", reason)
            
            # Update metrics
            self._update_metrics(order.symbol, 'cancelled', 0)
            
            logger.info(f"Order cancelled: {order_id} - {reason}")
            return order
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            await self._transition_state(order, OrderStatus.REJECTED)
            raise

    async def handle_fill(self, fill: Fill):
        """Process a fill event"""
        order = self.active_orders.get(fill.order_id)
        if not order:
            logger.warning(f"Fill received for unknown order: {fill.order_id}")
            return
        
        try:
            # Update order quantities
            order.filled_quantity += fill.quantity
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.last_fill_price = fill.price
            order.last_fill_quantity = fill.quantity
            
            # Update average fill price
            if order.average_fill_price:
                total_value = (order.average_fill_price * (order.filled_quantity - fill.quantity)) + \
                             (fill.price * fill.quantity)
                order.average_fill_price = total_value / order.filled_quantity
            else:
                order.average_fill_price = fill.price
            
            # Update commission
            order.commission += fill.commission
            
            # Calculate slippage
            if order.order_type == OrderType.MARKET:
                # Compare to last known price before order
                expected_price = order.metadata.get('expected_price')
                if expected_price:
                    order.slippage = abs(fill.price - Decimal(str(expected_price)))
            elif order.limit_price:
                order.slippage = abs(fill.price - order.limit_price)
            
            # Update status
            if order.filled_quantity >= order.quantity:
                await self._transition_state(order, OrderStatus.FILLED)
                order.filled_at = datetime.utcnow()
                
                # Remove from active orders
                del self.active_orders[order.order_id]
                
                # Update metrics
                self._update_metrics(order.symbol, 'filled', 0)
            else:
                await self._transition_state(order, OrderStatus.PARTIALLY_FILLED)
            
            # Update positions
            await self.position_manager.update_position(fill)
            
            # Save changes
            await self._save_order(order)
            
            # Emit events
            await self._emit_order_event(order, "fill", fill.dict())
            
            logger.info(f"Fill processed: {fill.order_id} - {fill.quantity}@{fill.price}")
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
            raise

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check database
        return await self.repository.get_by_id(order_id)

    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders"""
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return sorted(orders, key=lambda x: x.created_at, reverse=True)

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get historical orders"""
        return await self.repository.get_recent_orders(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

    # Private methods
    
    async def _process_pending_orders(self):
        """Process pending order queue"""
        while self._running:
            try:
                # Get pending order with timeout
                order = await asyncio.wait_for(
                    self.pending_orders.get(),
                    timeout=1.0
                )
                
                # Submit to executor
                await self._submit_order(order)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing pending orders: {e}")
                await asyncio.sleep(1)

    async def _submit_order(self, order: Order):
        """Submit order for execution"""
        try:
            # Update state
            await self._transition_state(order, OrderStatus.NEW)
            order.submitted_at = datetime.utcnow()
            
            # Execute order
            result = await self.executor.submit_order(order)
            
            # Update with broker order ID
            order.metadata['broker_order_id'] = result.get('order_id')
            
            # Save order
            await self._save_order(order)
            
            # Emit event
            await self._emit_order_event(order, "submitted")
            
            # Update metrics
            latency = (datetime.utcnow() - order.created_at).total_seconds() * 1000
            self._update_metrics(order.symbol, 'submitted', latency)
            
            logger.info(f"Order submitted: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {e}")
            await self._transition_state(order, OrderStatus.REJECTED)
            order.notes = f"Submission failed: {str(e)}"
            await self._save_order(order)
            
            # Remove from active orders
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

    async def _monitor_active_orders(self):
        """Monitor active orders for updates"""
        while self._running:
            try:
                for order in list(self.active_orders.values()):
                    # Check for updates from broker
                    broker_status = await self.executor.get_order_status(order)
                    
                    if broker_status and broker_status != order.status:
                        # Handle status change
                        await self._handle_status_change(order, broker_status)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(5)

    async def _check_order_expiry(self):
        """Check for expired orders"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                
                for order in list(self.active_orders.values()):
                    # Check day orders at market close
                    if order.time_in_force == TimeInForce.DAY:
                        # Simplified check - should use market calendar
                        if current_time.hour >= 16:  # 4 PM UTC
                            await self._expire_order(order)
                    
                    # Check for custom expiry
                    elif order.metadata.get('expire_at'):
                        expire_at = datetime.fromisoformat(order.metadata['expire_at'])
                        if current_time >= expire_at:
                            await self._expire_order(order)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error checking order expiry: {e}")
                await asyncio.sleep(60)

    async def _expire_order(self, order: Order):
        """Expire an order"""
        try:
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                await self._transition_state(order, OrderStatus.EXPIRED)
                order.expired_at = datetime.utcnow()
                
                # Cancel at broker
                await self.executor.cancel_order(order)
                
                # Release reservations
                await self.position_manager.release_reservation(order)
                
                # Remove from active orders
                del self.active_orders[order.order_id]
                
                # Save changes
                await self._save_order(order)
                
                # Emit event
                await self._emit_order_event(order, "expired")
                
                logger.info(f"Order expired: {order.order_id}")
                
        except Exception as e:
            logger.error(f"Error expiring order {order.order_id}: {e}")

    async def _reconcile_orders(self):
        """Reconcile orders with broker"""
        while self._running:
            try:
                # Get all orders from broker
                broker_orders = await self.executor.get_all_orders()
                
                # Reconcile with local orders
                for broker_order in broker_orders:
                    local_order = self.active_orders.get(broker_order['client_order_id'])
                    
                    if local_order:
                        # Update local order with broker state
                        if broker_order['status'] != local_order.status.value:
                            await self._handle_status_change(
                                local_order,
                                OrderStatus(broker_order['status'])
                            )
                        
                        # Update fill information
                        if broker_order.get('filled_qty'):
                            local_order.filled_quantity = Decimal(str(broker_order['filled_qty']))
                            local_order.remaining_quantity = local_order.quantity - local_order.filled_quantity
                    else:
                        # Unknown order - log for investigation
                        logger.warning(f"Unknown order from broker: {broker_order}")
                
                await asyncio.sleep(30)  # Reconcile every 30 seconds
                
            except Exception as e:
                logger.error(f"Error reconciling orders: {e}")
                await asyncio.sleep(60)

    async def _transition_state(self, order: Order, new_status: OrderStatus):
        """Validate and perform state transition"""
        if new_status not in self.valid_transitions.get(order.status, []):
            raise ValueError(
                f"Invalid state transition: {order.status} -> {new_status}"
            )
        
        old_status = order.status
        order.status = new_status
        order.updated_at = datetime.utcnow()
        
        logger.debug(f"Order {order.order_id} state: {old_status} -> {new_status}")

    async def _handle_status_change(self, order: Order, new_status: OrderStatus):
        """Handle order status change from broker"""
        try:
            old_status = order.status
            await self._transition_state(order, new_status)
            
            # Handle specific status changes
            if new_status == OrderStatus.FILLED:
                order.filled_at = datetime.utcnow()
                del self.active_orders[order.order_id]
            elif new_status == OrderStatus.CANCELLED:
                order.cancelled_at = datetime.utcnow()
                del self.active_orders[order.order_id]
            elif new_status == OrderStatus.REJECTED:
                del self.active_orders[order.order_id]
            
            # Save changes
            await self._save_order(order)
            
            # Emit event
            await self._emit_order_event(
                order,
                "status_changed",
                {"old_status": old_status, "new_status": new_status}
            )
            
        except Exception as e:
            logger.error(f"Error handling status change for {order.order_id}: {e}")

    async def _save_order(self, order: Order):
        """Save order to database"""
        try:
            await self.repository.create(order)
        except Exception as e:
            logger.error(f"Error saving order {order.order_id}: {e}")

    async def _load_active_orders(self):
        """Load active orders from database on startup"""
        try:
            orders = await self.repository.get_active_orders()
            for order in orders:
                self.active_orders[order.order_id] = order
            
            logger.info(f"Loaded {len(orders)} active orders")
            
        except Exception as e:
            logger.error(f"Error loading active orders: {e}")

    async def _emit_order_event(self, order: Order, event_type: str, data: Any = None):
        """Emit order event"""
        # This would integrate with an event bus system
        logger.info(f"Order event: {event_type} for {order.order_id}")

    def _update_metrics(self, symbol: str, metric_type: str, latency: float):
        """Update performance metrics"""
        self.metrics[symbol]['total'] += 1
        
        if metric_type == 'submitted':
            self.metrics[symbol]['successful'] += 1
        elif metric_type == 'rejected':
            self.metrics[symbol]['rejected'] += 1
        elif metric_type == 'cancelled':
            self.metrics[symbol]['cancelled'] += 1
        
        # Update average latency
        if latency > 0:
            current_avg = self.metrics[symbol]['avg_latency_ms']
            count = self.metrics[symbol]['total']
            self.metrics[symbol]['avg_latency_ms'] = (
                (current_avg * (count - 1) + latency) / count
            )

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'active_orders': len(self.active_orders),
            'pending_orders': self.pending_orders.qsize(),
            'symbol_metrics': dict(self.metrics),
            'total_orders_today': sum(m['total'] for m in self.metrics.values()),
            'success_rate': sum(m['successful'] for m in self.metrics.values()) / 
                           max(sum(m['total'] for m in self.metrics.values()), 1)
        }

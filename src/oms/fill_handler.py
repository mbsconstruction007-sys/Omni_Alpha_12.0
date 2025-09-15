"""
Fill Handler - Processes trade fills and updates
Handles fill events from brokers and updates order status
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
import logging

from .models import Fill, Order, OrderStatus, OrderSide
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

class FillHandler:
    """Handles trade fill processing"""
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        self.fill_history: List[Fill] = []
        self.fill_metrics = {
            'total_fills': 0,
            'total_volume': Decimal('0'),
            'total_commission': Decimal('0'),
            'avg_fill_size': Decimal('0'),
            'avg_fill_price': Decimal('0')
        }

    async def process_fill(self, fill_data: Dict[str, Any], order: Order) -> Fill:
        """Process a fill event from broker"""
        try:
            # Create fill object
            fill = Fill(
                order_id=order.order_id,
                symbol=fill_data.get('symbol', order.symbol),
                side=OrderSide(fill_data.get('side', order.side.value)),
                quantity=Decimal(str(fill_data.get('quantity', 0))),
                price=Decimal(str(fill_data.get('price', 0))),
                venue=fill_data.get('venue', 'unknown'),
                commission=Decimal(str(fill_data.get('commission', 0))),
                liquidity=fill_data.get('liquidity', 'taker'),
                metadata=fill_data.get('metadata', {})
            )
            
            # Validate fill
            if not self._validate_fill(fill, order):
                raise ValueError(f"Invalid fill data: {fill_data}")
            
            # Update order with fill information
            await self._update_order_with_fill(order, fill)
            
            # Update position
            await self.position_manager.update_position(fill)
            
            # Record fill
            self.fill_history.append(fill)
            self._update_fill_metrics(fill)
            
            logger.info(f"Fill processed: {fill.order_id} - {fill.quantity}@{fill.price}")
            
            return fill
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
            raise

    async def process_partial_fill(self, fill_data: Dict[str, Any], order: Order) -> Fill:
        """Process a partial fill"""
        fill = await self.process_fill(fill_data, order)
        
        # Update order status to partially filled
        if order.filled_quantity < order.quantity:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        return fill

    async def process_complete_fill(self, fill_data: Dict[str, Any], order: Order) -> Fill:
        """Process a complete fill"""
        fill = await self.process_fill(fill_data, order)
        
        # Update order status to filled
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        
        return fill

    def _validate_fill(self, fill: Fill, order: Order) -> bool:
        """Validate fill data against order"""
        # Check symbol matches
        if fill.symbol != order.symbol:
            logger.error(f"Fill symbol {fill.symbol} doesn't match order symbol {order.symbol}")
            return False
        
        # Check side matches
        if fill.side != order.side:
            logger.error(f"Fill side {fill.side} doesn't match order side {order.side}")
            return False
        
        # Check quantity is positive
        if fill.quantity <= 0:
            logger.error(f"Invalid fill quantity: {fill.quantity}")
            return False
        
        # Check price is positive
        if fill.price <= 0:
            logger.error(f"Invalid fill price: {fill.price}")
            return False
        
        # Check fill doesn't exceed remaining quantity
        remaining = order.quantity - order.filled_quantity
        if fill.quantity > remaining:
            logger.error(f"Fill quantity {fill.quantity} exceeds remaining {remaining}")
            return False
        
        return True

    async def _update_order_with_fill(self, order: Order, fill: Fill):
        """Update order with fill information"""
        # Update filled quantities
        order.filled_quantity += fill.quantity
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        # Update fill prices
        order.last_fill_price = fill.price
        order.last_fill_quantity = fill.quantity
        
        # Update average fill price
        if order.average_fill_price:
            # Calculate weighted average
            total_value = (order.average_fill_price * (order.filled_quantity - fill.quantity)) + \
                         (fill.price * fill.quantity)
            order.average_fill_price = total_value / order.filled_quantity
        else:
            order.average_fill_price = fill.price
        
        # Update commission
        order.commission += fill.commission
        
        # Calculate slippage
        if order.order_type.value == 'market':
            # For market orders, compare to expected price
            expected_price = order.metadata.get('expected_price')
            if expected_price:
                order.slippage = abs(fill.price - Decimal(str(expected_price)))
        elif order.limit_price:
            # For limit orders, slippage is difference from limit price
            if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                # For buy orders, positive slippage is bad (paid more than limit)
                order.slippage = max(Decimal('0'), fill.price - order.limit_price)
            else:
                # For sell orders, positive slippage is bad (received less than limit)
                order.slippage = max(Decimal('0'), order.limit_price - fill.price)
        
        # Update timestamp
        order.updated_at = datetime.utcnow()

    def _update_fill_metrics(self, fill: Fill):
        """Update fill performance metrics"""
        self.fill_metrics['total_fills'] += 1
        self.fill_metrics['total_volume'] += fill.quantity
        self.fill_metrics['total_commission'] += fill.commission
        
        # Update averages
        total_fills = self.fill_metrics['total_fills']
        
        # Average fill size
        self.fill_metrics['avg_fill_size'] = self.fill_metrics['total_volume'] / total_fills
        
        # Average fill price (volume-weighted)
        if total_fills == 1:
            self.fill_metrics['avg_fill_price'] = fill.price
        else:
            # Volume-weighted average
            current_avg = self.fill_metrics['avg_fill_price']
            total_volume = self.fill_metrics['total_volume']
            new_volume = fill.quantity
            
            self.fill_metrics['avg_fill_price'] = (
                (current_avg * (total_volume - new_volume) + fill.price * new_volume) / total_volume
            )

    async def get_fill_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Fill]:
        """Get fill history with optional filters"""
        fills = self.fill_history
        
        # Apply filters
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        if start_date:
            fills = [f for f in fills if f.timestamp >= start_date]
        
        if end_date:
            fills = [f for f in fills if f.timestamp <= end_date]
        
        # Sort by timestamp (newest first)
        fills.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        return fills[:limit]

    async def get_fill_metrics(self) -> Dict[str, Any]:
        """Get fill performance metrics"""
        return {
            **self.fill_metrics,
            'avg_commission_per_fill': (
                self.fill_metrics['total_commission'] / max(self.fill_metrics['total_fills'], 1)
            ),
            'total_fill_value': self.fill_metrics['total_volume'] * self.fill_metrics['avg_fill_price']
        }

    async def get_fills_by_order(self, order_id: str) -> List[Fill]:
        """Get all fills for a specific order"""
        return [f for f in self.fill_history if f.order_id == order_id]

    async def get_fills_by_symbol(self, symbol: str) -> List[Fill]:
        """Get all fills for a specific symbol"""
        return [f for f in self.fill_history if f.symbol == symbol]

    def clear_history(self):
        """Clear fill history (for testing)"""
        self.fill_history.clear()
        self.fill_metrics = {
            'total_fills': 0,
            'total_volume': Decimal('0'),
            'total_commission': Decimal('0'),
            'avg_fill_size': Decimal('0'),
            'avg_fill_price': Decimal('0')
        }

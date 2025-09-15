"""
Position Manager - Tracks and manages trading positions
Handles position updates, P&L calculations, and risk monitoring
"""

from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
import logging

from .models import Order, Fill, Position, OrderSide
try:
    from src.brokers import BrokerManager
except ImportError:
    # For testing purposes, create mock class
    class BrokerManager:
        pass

logger = logging.getLogger(__name__)

class PositionManager:
    """Position tracking and management"""
    
    def __init__(self, broker_manager: BrokerManager):
        self.broker_manager = broker_manager
        self.positions: Dict[str, Position] = {}
        self.reservations: Dict[str, Decimal] = {}  # Reserved quantities for pending orders

    async def reserve_for_order(self, order: Order):
        """Reserve quantity for a pending order"""
        symbol = order.symbol
        quantity = order.quantity
        
        if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
            # For buy orders, reserve cash (handled by risk checker)
            pass
        else:
            # For sell orders, reserve shares
            current_reservation = self.reservations.get(symbol, Decimal('0'))
            self.reservations[symbol] = current_reservation + quantity
            
            # Check if we have enough shares
            position = self.positions.get(symbol)
            if position and position.quantity < self.reservations[symbol]:
                raise ValueError(f"Insufficient shares to sell {quantity} of {symbol}")
        
        logger.debug(f"Reserved {quantity} {symbol} for order {order.order_id}")

    async def release_reservation(self, order: Order):
        """Release reservation for cancelled order"""
        symbol = order.symbol
        quantity = order.quantity
        
        if order.side in [OrderSide.SELL, OrderSide.SELL_SHORT]:
            current_reservation = self.reservations.get(symbol, Decimal('0'))
            self.reservations[symbol] = max(Decimal('0'), current_reservation - quantity)
            
            # Remove if no reservation left
            if self.reservations[symbol] == 0:
                del self.reservations[symbol]
        
        logger.debug(f"Released reservation for {quantity} {symbol}")

    async def update_position(self, fill: Fill):
        """Update position based on fill"""
        symbol = fill.symbol
        quantity = fill.quantity
        price = fill.price
        
        # Get current position
        position = self.positions.get(symbol)
        
        if not position:
            # Create new position
            position = Position(
                symbol=symbol,
                quantity=Decimal('0'),
                side='long',
                average_entry_price=Decimal('0'),
                cost_basis=Decimal('0')
            )
            self.positions[symbol] = position
        
        # Update position based on fill side
        if fill.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
            # Buying - increase position
            if position.quantity >= 0:
                # Adding to long position
                total_cost = (position.quantity * position.average_entry_price) + (quantity * price)
                total_quantity = position.quantity + quantity
                position.average_entry_price = total_cost / total_quantity if total_quantity > 0 else Decimal('0')
                position.quantity = total_quantity
                position.side = 'long'
            else:
                # Covering short position
                if abs(position.quantity) >= quantity:
                    # Partial cover
                    position.quantity += quantity
                    if position.quantity == 0:
                        # Position closed
                        position.realized_pnl += (position.average_entry_price - price) * quantity
                        del self.positions[symbol]
                        return
                else:
                    # Over-cover - flip to long
                    cover_quantity = abs(position.quantity)
                    remaining_quantity = quantity - cover_quantity
                    position.realized_pnl += (position.average_entry_price - price) * cover_quantity
                    position.quantity = remaining_quantity
                    position.average_entry_price = price
                    position.side = 'long'
        else:
            # Selling - decrease position
            if position.quantity <= 0:
                # Adding to short position
                total_cost = (abs(position.quantity) * position.average_entry_price) + (quantity * price)
                total_quantity = abs(position.quantity) + quantity
                position.average_entry_price = total_cost / total_quantity if total_quantity > 0 else Decimal('0')
                position.quantity = -total_quantity
                position.side = 'short'
            else:
                # Reducing long position
                if position.quantity >= quantity:
                    # Partial sell
                    position.quantity -= quantity
                    position.realized_pnl += (price - position.average_entry_price) * quantity
                    if position.quantity == 0:
                        # Position closed
                        del self.positions[symbol]
                        return
                else:
                    # Over-sell - flip to short
                    sell_quantity = position.quantity
                    remaining_quantity = quantity - sell_quantity
                    position.realized_pnl += (price - position.average_entry_price) * sell_quantity
                    position.quantity = -remaining_quantity
                    position.average_entry_price = price
                    position.side = 'short'
        
        # Update cost basis
        position.cost_basis = abs(position.quantity) * position.average_entry_price
        position.updated_at = datetime.utcnow()
        
        # Update current price and P&L
        await self._update_market_values(position)
        
        logger.info(f"Position updated: {symbol} {position.quantity} @ {position.average_entry_price}")

    async def _update_market_values(self, position: Position):
        """Update current market values and P&L"""
        try:
            # Get current price from broker
            broker = await self.broker_manager.get_primary_broker()
            if broker:
                quote = await broker.get_quote(position.symbol)
                position.current_price = Decimal(str(quote.get('last', 0)))
            else:
                position.current_price = position.average_entry_price  # Fallback
            
            # Calculate market value
            position.market_value = abs(position.quantity) * position.current_price
            
            # Calculate unrealized P&L
            if position.quantity > 0:  # Long position
                position.unrealized_pnl = (position.current_price - position.average_entry_price) * position.quantity
            else:  # Short position
                position.unrealized_pnl = (position.average_entry_price - position.current_price) * abs(position.quantity)
            
        except Exception as e:
            logger.error(f"Error updating market values for {position.symbol}: {e}")

    async def get_positions(self) -> List[Position]:
        """Get all current positions"""
        # Update market values for all positions
        for position in self.positions.values():
            await self._update_market_values(position)
        
        return list(self.positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        position = self.positions.get(symbol)
        if position:
            await self._update_market_values(position)
        return position

    async def get_total_pnl(self) -> Decimal:
        """Get total P&L across all positions"""
        total_pnl = Decimal('0')
        
        for position in self.positions.values():
            await self._update_market_values(position)
            total_pnl += position.realized_pnl + (position.unrealized_pnl or Decimal('0'))
        
        return total_pnl

    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value"""
        total_value = Decimal('0')
        
        for position in self.positions.values():
            await self._update_market_values(position)
            total_value += position.market_value or Decimal('0')
        
        return total_value

    def get_reserved_quantity(self, symbol: str) -> Decimal:
        """Get reserved quantity for symbol"""
        return self.reservations.get(symbol, Decimal('0'))

    def get_available_quantity(self, symbol: str) -> Decimal:
        """Get available quantity for trading"""
        position = self.positions.get(symbol)
        if not position:
            return Decimal('0')
        
        reserved = self.reservations.get(symbol, Decimal('0'))
        return max(Decimal('0'), position.quantity - reserved)

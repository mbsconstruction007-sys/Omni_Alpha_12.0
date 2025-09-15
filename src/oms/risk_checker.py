"""
Risk Checker - Pre-trade risk management
Validates orders against risk limits and compliance rules
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import logging

from .models import Order, OrderSide, OrderType
try:
    from src.brokers import BrokerManager
except ImportError:
    # For testing purposes, create mock class
    class BrokerManager:
        pass

logger = logging.getLogger(__name__)

@dataclass
class RiskCheckResult:
    """Risk check result"""
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = None
    checks_performed: List[str] = None

class RiskChecker:
    """Pre-trade risk validation"""
    
    def __init__(
        self,
        broker_manager: BrokerManager,
        config: Dict[str, Any]
    ):
        self.broker_manager = broker_manager
        self.config = config
        
        # Risk limits
        self.max_order_value = Decimal(str(config.get('max_order_value', 100000)))
        self.max_position_value = Decimal(str(config.get('max_position_value', 1000000)))
        self.max_daily_trades = config.get('max_daily_trades', 500)
        self.max_daily_volume = Decimal(str(config.get('max_daily_volume', 5000000)))
        self.max_concentration = Decimal(str(config.get('max_concentration', 0.20)))
        self.min_liquidity_ratio = Decimal(str(config.get('min_liquidity_ratio', 0.01)))
        
        # Position limits
        self.max_open_positions = config.get('max_open_positions', 50)
        self.max_position_size = Decimal(str(config.get('max_position_size', 100000)))
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_volume = Decimal('0')
        self.last_reset = datetime.utcnow().date()
        
        # Position tracking
        self.positions: Dict[str, Decimal] = {}
        self.total_exposure = Decimal('0')
        
        # Restricted lists
        self.restricted_symbols = set(config.get('restricted_symbols', []))
        self.watch_list_symbols = set(config.get('watch_list_symbols', []))

    async def check_order(self, order: Order) -> RiskCheckResult:
        """Perform all risk checks on an order"""
        checks_performed = []
        details = {}
        
        try:
            # Reset daily counters if needed
            self._reset_daily_counters_if_needed()
            
            # 1. Symbol restrictions
            if not await self._check_symbol_restrictions(order):
                return RiskCheckResult(
                    passed=False,
                    reason=f"Symbol {order.symbol} is restricted",
                    details={'restricted_symbol': order.symbol},
                    checks_performed=['symbol_restrictions']
                )
            checks_performed.append('symbol_restrictions')
            
            # 2. Order value limits
            order_value = await self._calculate_order_value(order)
            details['order_value'] = float(order_value)
            
            if order_value > self.max_order_value:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Order value ${order_value} exceeds limit ${self.max_order_value}",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('order_value_limit')
            
            # 3. Daily trade limits
            if self.daily_trades >= self.max_daily_trades:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Daily trade limit reached ({self.max_daily_trades})",
                    details={'daily_trades': self.daily_trades},
                    checks_performed=checks_performed
                )
            checks_performed.append('daily_trade_limit')
            
            # 4. Daily volume limits
            if self.daily_volume + order_value > self.max_daily_volume:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Would exceed daily volume limit ${self.max_daily_volume}",
                    details={'current_volume': float(self.daily_volume)},
                    checks_performed=checks_performed
                )
            checks_performed.append('daily_volume_limit')
            
            # 5. Position concentration
            concentration = await self._check_concentration(order, order_value)
            details['concentration'] = float(concentration)
            
            if concentration > self.max_concentration:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Position concentration {concentration:.1%} exceeds limit {self.max_concentration:.1%}",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('concentration_limit')
            
            # 6. Liquidity check
            liquidity_ratio = await self._check_liquidity(order)
            details['liquidity_ratio'] = float(liquidity_ratio)
            
            if liquidity_ratio < self.min_liquidity_ratio:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Insufficient liquidity (ratio: {liquidity_ratio:.3f})",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('liquidity_check')
            
            # 7. Position limits
            if len(self.positions) >= self.max_open_positions:
                if order.symbol not in self.positions:
                    return RiskCheckResult(
                        passed=False,
                        reason=f"Maximum open positions reached ({self.max_open_positions})",
                        details={'open_positions': len(self.positions)},
                        checks_performed=checks_performed
                    )
            checks_performed.append('position_limit')
            
            # 8. Margin requirements (if applicable)
            margin_required = await self._calculate_margin_requirement(order)
            details['margin_required'] = float(margin_required)
            
            available_margin = await self._get_available_margin()
            details['available_margin'] = float(available_margin)
            
            if margin_required > available_margin:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Insufficient margin (required: ${margin_required}, available: ${available_margin})",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('margin_check')
            
            # 9. Price reasonability
            if not await self._check_price_reasonability(order):
                return RiskCheckResult(
                    passed=False,
                    reason="Order price outside reasonable range",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('price_reasonability')
            
            # 10. Fat finger check
            if not await self._check_fat_finger(order):
                return RiskCheckResult(
                    passed=False,
                    reason="Potential fat finger error detected",
                    details=details,
                    checks_performed=checks_performed
                )
            checks_performed.append('fat_finger_check')
            
            # All checks passed
            return RiskCheckResult(
                passed=True,
                reason="All risk checks passed",
                details=details,
                checks_performed=checks_performed
            )
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return RiskCheckResult(
                passed=False,
                reason=f"Risk check error: {str(e)}",
                details=details,
                checks_performed=checks_performed
            )

    async def update_position(self, symbol: str, quantity: Decimal, side: OrderSide):
        """Update position tracking"""
        current_position = self.positions.get(symbol, Decimal('0'))
        
        if side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
            self.positions[symbol] = current_position + quantity
        else:
            self.positions[symbol] = current_position - quantity
        
        # Remove if position is closed
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        # Recalculate total exposure
        await self._recalculate_exposure()

    # Private methods
    
    async def _check_symbol_restrictions(self, order: Order) -> bool:
        """Check if symbol is restricted"""
        if order.symbol in self.restricted_symbols:
            logger.warning(f"Order for restricted symbol: {order.symbol}")
            return False
        
        if order.symbol in self.watch_list_symbols:
            logger.info(f"Order for watch list symbol: {order.symbol}")
            # Additional checks could be performed here
        
        return True

    async def _calculate_order_value(self, order: Order) -> Decimal:
        """Calculate total order value"""
        if order.limit_price:
            price = order.limit_price
        else:
            # Get current market price from broker
            try:
                broker = await self.broker_manager.get_primary_broker()
                if broker:
                    quote = await broker.get_quote(order.symbol)
                    if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                        price = Decimal(str(quote.get('ask', 0)))
                    else:
                        price = Decimal(str(quote.get('bid', 0)))
                else:
                    # Fallback to a default price
                    price = Decimal('100.00')
            except Exception as e:
                logger.error(f"Error getting quote for {order.symbol}: {e}")
                price = Decimal('100.00')  # Fallback
        
        return price * order.quantity

    async def _check_concentration(self, order: Order, order_value: Decimal) -> Decimal:
        """Check position concentration"""
        # Calculate new position value
        current_position = self.positions.get(order.symbol, Decimal('0'))
        
        if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
            new_position_value = (current_position + order.quantity) * order_value / order.quantity
        else:
            new_position_value = abs(current_position - order.quantity) * order_value / order.quantity
        
        # Calculate concentration
        total_portfolio_value = self.total_exposure + order_value
        
        if total_portfolio_value == 0:
            return Decimal('0')
        
        return new_position_value / total_portfolio_value

    async def _check_liquidity(self, order: Order) -> Decimal:
        """Check market liquidity"""
        try:
            # Get average daily volume from broker
            broker = await self.broker_manager.get_primary_broker()
            if broker:
                stats = await broker.get_stats(order.symbol)
                avg_volume = Decimal(str(stats.get('avg_volume', 0)))
            else:
                avg_volume = Decimal('1000000')  # Default
            
            if avg_volume == 0:
                return Decimal('0')
            
            # Calculate liquidity ratio (order size vs average volume)
            return order.quantity / avg_volume
            
        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            return Decimal('1')  # Assume sufficient liquidity on error

    async def _calculate_margin_requirement(self, order: Order) -> Decimal:
        """Calculate margin requirement for order"""
        order_value = await self._calculate_order_value(order)
        
        # Simplified margin calculation
        # In production, this would be based on asset class, volatility, etc.
        if order.asset_class == 'equity':
            margin_rate = Decimal('0.5')  # 50% margin for equities
        elif order.asset_class == 'options':
            margin_rate = Decimal('1.0')  # 100% for options
        else:
            margin_rate = Decimal('0.25')  # 25% for others
        
        return order_value * margin_rate

    async def _get_available_margin(self) -> Decimal:
        """Get available margin from broker"""
        try:
            # Get account info from broker
            broker = await self.broker_manager.get_primary_broker()
            if broker:
                account = await broker.get_account()
                return Decimal(str(account.get('buying_power', 100000)))
            else:
                return Decimal('100000')  # Default
        except Exception as e:
            logger.error(f"Error getting available margin: {e}")
            return Decimal('0')

    async def _check_price_reasonability(self, order: Order) -> bool:
        """Check if order price is reasonable"""
        if order.order_type == OrderType.MARKET:
            return True  # Market orders always pass
        
        try:
            # Get current market price from broker
            broker = await self.broker_manager.get_primary_broker()
            if broker:
                quote = await broker.get_quote(order.symbol)
                mid_price = Decimal(str((quote['bid'] + quote['ask']) / 2))
                
                # Check if limit price is within 10% of market
                if order.limit_price:
                    deviation = abs(order.limit_price - mid_price) / mid_price
                    if deviation > Decimal('0.1'):  # 10% threshold
                        logger.warning(f"Order price deviates {deviation:.1%} from market")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking price reasonability: {e}")
            return True  # Pass on error

    async def _check_fat_finger(self, order: Order) -> bool:
        """Check for potential fat finger errors"""
        # Check for unusually large quantities
        if order.quantity > self.max_position_size:
            logger.warning(f"Potential fat finger: quantity {order.quantity} exceeds max")
            return False
        
        # Check for decimal place errors in price
        if order.limit_price:
            price_str = str(order.limit_price)
            if '.' in price_str:
                decimal_places = len(price_str.split('.')[1])
                if decimal_places > 4:  # More than 4 decimal places is suspicious
                    logger.warning(f"Potential fat finger: unusual price precision {order.limit_price}")
                    return False
        
        return True

    def _reset_daily_counters_if_needed(self):
        """Reset daily counters at start of new day"""
        current_date = datetime.utcnow().date()
        
        if current_date > self.last_reset:
            self.daily_trades = 0
            self.daily_volume = Decimal('0')
            self.last_reset = current_date
            logger.info("Daily risk counters reset")

    async def _recalculate_exposure(self):
        """Recalculate total portfolio exposure"""
        total = Decimal('0')
        
        for symbol, quantity in self.positions.items():
            try:
                broker = await self.broker_manager.get_primary_broker()
                if broker:
                    quote = await broker.get_quote(symbol)
                    price = Decimal(str(quote.get('last', 0)))
                else:
                    price = Decimal('100.00')  # Default
                total += abs(quantity) * price
            except Exception as e:
                logger.error(f"Error calculating exposure for {symbol}: {e}")
        
        self.total_exposure = total

    def increment_daily_counters(self, order_value: Decimal):
        """Increment daily risk counters"""
        self.daily_trades += 1
        self.daily_volume += order_value

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            'daily_trades': self.daily_trades,
            'daily_volume': float(self.daily_volume),
            'open_positions': len(self.positions),
            'total_exposure': float(self.total_exposure),
            'limits': {
                'max_order_value': float(self.max_order_value),
                'max_daily_trades': self.max_daily_trades,
                'max_daily_volume': float(self.max_daily_volume),
                'max_concentration': float(self.max_concentration),
                'max_open_positions': self.max_open_positions
            }
        }

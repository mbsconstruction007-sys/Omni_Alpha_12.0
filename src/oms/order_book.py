"""
Order Book - Manages order book state and market data
Tracks order book depth and provides market data services
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import logging
from collections import defaultdict, deque

from .models import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)

class OrderBookLevel:
    """Represents a price level in the order book"""
    
    def __init__(self, price: Decimal, quantity: Decimal):
        self.price = price
        self.quantity = quantity
        self.order_count = 1
        self.last_updated = datetime.utcnow()

class OrderBook:
    """Order book management system"""
    
    def __init__(self, symbol: str, max_levels: int = 10):
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Order book levels
        self.bids: Dict[Decimal, OrderBookLevel] = {}  # price -> level
        self.asks: Dict[Decimal, OrderBookLevel] = {}  # price -> level
        
        # Best bid/ask
        self.best_bid: Optional[Decimal] = None
        self.best_ask: Optional[Decimal] = None
        
        # Market data
        self.last_trade_price: Optional[Decimal] = None
        self.last_trade_quantity: Optional[Decimal] = None
        self.last_trade_time: Optional[datetime] = None
        
        # Volume tracking
        self.volume_today: Decimal = Decimal('0')
        self.volume_24h: Decimal = Decimal('0')
        
        # Price history
        self.price_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'total_volume': Decimal('0'),
            'avg_order_size': Decimal('0'),
            'last_updated': datetime.utcnow()
        }

    def add_order(self, order: Order):
        """Add order to order book"""
        try:
            price = order.limit_price or order.stop_price
            if not price:
                return  # Market orders don't go in order book
            
            if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                self._add_bid(price, order.quantity)
            else:
                self._add_ask(price, order.quantity)
            
            self._update_stats(order)
            self._update_best_prices()
            
            logger.debug(f"Added {order.side} order to book: {order.quantity}@{price}")
            
        except Exception as e:
            logger.error(f"Error adding order to book: {e}")

    def remove_order(self, order: Order):
        """Remove order from order book"""
        try:
            price = order.limit_price or order.stop_price
            if not price:
                return
            
            if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                self._remove_bid(price, order.quantity)
            else:
                self._remove_ask(price, order.quantity)
            
            self._update_best_prices()
            
            logger.debug(f"Removed {order.side} order from book: {order.quantity}@{price}")
            
        except Exception as e:
            logger.error(f"Error removing order from book: {e}")

    def update_order(self, old_order: Order, new_order: Order):
        """Update order in order book"""
        self.remove_order(old_order)
        self.add_order(new_order)

    def add_trade(self, price: Decimal, quantity: Decimal):
        """Add trade to order book"""
        self.last_trade_price = price
        self.last_trade_quantity = quantity
        self.last_trade_time = datetime.utcnow()
        
        # Update volume
        self.volume_today += quantity
        self.volume_24h += quantity
        
        # Add to price history
        self.price_history.append({
            'price': price,
            'quantity': quantity,
            'timestamp': self.last_trade_time
        })
        
        logger.debug(f"Trade added: {quantity}@{price}")

    def get_best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.best_bid

    def get_best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.best_ask

    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def get_bid_levels(self, levels: int = None) -> List[Tuple[Decimal, Decimal]]:
        """Get bid levels (price, quantity)"""
        if levels is None:
            levels = self.max_levels
        
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        return [(price, self.bids[price].quantity) for price in sorted_bids[:levels]]

    def get_ask_levels(self, levels: int = None) -> List[Tuple[Decimal, Decimal]]:
        """Get ask levels (price, quantity)"""
        if levels is None:
            levels = self.max_levels
        
        sorted_asks = sorted(self.asks.keys())
        return [(price, self.asks[price].quantity) for price in sorted_asks[:levels]]

    def get_market_depth(self, levels: int = None) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Get market depth"""
        return {
            'bids': self.get_bid_levels(levels),
            'asks': self.get_ask_levels(levels)
        }

    def get_volume_at_price(self, price: Decimal, side: OrderSide) -> Decimal:
        """Get total volume at specific price level"""
        if side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
            level = self.bids.get(price)
        else:
            level = self.asks.get(price)
        
        return level.quantity if level else Decimal('0')

    def get_total_bid_volume(self) -> Decimal:
        """Get total bid volume"""
        return sum(level.quantity for level in self.bids.values())

    def get_total_ask_volume(self) -> Decimal:
        """Get total ask volume"""
        return sum(level.quantity for level in self.asks.values())

    def get_imbalance_ratio(self) -> float:
        """Get order book imbalance ratio"""
        bid_volume = self.get_total_bid_volume()
        ask_volume = self.get_total_ask_volume()
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.5  # Balanced
        
        return float(bid_volume / total_volume)

    def get_vwap(self, lookback_periods: int = 20) -> Optional[Decimal]:
        """Get Volume Weighted Average Price"""
        if len(self.price_history) < 2:
            return None
        
        recent_trades = list(self.price_history)[-lookback_periods:]
        
        total_value = Decimal('0')
        total_volume = Decimal('0')
        
        for trade in recent_trades:
            total_value += trade['price'] * trade['quantity']
            total_volume += trade['quantity']
        
        return total_value / total_volume if total_volume > 0 else None

    def get_price_change(self, periods: int = 1) -> Optional[Decimal]:
        """Get price change over specified periods"""
        if len(self.price_history) < periods + 1:
            return None
        
        current_price = self.price_history[-1]['price']
        past_price = self.price_history[-(periods + 1)]['price']
        
        return current_price - past_price

    def get_price_change_percent(self, periods: int = 1) -> Optional[float]:
        """Get price change percentage"""
        price_change = self.get_price_change(periods)
        if price_change is None:
            return None
        
        past_price = self.price_history[-(periods + 1)]['price']
        return float(price_change / past_price * 100)

    def _add_bid(self, price: Decimal, quantity: Decimal):
        """Add bid to order book"""
        if price in self.bids:
            self.bids[price].quantity += quantity
            self.bids[price].order_count += 1
        else:
            self.bids[price] = OrderBookLevel(price, quantity)

    def _add_ask(self, price: Decimal, quantity: Decimal):
        """Add ask to order book"""
        if price in self.asks:
            self.asks[price].quantity += quantity
            self.asks[price].order_count += 1
        else:
            self.asks[price] = OrderBookLevel(price, quantity)

    def _remove_bid(self, price: Decimal, quantity: Decimal):
        """Remove bid from order book"""
        if price in self.bids:
            level = self.bids[price]
            level.quantity -= quantity
            level.order_count -= 1
            
            if level.quantity <= 0 or level.order_count <= 0:
                del self.bids[price]

    def _remove_ask(self, price: Decimal, quantity: Decimal):
        """Remove ask from order book"""
        if price in self.asks:
            level = self.asks[price]
            level.quantity -= quantity
            level.order_count -= 1
            
            if level.quantity <= 0 or level.order_count <= 0:
                del self.asks[price]

    def _update_best_prices(self):
        """Update best bid and ask prices"""
        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None

    def _update_stats(self, order: Order):
        """Update order book statistics"""
        self.stats['total_orders'] += 1
        self.stats['total_volume'] += order.quantity
        self.stats['avg_order_size'] = self.stats['total_volume'] / self.stats['total_orders']
        self.stats['last_updated'] = datetime.utcnow()

    def get_stats(self) -> Dict[str, any]:
        """Get order book statistics"""
        return {
            **self.stats,
            'best_bid': float(self.best_bid) if self.best_bid else None,
            'best_ask': float(self.best_ask) if self.best_ask else None,
            'spread': float(self.get_spread()) if self.get_spread() else None,
            'mid_price': float(self.get_mid_price()) if self.get_mid_price() else None,
            'imbalance_ratio': self.get_imbalance_ratio(),
            'total_bid_volume': float(self.get_total_bid_volume()),
            'total_ask_volume': float(self.get_total_ask_volume()),
            'volume_today': float(self.volume_today),
            'last_trade_price': float(self.last_trade_price) if self.last_trade_price else None
        }

    def clear(self):
        """Clear order book"""
        self.bids.clear()
        self.asks.clear()
        self.best_bid = None
        self.best_ask = None
        self.price_history.clear()
        self.volume_today = Decimal('0')
        self.stats = {
            'total_orders': 0,
            'total_volume': Decimal('0'),
            'avg_order_size': Decimal('0'),
            'last_updated': datetime.utcnow()
        }

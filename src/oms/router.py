"""
Smart Order Router - Intelligent order routing
Selects optimal execution venue based on various factors
"""

from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
import logging

from .models import Order, ExecutionVenue
try:
    from src.brokers import BrokerManager, BrokerType
except ImportError:
    # For testing purposes, create mock classes
    class BrokerManager:
        pass
    class BrokerType:
        ALPACA = "alpaca"
        UPSTOX = "upstox"

logger = logging.getLogger(__name__)

class SmartOrderRouter:
    """Smart order routing system"""
    
    def __init__(self, broker_manager: BrokerManager):
        self.broker_manager = broker_manager
        
        # Routing preferences
        self.routing_strategies = {
            'best_execution': self._route_best_execution,
            'lowest_cost': self._route_lowest_cost,
            'fastest_execution': self._route_fastest_execution,
            'primary_only': self._route_primary_only,
            'round_robin': self._route_round_robin
        }
        
        # Venue preferences
        self.venue_preferences = {
            ExecutionVenue.ALPACA: {
                'priority': 1,
                'cost_per_share': Decimal('0.005'),
                'avg_latency_ms': 50,
                'reliability': 0.99
            },
            ExecutionVenue.UPSTOX: {
                'priority': 2,
                'cost_per_share': Decimal('0.003'),
                'avg_latency_ms': 100,
                'reliability': 0.95
            }
        }
        
        # Performance tracking
        self.venue_performance = {
            venue: {
                'total_orders': 0,
                'successful_orders': 0,
                'avg_latency': 0,
                'total_cost': Decimal('0')
            }
            for venue in self.venue_preferences.keys()
        }

    async def select_venue(self, order: Order, strategy: str = 'best_execution') -> ExecutionVenue:
        """Select optimal venue for order execution"""
        try:
            # Get available venues
            available_venues = await self._get_available_venues()
            
            if not available_venues:
                raise ValueError("No venues available for order execution")
            
            # Apply routing strategy
            if strategy in self.routing_strategies:
                venue = await self.routing_strategies[strategy](order, available_venues)
            else:
                # Default to best execution
                venue = await self._route_best_execution(order, available_venues)
            
            # Update performance tracking
            self._update_venue_selection(venue)
            
            logger.info(f"Selected venue {venue} for order {order.order_id} using strategy {strategy}")
            return venue
            
        except Exception as e:
            logger.error(f"Error selecting venue: {e}")
            # Fallback to primary venue
            return ExecutionVenue.ALPACA

    async def _get_available_venues(self) -> list[ExecutionVenue]:
        """Get list of available venues"""
        available = []
        
        try:
            # Check Alpaca
            alpaca_broker = await self.broker_manager.get_broker(BrokerType.ALPACA)
            if alpaca_broker and alpaca_broker.status.value == 'connected':
                available.append(ExecutionVenue.ALPACA)
            
            # Check Upstox
            upstox_broker = await self.broker_manager.get_broker(BrokerType.UPSTOX)
            if upstox_broker and upstox_broker.status.value == 'connected':
                available.append(ExecutionVenue.UPSTOX)
            
        except Exception as e:
            logger.error(f"Error checking venue availability: {e}")
        
        return available

    async def _route_best_execution(self, order: Order, available_venues: list[ExecutionVenue]) -> ExecutionVenue:
        """Route to venue with best execution quality"""
        best_venue = None
        best_score = -1
        
        for venue in available_venues:
            score = await self._calculate_execution_score(order, venue)
            if score > best_score:
                best_score = score
                best_venue = venue
        
        return best_venue or available_venues[0]

    async def _route_lowest_cost(self, order: Order, available_venues: list[ExecutionVenue]) -> ExecutionVenue:
        """Route to venue with lowest cost"""
        lowest_cost_venue = None
        lowest_cost = Decimal('999999')
        
        for venue in available_venues:
            venue_info = self.venue_preferences.get(venue, {})
            cost_per_share = venue_info.get('cost_per_share', Decimal('0.01'))
            total_cost = cost_per_share * order.quantity
            
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                lowest_cost_venue = venue
        
        return lowest_cost_venue or available_venues[0]

    async def _route_fastest_execution(self, order: Order, available_venues: list[ExecutionVenue]) -> ExecutionVenue:
        """Route to venue with fastest execution"""
        fastest_venue = None
        lowest_latency = float('inf')
        
        for venue in available_venues:
            venue_info = self.venue_preferences.get(venue, {})
            latency = venue_info.get('avg_latency_ms', 1000)
            
            if latency < lowest_latency:
                lowest_latency = latency
                fastest_venue = venue
        
        return fastest_venue or available_venues[0]

    async def _route_primary_only(self, order: Order, available_venues: list[ExecutionVenue]) -> ExecutionVenue:
        """Route to primary venue only"""
        if ExecutionVenue.ALPACA in available_venues:
            return ExecutionVenue.ALPACA
        elif ExecutionVenue.UPSTOX in available_venues:
            return ExecutionVenue.UPSTOX
        else:
            return available_venues[0]

    async def _route_round_robin(self, order: Order, available_venues: list[ExecutionVenue]) -> ExecutionVenue:
        """Route using round-robin selection"""
        # Simple round-robin based on order ID hash
        index = hash(order.order_id) % len(available_venues)
        return available_venues[index]

    async def _calculate_execution_score(self, order: Order, venue: ExecutionVenue) -> float:
        """Calculate execution quality score for venue"""
        venue_info = self.venue_preferences.get(venue, {})
        performance = self.venue_performance.get(venue, {})
        
        # Base score from venue preferences
        base_score = 0.0
        
        # Cost factor (lower is better)
        cost_per_share = venue_info.get('cost_per_share', Decimal('0.01'))
        cost_score = max(0, 1.0 - float(cost_per_share * 100))  # Normalize to 0-1
        
        # Latency factor (lower is better)
        avg_latency = venue_info.get('avg_latency_ms', 1000)
        latency_score = max(0, 1.0 - (avg_latency / 1000))  # Normalize to 0-1
        
        # Reliability factor
        reliability = venue_info.get('reliability', 0.9)
        reliability_score = reliability
        
        # Performance history factor
        total_orders = performance.get('total_orders', 0)
        successful_orders = performance.get('successful_orders', 0)
        success_rate = successful_orders / max(total_orders, 1)
        
        # Weighted score
        score = (
            cost_score * 0.3 +
            latency_score * 0.3 +
            reliability_score * 0.2 +
            success_rate * 0.2
        )
        
        return score

    def _update_venue_selection(self, venue: ExecutionVenue):
        """Update venue selection statistics"""
        if venue in self.venue_performance:
            self.venue_performance[venue]['total_orders'] += 1

    def update_venue_performance(self, venue: ExecutionVenue, success: bool, latency_ms: float, cost: Decimal):
        """Update venue performance metrics"""
        if venue in self.venue_performance:
            performance = self.venue_performance[venue]
            
            if success:
                performance['successful_orders'] += 1
            
            # Update average latency
            total_orders = performance['total_orders']
            current_avg = performance['avg_latency']
            performance['avg_latency'] = (
                (current_avg * (total_orders - 1) + latency_ms) / total_orders
            )
            
            # Update total cost
            performance['total_cost'] += cost

    async def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""
        return {
            'venue_performance': self.venue_performance,
            'venue_preferences': self.venue_preferences,
            'total_orders_routed': sum(p['total_orders'] for p in self.venue_performance.values()),
            'overall_success_rate': sum(p['successful_orders'] for p in self.venue_performance.values()) / 
                                   max(sum(p['total_orders'] for p in self.venue_performance.values()), 1)
        }

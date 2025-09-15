"""
Market Microstructure Analysis Components
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

# ============================================
# MICROSTRUCTURE ANALYZER
# ============================================

class MicrostructureAnalyzer:
    """
    Advanced market microstructure analysis
    """
    
    def __init__(self):
        self.order_book_analyzer = OrderBookAnalyzer()
        self.flow_analyzer = FlowAnalyzer()
        self.toxicity_calculator = ToxicityCalculator()
        self.venue_analyzer = VenueAnalyzer()
        
    async def initialize(self):
        """Initialize microstructure components"""
        await self.order_book_analyzer.initialize()
        await self.flow_analyzer.initialize()
        
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive microstructure analysis"""
        
        # Analyze order book dynamics
        book_signals = await self.order_book_analyzer.analyze_book(
            market_data.get('order_book', {})
        )
        
        # Analyze order flow
        flow_signals = await self.flow_analyzer.analyze_flow(
            market_data.get('trades', [])
        )
        
        # Calculate toxicity metrics
        toxicity = await self.toxicity_calculator.calculate_vpin(
            market_data.get('trades', [])
        )
        
        # Analyze venue quality
        venue_scores = await self.venue_analyzer.score_venues(
            market_data.get('venue_data', {})
        )
        
        return {
            'book_signals': book_signals,
            'flow_signals': flow_signals,
            'toxicity': toxicity,
            'venue_scores': venue_scores,
            'timestamp': datetime.now()
        }

class OrderBookAnalyzer:
    """Order book dynamics analysis"""
    
    async def initialize(self):
        self.imbalance_threshold = 0.7
        self.depth_levels = 10
        
    async def analyze_book(self, order_book: Dict) -> Dict[str, Any]:
        """Analyze order book for signals"""
        if not order_book:
            return {}
        
        # Calculate order book imbalance
        bid_volume = sum([level['size'] for level in order_book.get('bids', [])])
        ask_volume = sum([level['size'] for level in order_book.get('asks', [])])
        
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Detect hidden liquidity
        hidden_liquidity = self._detect_hidden_liquidity(order_book)
        
        # Calculate spread metrics
        spread_metrics = self._calculate_spread_metrics(order_book)
        
        # Queue position estimation
        queue_position = self._estimate_queue_position(order_book)
        
        return {
            'imbalance': imbalance,
            'hidden_liquidity': hidden_liquidity,
            'spread_metrics': spread_metrics,
            'queue_position': queue_position,
            'signal': 'BUY' if imbalance > self.imbalance_threshold else 'SELL' if imbalance < -self.imbalance_threshold else 'NEUTRAL'
        }
    
    def _detect_hidden_liquidity(self, order_book: Dict) -> float:
        """Detect potential hidden liquidity"""
        # Analyze order book shape for hidden orders
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        # Look for unusual patterns indicating hidden orders
        bid_sizes = [level['size'] for level in bids[:5]]
        ask_sizes = [level['size'] for level in asks[:5]]
        
        # Check for round number clustering (often indicates hidden orders)
        round_numbers = sum(1 for size in bid_sizes + ask_sizes if size % 100 == 0)
        
        return round_numbers / len(bid_sizes + ask_sizes) if bid_sizes + ask_sizes else 0
    
    def _calculate_spread_metrics(self, order_book: Dict) -> Dict[str, float]:
        """Calculate various spread metrics"""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return {}
        
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        
        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid) / 2
        spread_bps = (spread / mid_price) * 10000
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'mid_price': mid_price
        }
    
    def _estimate_queue_position(self, order_book: Dict) -> int:
        """Estimate queue position for order placement"""
        # Complex logic to estimate where an order would sit in queue
        # This is simplified - real implementation would be more sophisticated
        bids = order_book.get('bids', [])
        
        if not bids:
            return 0
        
        # Estimate based on current book depth
        total_ahead = sum([level['size'] for level in bids[:3]])
        
        return int(total_ahead)

class FlowAnalyzer:
    """Order flow analysis"""
    
    async def initialize(self):
        self.trade_history = deque(maxlen=10000)
        self.volume_profile = defaultdict(float)
        
    async def analyze_flow(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze order flow patterns"""
        if not trades:
            return {}
        
        # Update trade history
        for trade in trades:
            self.trade_history.append(trade)
        
        # Calculate flow metrics
        buy_volume = sum(trade['size'] for trade in trades if trade.get('side') == 'BUY')
        sell_volume = sum(trade['size'] for trade in trades if trade.get('side') == 'SELL')
        
        net_flow = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        # Calculate flow imbalance
        flow_imbalance = net_flow / total_volume if total_volume > 0 else 0
        
        # Detect large trades
        large_trades = [t for t in trades if t['size'] > 10000]
        
        # Calculate volume-weighted average price
        vwap = self._calculate_vwap(trades)
        
        return {
            'net_flow': net_flow,
            'flow_imbalance': flow_imbalance,
            'large_trades_count': len(large_trades),
            'vwap': vwap,
            'signal': 'BUY' if flow_imbalance > 0.3 else 'SELL' if flow_imbalance < -0.3 else 'NEUTRAL'
        }
    
    def _calculate_vwap(self, trades: List[Dict]) -> float:
        """Calculate volume-weighted average price"""
        if not trades:
            return 0.0
        
        total_value = sum(trade['price'] * trade['size'] for trade in trades)
        total_volume = sum(trade['size'] for trade in trades)
        
        return total_value / total_volume if total_volume > 0 else 0.0

class ToxicityCalculator:
    """Calculate market toxicity metrics"""
    
    async def calculate_vpin(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate Volume-Synchronized Probability of Informed Trading (VPIN)"""
        if not trades:
            return {'vpin': 0.0, 'toxicity': 'LOW'}
        
        # Simplified VPIN calculation
        # Real implementation would be more sophisticated
        
        # Group trades into volume buckets
        bucket_size = 10000  # 10k shares per bucket
        buckets = []
        current_bucket = {'buy_volume': 0, 'sell_volume': 0}
        
        for trade in trades:
            if trade.get('side') == 'BUY':
                current_bucket['buy_volume'] += trade['size']
            else:
                current_bucket['sell_volume'] += trade['size']
            
            # Check if bucket is full
            total_volume = current_bucket['buy_volume'] + current_bucket['sell_volume']
            if total_volume >= bucket_size:
                buckets.append(current_bucket)
                current_bucket = {'buy_volume': 0, 'sell_volume': 0}
        
        if not buckets:
            return {'vpin': 0.0, 'toxicity': 'LOW'}
        
        # Calculate VPIN
        vpin_values = []
        for bucket in buckets:
            total_volume = bucket['buy_volume'] + bucket['sell_volume']
            if total_volume > 0:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume'])
                vpin = imbalance / total_volume
                vpin_values.append(vpin)
        
        avg_vpin = np.mean(vpin_values) if vpin_values else 0.0
        
        # Determine toxicity level
        if avg_vpin > 0.7:
            toxicity = 'HIGH'
        elif avg_vpin > 0.4:
            toxicity = 'MEDIUM'
        else:
            toxicity = 'LOW'
        
        return {
            'vpin': avg_vpin,
            'toxicity': toxicity,
            'buckets_analyzed': len(buckets)
        }

class VenueAnalyzer:
    """Analyze venue quality and performance"""
    
    async def score_venues(self, venue_data: Dict) -> Dict[str, float]:
        """Score trading venues based on quality metrics"""
        venue_scores = {}
        
        # Default venues with mock scores
        default_venues = ['NYSE', 'NASDAQ', 'ARCA', 'BATS', 'IEX']
        
        for venue in default_venues:
            # Mock scoring based on various factors
            # Real implementation would use actual venue data
            
            # Factors: liquidity, speed, cost, reliability
            liquidity_score = np.random.uniform(0.6, 1.0)
            speed_score = np.random.uniform(0.7, 1.0)
            cost_score = np.random.uniform(0.5, 0.9)
            reliability_score = np.random.uniform(0.8, 1.0)
            
            # Weighted average
            overall_score = (
                liquidity_score * 0.3 +
                speed_score * 0.25 +
                cost_score * 0.25 +
                reliability_score * 0.2
            )
            
            venue_scores[venue] = overall_score
        
        return venue_scores

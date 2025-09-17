"""
Step 13: Market Microstructure & Order Flow Analysis
World-class implementation for order book dynamics and flow analysis
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class OrderBookAnalyzer:
    """Analyzes order book dynamics and microstructure"""
    
    def __init__(self, api):
        self.api = api
        self.order_books = {}
        self.imbalance_history = deque(maxlen=1000)
        
    def get_order_book_imbalance(self, symbol, levels=5):
        """
        Calculate order book imbalance
        Positive = buying pressure, Negative = selling pressure
        """
        try:
            # Get latest trade data (Alpaca limitation - no full order book)
            latest_trade = self.api.get_latest_trade(symbol)
            last_price = float(latest_trade.price)
            
            # Get recent trades to simulate order book analysis
            trades = self.api.get_trades(
                symbol, 
                start=(datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                limit=100
            ).df
            
            if len(trades) < 10:
                return None
            
            # Calculate weighted buy/sell pressure
            buy_volume = 0
            sell_volume = 0
            
            for _, trade in trades.iterrows():
                # Use Lee-Ready algorithm to classify trades
                if trade['price'] >= last_price:
                    buy_volume += trade['size']
                else:
                    sell_volume += trade['size']
            
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
            else:
                imbalance = 0
                
            self.imbalance_history.append({
                'symbol': symbol,
                'imbalance': imbalance,
                'timestamp': datetime.now(),
                'buy_volume': buy_volume,
                'sell_volume': sell_volume
            })
            
            # Generate signal
            if imbalance > 0.3:
                signal = 'STRONG_BUY'
            elif imbalance > 0.1:
                signal = 'BUY'
            elif imbalance < -0.3:
                signal = 'STRONG_SELL'
            elif imbalance < -0.1:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
            
            return {
                'imbalance': imbalance,
                'buy_pressure': buy_volume,
                'sell_pressure': sell_volume,
                'signal': signal,
                'strength': abs(imbalance),
                'confidence': min(95, abs(imbalance) * 200)
            }
            
        except Exception as e:
            logger.error(f"Order book imbalance error for {symbol}: {e}")
            return None
    
    def calculate_vpin_toxicity(self, symbol, bucket_size=50):
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN)
        High VPIN = High probability of adverse selection
        """
        try:
            # Get sufficient trade data
            trades = self.api.get_trades(
                symbol, 
                start=(datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                limit=bucket_size * 10
            ).df
            
            if len(trades) < bucket_size * 2:
                return None
            
            # Create volume buckets
            trades = trades.sort_index()
            trades['cumsum_volume'] = trades['size'].cumsum()
            bucket_volume = trades['size'].sum() / bucket_size
            
            toxicity_scores = []
            
            # Process each bucket
            for i in range(bucket_size):
                start_vol = i * bucket_volume
                end_vol = (i + 1) * bucket_volume
                
                bucket_trades = trades[
                    (trades['cumsum_volume'] > start_vol) & 
                    (trades['cumsum_volume'] <= end_vol)
                ]
                
                if len(bucket_trades) == 0:
                    continue
                
                # Calculate VWAP for bucket
                vwap = (bucket_trades['price'] * bucket_trades['size']).sum() / bucket_trades['size'].sum()
                
                # Classify as buy/sell based on price relative to VWAP
                buy_volume = bucket_trades[bucket_trades['price'] >= vwap]['size'].sum()
                sell_volume = bucket_trades[bucket_trades['price'] < vwap]['size'].sum()
                
                total_vol = buy_volume + sell_volume
                if total_vol > 0:
                    order_imbalance = abs(buy_volume - sell_volume) / total_vol
                    toxicity_scores.append(order_imbalance)
            
            if not toxicity_scores:
                return None
            
            vpin = np.mean(toxicity_scores)
            
            # Determine toxicity level
            if vpin > 0.8:
                toxicity_level = 'EXTREME'
                recommendation = 'AVOID_TRADING'
            elif vpin > 0.6:
                toxicity_level = 'HIGH'
                recommendation = 'REDUCE_SIZE'
            elif vpin > 0.4:
                toxicity_level = 'MEDIUM'
                recommendation = 'NORMAL_CAUTION'
            else:
                toxicity_level = 'LOW'
                recommendation = 'NORMAL'
            
            return {
                'vpin': vpin,
                'toxicity_level': toxicity_level,
                'trading_safe': vpin < 0.6,
                'recommendation': recommendation,
                'bucket_count': len(toxicity_scores),
                'risk_score': min(100, vpin * 125)
            }
            
        except Exception as e:
            logger.error(f"VPIN calculation error for {symbol}: {e}")
            return None
    
    def detect_large_orders(self, symbol, threshold_multiplier=3):
        """Detect unusually large orders indicating institutional activity"""
        try:
            # Get recent trades
            trades = self.api.get_trades(
                symbol, 
                start=(datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                limit=500
            ).df
            
            if len(trades) < 50:
                return None
            
            # Calculate size statistics
            mean_size = trades['size'].mean()
            std_size = trades['size'].std()
            threshold = mean_size + (std_size * threshold_multiplier)
            
            # Find large orders
            large_orders = trades[trades['size'] > threshold]
            
            if len(large_orders) == 0:
                return {
                    'detected': False,
                    'institutional_interest': 'NONE',
                    'large_order_count': 0
                }
            
            # Analyze large order characteristics
            recent_large = large_orders.tail(5)  # Last 5 large orders
            total_large_volume = large_orders['size'].sum()
            total_volume = trades['size'].sum()
            
            # Determine predominant direction
            latest_price = trades['price'].iloc[-1]
            large_buy_volume = large_orders[large_orders['price'] >= latest_price]['size'].sum()
            large_sell_volume = large_orders[large_orders['price'] < latest_price]['size'].sum()
            
            if large_buy_volume > large_sell_volume:
                direction = 'BUY'
                directional_strength = large_buy_volume / (large_buy_volume + large_sell_volume)
            else:
                direction = 'SELL'
                directional_strength = large_sell_volume / (large_buy_volume + large_sell_volume)
            
            participation_rate = (total_large_volume / total_volume) * 100
            
            return {
                'detected': True,
                'large_order_count': len(large_orders),
                'avg_large_size': large_orders['size'].mean(),
                'max_order_size': large_orders['size'].max(),
                'direction': direction,
                'directional_strength': directional_strength,
                'institutional_interest': direction,
                'participation_rate': participation_rate,
                'confidence': min(95, 40 + (participation_rate * 1.5)),
                'time_concentration': self._analyze_time_concentration(large_orders)
            }
            
        except Exception as e:
            logger.error(f"Large order detection error for {symbol}: {e}")
            return None
    
    def _analyze_time_concentration(self, large_orders):
        """Analyze if large orders are concentrated in time (algorithmic trading)"""
        if len(large_orders) < 3:
            return 'SPARSE'
        
        # Calculate time differences between large orders
        time_diffs = large_orders.index.to_series().diff().dt.total_seconds().dropna()
        
        if len(time_diffs) == 0:
            return 'SPARSE'
        
        # If most large orders happen within short time windows
        rapid_orders = (time_diffs < 10).sum()  # Within 10 seconds
        
        if rapid_orders / len(time_diffs) > 0.5:
            return 'ALGORITHMIC'
        elif rapid_orders / len(time_diffs) > 0.2:
            return 'CLUSTERED'
        else:
            return 'DISTRIBUTED'
    
    def analyze_spread_dynamics(self, symbol):
        """Analyze bid-ask spread patterns and liquidity"""
        try:
            # Get current quote
            quote = self.api.get_latest_quote(symbol)
            
            spread = quote.ap - quote.bp
            mid_price = (quote.ap + quote.bp) / 2
            relative_spread = (spread / mid_price) * 10000  # In basis points
            
            # Get recent bars for volatility context
            bars = self.api.get_bars(symbol, '1Min', limit=60).df
            
            if len(bars) > 10:
                # Calculate average true range for volatility
                bars['tr'] = np.maximum(
                    bars['high'] - bars['low'],
                    np.maximum(
                        abs(bars['high'] - bars['close'].shift(1)),
                        abs(bars['low'] - bars['close'].shift(1))
                    )
                )
                atr = bars['tr'].mean()
                spread_to_atr_ratio = spread / atr if atr > 0 else 0
            else:
                spread_to_atr_ratio = 0
            
            # Liquidity assessment
            if relative_spread < 5:
                liquidity = 'EXCELLENT'
                tradeable_score = 95
            elif relative_spread < 15:
                liquidity = 'GOOD'
                tradeable_score = 80
            elif relative_spread < 30:
                liquidity = 'FAIR'
                tradeable_score = 60
            elif relative_spread < 100:
                liquidity = 'POOR'
                tradeable_score = 30
            else:
                liquidity = 'VERY_POOR'
                tradeable_score = 10
            
            return {
                'spread': spread,
                'relative_spread_bps': relative_spread,
                'mid_price': mid_price,
                'bid': quote.bp,
                'ask': quote.ap,
                'liquidity': liquidity,
                'tradeable_score': tradeable_score,
                'spread_to_atr_ratio': spread_to_atr_ratio,
                'market_impact_risk': 'HIGH' if relative_spread > 50 else 'MEDIUM' if relative_spread > 20 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Spread analysis error for {symbol}: {e}")
            return None

class VolumeProfileAnalyzer:
    """Analyzes volume distribution and profiles"""
    
    def __init__(self, api):
        self.api = api
        self.profiles = {}
    
    def calculate_volume_profile(self, symbol, period_days=5):
        """Calculate volume profile for price levels"""
        try:
            # Get intraday bars
            bars = self.api.get_bars(
                symbol, '5Min',
                start=(datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d'),
                limit=2000
            ).df
            
            if len(bars) < 50:
                return None
            
            # Define price range and bins
            price_min = bars['low'].min()
            price_max = bars['high'].max()
            price_range = price_max - price_min
            
            # Create price bins (aim for ~$0.01 resolution for stocks)
            num_bins = min(200, max(50, int(price_range * 100)))
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            
            # Initialize volume profile
            volume_profile = np.zeros(num_bins)
            
            # Distribute volume across price levels within each bar
            for _, bar in bars.iterrows():
                bar_range = bar['high'] - bar['low']
                if bar_range == 0:
                    # Single price bar
                    bin_idx = np.searchsorted(price_bins[:-1], bar['close'], side='right') - 1
                    bin_idx = max(0, min(bin_idx, num_bins - 1))
                    volume_profile[bin_idx] += bar['volume']
                else:
                    # Distribute volume across the bar's range
                    # Assume uniform distribution (could be enhanced with tick data)
                    low_bin = np.searchsorted(price_bins[:-1], bar['low'], side='right') - 1
                    high_bin = np.searchsorted(price_bins[:-1], bar['high'], side='right') - 1
                    
                    low_bin = max(0, min(low_bin, num_bins - 1))
                    high_bin = max(0, min(high_bin, num_bins - 1))
                    
                    if high_bin >= low_bin:
                        bins_spanned = high_bin - low_bin + 1
                        volume_per_bin = bar['volume'] / bins_spanned
                        
                        for i in range(low_bin, high_bin + 1):
                            volume_profile[i] += volume_per_bin
            
            # Find Point of Control (POC) - price level with highest volume
            poc_index = np.argmax(volume_profile)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.7
            
            # Find value area by expanding around POC
            value_area_indices = [poc_index]
            current_volume = volume_profile[poc_index]
            
            left_idx = poc_index - 1
            right_idx = poc_index + 1
            
            while current_volume < value_area_volume and (left_idx >= 0 or right_idx < num_bins):
                left_vol = volume_profile[left_idx] if left_idx >= 0 else 0
                right_vol = volume_profile[right_idx] if right_idx < num_bins else 0
                
                if left_vol >= right_vol and left_idx >= 0:
                    value_area_indices.append(left_idx)
                    current_volume += left_vol
                    left_idx -= 1
                elif right_idx < num_bins:
                    value_area_indices.append(right_idx)
                    current_volume += right_vol
                    right_idx += 1
                else:
                    break
            
            value_area_low = price_bins[min(value_area_indices)]
            value_area_high = price_bins[max(value_area_indices) + 1]
            
            # Current price position
            current_price = bars['close'].iloc[-1]
            
            if current_price > value_area_high:
                position = 'ABOVE_VALUE'
                bias = 'BEARISH_REVERSION'
            elif current_price < value_area_low:
                position = 'BELOW_VALUE'
                bias = 'BULLISH_REVERSION'
            else:
                position = 'IN_VALUE'
                bias = 'NEUTRAL'
            
            return {
                'poc': poc_price,
                'value_area_low': value_area_low,
                'value_area_high': value_area_high,
                'current_price': current_price,
                'position_in_profile': position,
                'bias': bias,
                'volume_profile': volume_profile,
                'price_bins': price_bins,
                'total_volume': total_volume,
                'value_area_volume_pct': (current_volume / total_volume) * 100
            }
            
        except Exception as e:
            logger.error(f"Volume profile error for {symbol}: {e}")
            return None
    
    def find_poc_levels(self, symbol):
        """Find Point of Control and nearby high-volume levels"""
        profile = self.calculate_volume_profile(symbol)
        
        if not profile:
            return None
        
        volume_profile = profile['volume_profile']
        price_bins = profile['price_bins']
        
        # Find local maxima (POC levels)
        from scipy.signal import find_peaks
        
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(volume_profile, prominence=volume_profile.max() * 0.1)
        
        poc_levels = []
        for peak_idx in peaks:
            price = (price_bins[peak_idx] + price_bins[peak_idx + 1]) / 2
            volume = volume_profile[peak_idx]
            strength = volume / volume_profile.max()
            
            poc_levels.append({
                'price': price,
                'volume': volume,
                'strength': strength,
                'type': 'PRIMARY_POC' if peak_idx == np.argmax(volume_profile) else 'SECONDARY_POC'
            })
        
        # Sort by volume
        poc_levels.sort(key=lambda x: x['volume'], reverse=True)
        
        return {
            'poc_levels': poc_levels[:5],  # Top 5 POC levels
            'primary_poc': poc_levels[0] if poc_levels else None,
            'support_levels': [p for p in poc_levels if p['price'] < profile['current_price']],
            'resistance_levels': [p for p in poc_levels if p['price'] > profile['current_price']]
        }
    
    def identify_hvn_lvn(self, symbol):
        """Identify High Volume Nodes and Low Volume Nodes"""
        profile = self.calculate_volume_profile(symbol)
        
        if not profile:
            return None
        
        volume_profile = profile['volume_profile']
        price_bins = profile['price_bins']
        
        # Statistical thresholds
        mean_volume = volume_profile.mean()
        std_volume = volume_profile.std()
        
        hvn_threshold = mean_volume + (std_volume * 1.5)
        lvn_threshold = mean_volume - (std_volume * 0.5)
        
        hvn_levels = []
        lvn_levels = []
        
        for i, vol in enumerate(volume_profile):
            if vol == 0:
                continue
                
            price = (price_bins[i] + price_bins[i + 1]) / 2
            
            if vol > hvn_threshold:
                hvn_levels.append({
                    'price': price,
                    'volume': vol,
                    'strength': vol / volume_profile.max(),
                    'type': 'SUPPORT' if price < profile['current_price'] else 'RESISTANCE'
                })
            elif vol < lvn_threshold:
                lvn_levels.append({
                    'price': price,
                    'volume': vol,
                    'weakness': 1 - (vol / volume_profile.max()),
                    'breakout_potential': 'HIGH'
                })
        
        # Sort by strength/weakness
        hvn_levels.sort(key=lambda x: x['strength'], reverse=True)
        lvn_levels.sort(key=lambda x: x['weakness'], reverse=True)
        
        return {
            'hvn_levels': hvn_levels[:5],
            'lvn_levels': lvn_levels[:5],
            'hvn_count': len(hvn_levels),
            'lvn_count': len(lvn_levels),
            'nearest_hvn': min(hvn_levels, key=lambda x: abs(x['price'] - profile['current_price'])) if hvn_levels else None,
            'nearest_lvn': min(lvn_levels, key=lambda x: abs(x['price'] - profile['current_price'])) if lvn_levels else None
        }

class OrderFlowTracker:
    """Tracks and analyzes order flow patterns"""
    
    def __init__(self, api):
        self.api = api
        self.flow_history = deque(maxlen=5000)
        
    def classify_aggressor_side(self, symbol):
        """Classify trades as buyer or seller initiated using Lee-Ready algorithm"""
        try:
            # Get recent trades and quote
            trades = self.api.get_trades(
                symbol, 
                start=(datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                limit=200
            ).df
            
            if len(trades) < 10:
                return None
            
            quote = self.api.get_latest_quote(symbol)
            mid_price = (quote.ap + quote.bp) / 2
            
            # Classify each trade
            buy_volume = 0
            sell_volume = 0
            buy_count = 0
            sell_count = 0
            
            for i, (timestamp, trade) in enumerate(trades.iterrows()):
                # Lee-Ready classification
                if trade['price'] > mid_price:
                    # Above mid - buyer initiated
                    buy_volume += trade['size']
                    buy_count += 1
                elif trade['price'] < mid_price:
                    # Below mid - seller initiated
                    sell_volume += trade['size']
                    sell_count += 1
                else:
                    # At mid - use tick rule
                    if i > 0:
                        prev_price = trades.iloc[i-1]['price']
                        if trade['price'] > prev_price:
                            buy_volume += trade['size']
                            buy_count += 1
                        elif trade['price'] < prev_price:
                            sell_volume += trade['size']
                            sell_count += 1
                        # If same price, classify as neutral (split equally)
                        else:
                            buy_volume += trade['size'] / 2
                            sell_volume += trade['size'] / 2
            
            total_volume = buy_volume + sell_volume
            total_trades = buy_count + sell_count
            
            if total_volume == 0:
                return None
            
            # Calculate flow metrics
            flow_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            net_flow = buy_volume - sell_volume
            flow_imbalance = net_flow / total_volume
            
            # Determine aggressor side and confidence
            if flow_imbalance > 0.2:
                aggressor_side = 'STRONG_BUYERS'
                confidence = min(95, abs(flow_imbalance) * 200)
            elif flow_imbalance > 0.05:
                aggressor_side = 'BUYERS'
                confidence = min(80, abs(flow_imbalance) * 300)
            elif flow_imbalance < -0.2:
                aggressor_side = 'STRONG_SELLERS'
                confidence = min(95, abs(flow_imbalance) * 200)
            elif flow_imbalance < -0.05:
                aggressor_side = 'SELLERS'
                confidence = min(80, abs(flow_imbalance) * 300)
            else:
                aggressor_side = 'BALANCED'
                confidence = 50
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'total_volume': total_volume,
                'net_flow': net_flow,
                'flow_ratio': flow_ratio,
                'flow_imbalance': flow_imbalance,
                'aggressor_side': aggressor_side,
                'confidence': confidence,
                'avg_buy_size': buy_volume / buy_count if buy_count > 0 else 0,
                'avg_sell_size': sell_volume / sell_count if sell_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Aggressor classification error for {symbol}: {e}")
            return None
    
    def track_institutional_flow(self, symbol):
        """Track potential institutional order flow patterns"""
        try:
            # Get extended trade data
            trades = self.api.get_trades(
                symbol, 
                start=(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                limit=1000
            ).df
            
            if len(trades) < 100:
                return None
            
            # Calculate trade size statistics
            trade_sizes = trades['size']
            size_mean = trade_sizes.mean()
            size_std = trade_sizes.std()
            size_95th = trade_sizes.quantile(0.95)
            size_99th = trade_sizes.quantile(0.99)
            
            # Identify large trades (institutional threshold)
            large_threshold = size_mean + (size_std * 3)
            block_threshold = max(large_threshold, size_95th)
            
            large_trades = trades[trades['size'] > large_threshold]
            block_trades = trades[trades['size'] > block_threshold]
            
            # Time-based clustering analysis
            trades['time_diff'] = trades.index.to_series().diff().dt.total_seconds()
            
            # Rapid-fire trading (algorithmic)
            rapid_trades = trades[trades['time_diff'] < 1]  # < 1 second apart
            
            # Calculate institutional participation metrics
            total_volume = trades['size'].sum()
            large_volume = large_trades['size'].sum()
            block_volume = block_trades['size'].sum()
            
            large_participation = (large_volume / total_volume * 100) if total_volume > 0 else 0
            block_participation = (block_volume / total_volume * 100) if total_volume > 0 else 0
            
            # Pattern detection
            patterns = {
                'large_block_trading': len(block_trades) > 5,
                'algorithmic_trading': len(rapid_trades) > 20,
                'accumulation_pattern': self._detect_accumulation_pattern(large_trades),
                'distribution_pattern': self._detect_distribution_pattern(large_trades),
                'iceberg_orders': self._detect_iceberg_orders(trades)
            }
            
            # Overall institutional assessment
            institutional_score = 0
            if large_participation > 40:
                institutional_score += 30
            if block_participation > 20:
                institutional_score += 25
            if patterns['large_block_trading']:
                institutional_score += 20
            if patterns['algorithmic_trading']:
                institutional_score += 15
            if patterns['accumulation_pattern'] or patterns['distribution_pattern']:
                institutional_score += 10
            
            institutional_detected = institutional_score > 50
            
            # Determine dominant institutional direction
            if len(large_trades) > 0:
                latest_price = trades['price'].iloc[-1]
                large_buy_volume = large_trades[large_trades['price'] >= latest_price]['size'].sum()
                large_sell_volume = large_trades[large_trades['price'] < latest_price]['size'].sum()
                
                if large_buy_volume > large_sell_volume * 1.5:
                    institutional_direction = 'BUYING'
                elif large_sell_volume > large_buy_volume * 1.5:
                    institutional_direction = 'SELLING'
                else:
                    institutional_direction = 'NEUTRAL'
            else:
                institutional_direction = 'NONE'
            
            return {
                'institutional_detected': institutional_detected,
                'institutional_score': institutional_score,
                'institutional_direction': institutional_direction,
                'large_participation_pct': large_participation,
                'block_participation_pct': block_participation,
                'large_trade_count': len(large_trades),
                'block_trade_count': len(block_trades),
                'rapid_trade_count': len(rapid_trades),
                'patterns': patterns,
                'avg_large_size': large_trades['size'].mean() if len(large_trades) > 0 else 0,
                'max_trade_size': trades['size'].max(),
                'recommendation': self._generate_institutional_recommendation(
                    institutional_detected, institutional_direction, institutional_score
                )
            }
            
        except Exception as e:
            logger.error(f"Institutional tracking error for {symbol}: {e}")
            return None
    
    def _detect_accumulation_pattern(self, large_trades):
        """Detect if large trades show accumulation pattern"""
        if len(large_trades) < 5:
            return False
        
        # Check if large trades are predominantly at higher prices over time
        large_trades_sorted = large_trades.sort_index()
        prices = large_trades_sorted['price'].values
        
        # Simple trend check
        if len(prices) >= 3:
            return prices[-1] > prices[0] and np.corrcoef(range(len(prices)), prices)[0, 1] > 0.3
        
        return False
    
    def _detect_distribution_pattern(self, large_trades):
        """Detect if large trades show distribution pattern"""
        if len(large_trades) < 5:
            return False
        
        # Check if large trades are predominantly at lower prices over time
        large_trades_sorted = large_trades.sort_index()
        prices = large_trades_sorted['price'].values
        
        # Simple trend check
        if len(prices) >= 3:
            return prices[-1] < prices[0] and np.corrcoef(range(len(prices)), prices)[0, 1] < -0.3
        
        return False
    
    def _detect_iceberg_orders(self, trades):
        """Detect potential iceberg order execution"""
        # Look for repeated similar-sized trades at similar price levels
        if len(trades) < 20:
            return False
        
        # Group trades by similar size (within 10%)
        size_groups = trades.groupby(pd.cut(trades['size'], bins=20))['size'].count()
        
        # If any size group has many trades, might indicate iceberg
        return any(count > 10 for count in size_groups.values)
    
    def _generate_institutional_recommendation(self, detected, direction, score):
        """Generate trading recommendation based on institutional flow"""
        if not detected:
            return 'NO_INSTITUTIONAL_SIGNAL'
        
        if score > 80:
            if direction == 'BUYING':
                return 'STRONG_FOLLOW_INSTITUTIONAL_BUYING'
            elif direction == 'SELLING':
                return 'STRONG_FOLLOW_INSTITUTIONAL_SELLING'
            else:
                return 'HIGH_INSTITUTIONAL_ACTIVITY_CAUTION'
        elif score > 60:
            if direction == 'BUYING':
                return 'FOLLOW_INSTITUTIONAL_BUYING'
            elif direction == 'SELLING':
                return 'FOLLOW_INSTITUTIONAL_SELLING'
            else:
                return 'MODERATE_INSTITUTIONAL_ACTIVITY'
        else:
            return 'WEAK_INSTITUTIONAL_SIGNAL'
    
    def calculate_flow_metrics(self, symbol):
        """Calculate comprehensive order flow metrics"""
        try:
            aggressor_data = self.classify_aggressor_side(symbol)
            institutional_data = self.track_institutional_flow(symbol)
            
            if not aggressor_data or not institutional_data:
                return None
            
            # Combine metrics
            flow_metrics = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                
                # Aggressor flow
                'net_flow': aggressor_data['net_flow'],
                'flow_imbalance': aggressor_data['flow_imbalance'],
                'aggressor_side': aggressor_data['aggressor_side'],
                'flow_confidence': aggressor_data['confidence'],
                
                # Institutional flow
                'institutional_detected': institutional_data['institutional_detected'],
                'institutional_direction': institutional_data['institutional_direction'],
                'institutional_participation': institutional_data['large_participation_pct'],
                
                # Combined signal
                'combined_signal': self._calculate_combined_flow_signal(aggressor_data, institutional_data),
                'signal_strength': self._calculate_signal_strength(aggressor_data, institutional_data)
            }
            
            # Store in history
            self.flow_history.append(flow_metrics)
            
            return flow_metrics
            
        except Exception as e:
            logger.error(f"Flow metrics calculation error for {symbol}: {e}")
            return None
    
    def _calculate_combined_flow_signal(self, aggressor_data, institutional_data):
        """Calculate combined signal from aggressor and institutional flow"""
        signals = []
        
        # Aggressor signal
        if 'STRONG_BUYERS' in aggressor_data['aggressor_side']:
            signals.append('BUY')
        elif 'BUYERS' in aggressor_data['aggressor_side']:
            signals.append('WEAK_BUY')
        elif 'STRONG_SELLERS' in aggressor_data['aggressor_side']:
            signals.append('SELL')
        elif 'SELLERS' in aggressor_data['aggressor_side']:
            signals.append('WEAK_SELL')
        
        # Institutional signal
        if institutional_data['institutional_detected']:
            if institutional_data['institutional_direction'] == 'BUYING':
                signals.append('INSTITUTIONAL_BUY')
            elif institutional_data['institutional_direction'] == 'SELLING':
                signals.append('INSTITUTIONAL_SELL')
        
        # Combine signals
        buy_signals = sum(1 for s in signals if 'BUY' in s)
        sell_signals = sum(1 for s in signals if 'SELL' in s)
        institutional_signals = sum(1 for s in signals if 'INSTITUTIONAL' in s)
        
        if institutional_signals > 0:
            # Institutional signals have higher weight
            if 'INSTITUTIONAL_BUY' in signals:
                return 'STRONG_BUY'
            elif 'INSTITUTIONAL_SELL' in signals:
                return 'STRONG_SELL'
        
        if buy_signals > sell_signals:
            return 'BUY' if buy_signals > 1 else 'WEAK_BUY'
        elif sell_signals > buy_signals:
            return 'SELL' if sell_signals > 1 else 'WEAK_SELL'
        else:
            return 'NEUTRAL'
    
    def _calculate_signal_strength(self, aggressor_data, institutional_data):
        """Calculate overall signal strength"""
        strength = 0
        
        # Aggressor strength
        strength += aggressor_data['confidence'] * 0.4
        
        # Institutional strength
        if institutional_data['institutional_detected']:
            strength += institutional_data['institutional_score'] * 0.6
        
        return min(100, strength)

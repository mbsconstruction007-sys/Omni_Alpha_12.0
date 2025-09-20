"""
OMNI ALPHA 5.0 - WORKING TRADING STRATEGIES
==========================================
Actually working trading strategies for live trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TradingStrategies:
    """Actually working trading strategies"""
    
    def __init__(self, config):
        self.config = config
        self.min_confidence = 0.6
        self.strategy_weights = {
            'moving_average': 0.25,
            'rsi_mean_reversion': 0.25,
            'bollinger_bands': 0.25,
            'volume_breakout': 0.25
        }
    
    def moving_average_crossover(self, data: pd.DataFrame) -> Dict:
        """Simple but effective MA crossover strategy"""
        try:
            if len(data) < 50:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            data = data.copy()
            data['SMA_20'] = data['close'].rolling(20).mean()
            data['SMA_50'] = data['close'].rolling(50).mean()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 2:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data after MA calculation'}
            
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Calculate trend strength
            trend_strength = abs(last_row['SMA_20'] - last_row['SMA_50']) / last_row['close']
            
            # Bullish crossover
            if prev_row['SMA_20'] <= prev_row['SMA_50'] and last_row['SMA_20'] > last_row['SMA_50']:
                confidence = min(0.8, 0.6 + trend_strength * 10)
                return {
                    'signal': 'BUY', 
                    'confidence': confidence, 
                    'reason': f'Golden Cross (strength: {trend_strength:.3f})',
                    'entry_price': last_row['close'],
                    'stop_loss': last_row['SMA_50'] * 0.98,
                    'take_profit': last_row['close'] * 1.04
                }
            
            # Bearish crossover
            elif prev_row['SMA_20'] >= prev_row['SMA_50'] and last_row['SMA_20'] < last_row['SMA_50']:
                confidence = min(0.8, 0.6 + trend_strength * 10)
                return {
                    'signal': 'SELL', 
                    'confidence': confidence, 
                    'reason': f'Death Cross (strength: {trend_strength:.3f})',
                    'entry_price': last_row['close'],
                    'stop_loss': last_row['SMA_50'] * 1.02,
                    'take_profit': last_row['close'] * 0.96
                }
            
            # Trend continuation
            elif last_row['SMA_20'] > last_row['SMA_50'] and last_row['close'] > last_row['SMA_20']:
                return {
                    'signal': 'HOLD_LONG', 
                    'confidence': 0.55, 
                    'reason': 'Uptrend continuation',
                    'entry_price': last_row['close']
                }
            elif last_row['SMA_20'] < last_row['SMA_50'] and last_row['close'] < last_row['SMA_20']:
                return {
                    'signal': 'HOLD_SHORT', 
                    'confidence': 0.55, 
                    'reason': 'Downtrend continuation',
                    'entry_price': last_row['close']
                }
            
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'No clear MA signal'}
            
        except Exception as e:
            logger.error(f"Moving average strategy error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Strategy error: {e}'}
    
    def rsi_mean_reversion(self, data: pd.DataFrame) -> Dict:
        """RSI oversold/overbought strategy with dynamic thresholds"""
        try:
            if len(data) < 20:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data for RSI'}
            
            data = data.copy()
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 1:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid RSI data'}
            
            last_rsi = data['RSI'].iloc[-1]
            last_close = data['close'].iloc[-1]
            
            # Dynamic thresholds based on volatility
            volatility = data['close'].rolling(20).std().iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
            oversold_threshold = 30 - (volatility * 50)  # More sensitive in high volatility
            overbought_threshold = 70 + (volatility * 50)
            
            # RSI divergence check
            rsi_slope = (data['RSI'].iloc[-1] - data['RSI'].iloc[-5]) / 4 if len(data) >= 5 else 0
            price_slope = (data['close'].iloc[-1] - data['close'].iloc[-5]) / 4 if len(data) >= 5 else 0
            
            # Oversold conditions
            if last_rsi < oversold_threshold:
                confidence = min(0.75, 0.5 + (oversold_threshold - last_rsi) / 100)
                if rsi_slope > 0 and price_slope < 0:  # Bullish divergence
                    confidence += 0.1
                
                return {
                    'signal': 'BUY', 
                    'confidence': confidence, 
                    'reason': f'RSI oversold ({last_rsi:.1f})',
                    'entry_price': last_close,
                    'stop_loss': last_close * 0.97,
                    'take_profit': last_close * 1.05
                }
            
            # Overbought conditions
            elif last_rsi > overbought_threshold:
                confidence = min(0.75, 0.5 + (last_rsi - overbought_threshold) / 100)
                if rsi_slope < 0 and price_slope > 0:  # Bearish divergence
                    confidence += 0.1
                
                return {
                    'signal': 'SELL', 
                    'confidence': confidence, 
                    'reason': f'RSI overbought ({last_rsi:.1f})',
                    'entry_price': last_close,
                    'stop_loss': last_close * 1.03,
                    'take_profit': last_close * 0.95
                }
            
            # Neutral zone with momentum check
            elif 40 < last_rsi < 60:
                if rsi_slope > 2:
                    return {'signal': 'WEAK_BUY', 'confidence': 0.52, 'reason': f'RSI momentum up ({last_rsi:.1f})'}
                elif rsi_slope < -2:
                    return {'signal': 'WEAK_SELL', 'confidence': 0.52, 'reason': f'RSI momentum down ({last_rsi:.1f})'}
            
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'RSI neutral ({last_rsi:.1f})'}
            
        except Exception as e:
            logger.error(f"RSI strategy error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'RSI strategy error: {e}'}
    
    def bollinger_bands_strategy(self, data: pd.DataFrame) -> Dict:
        """Bollinger Bands mean reversion with squeeze detection"""
        try:
            if len(data) < 25:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data for Bollinger Bands'}
            
            data = data.copy()
            
            # Calculate Bollinger Bands
            period = 20
            std_dev = 2
            data['SMA'] = data['close'].rolling(period).mean()
            data['STD'] = data['close'].rolling(period).std()
            data['Upper'] = data['SMA'] + (data['STD'] * std_dev)
            data['Lower'] = data['SMA'] - (data['STD'] * std_dev)
            data['BB_Width'] = (data['Upper'] - data['Lower']) / data['SMA']
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 1:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid Bollinger Bands data'}
            
            last = data.iloc[-1]
            
            # Calculate position within bands
            bb_position = (last['close'] - last['Lower']) / (last['Upper'] - last['Lower'])
            
            # Bollinger Band squeeze detection
            current_width = last['BB_Width']
            avg_width = data['BB_Width'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_width
            is_squeeze = current_width < avg_width * 0.8
            
            # Band touch with momentum
            if last['close'] <= last['Lower']:
                confidence = 0.65
                if is_squeeze:
                    confidence += 0.1  # Higher confidence during squeeze
                
                # Check for reversal signs
                if len(data) >= 2 and data['close'].iloc[-2] < data['Lower'].iloc[-2]:
                    confidence += 0.05  # Multiple touches increase confidence
                
                return {
                    'signal': 'BUY', 
                    'confidence': min(0.8, confidence), 
                    'reason': f'Price at lower band (pos: {bb_position:.2f})',
                    'entry_price': last['close'],
                    'stop_loss': last['Lower'] * 0.99,
                    'take_profit': last['SMA']
                }
            
            elif last['close'] >= last['Upper']:
                confidence = 0.65
                if is_squeeze:
                    confidence += 0.1
                
                if len(data) >= 2 and data['close'].iloc[-2] > data['Upper'].iloc[-2]:
                    confidence += 0.05
                
                return {
                    'signal': 'SELL', 
                    'confidence': min(0.8, confidence), 
                    'reason': f'Price at upper band (pos: {bb_position:.2f})',
                    'entry_price': last['close'],
                    'stop_loss': last['Upper'] * 1.01,
                    'take_profit': last['SMA']
                }
            
            # Mean reversion towards SMA
            elif bb_position > 0.8:  # Near upper band
                return {
                    'signal': 'WEAK_SELL', 
                    'confidence': 0.55, 
                    'reason': f'Price near upper band ({bb_position:.2f})'
                }
            elif bb_position < 0.2:  # Near lower band
                return {
                    'signal': 'WEAK_BUY', 
                    'confidence': 0.55, 
                    'reason': f'Price near lower band ({bb_position:.2f})'
                }
            
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'Price within bands ({bb_position:.2f})'}
            
        except Exception as e:
            logger.error(f"Bollinger Bands strategy error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Bollinger Bands error: {e}'}
    
    def volume_breakout(self, data: pd.DataFrame) -> Dict:
        """Volume-based breakout strategy with price confirmation"""
        try:
            if len(data) < 25:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data for volume analysis'}
            
            data = data.copy()
            
            # Calculate volume metrics
            data['Volume_MA_20'] = data['volume'].rolling(20).mean()
            data['Volume_MA_5'] = data['volume'].rolling(5).mean()
            data['Price_Change'] = data['close'].pct_change()
            data['Volume_Ratio'] = data['volume'] / data['Volume_MA_20']
            
            # Price breakout levels
            data['High_20'] = data['high'].rolling(20).max()
            data['Low_20'] = data['low'].rolling(20).min()
            data['Price_Range'] = data['High_20'] - data['Low_20']
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 1:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid volume data'}
            
            last = data.iloc[-1]
            
            # Volume surge detection
            volume_surge = last['volume'] > last['Volume_MA_20'] * 1.5
            extreme_volume = last['volume'] > last['Volume_MA_20'] * 2.5
            
            # Price breakout detection
            price_breakout_up = last['close'] > last['High_20'] * 0.999
            price_breakout_down = last['close'] < last['Low_20'] * 1.001
            
            # Price momentum
            price_momentum = abs(last['Price_Change'])
            strong_momentum = price_momentum > 0.02  # 2% move
            
            # Volume-Price confirmation
            if volume_surge and strong_momentum:
                base_confidence = 0.65
                
                if extreme_volume:
                    base_confidence += 0.1
                
                if price_breakout_up and last['Price_Change'] > 0:
                    confidence = min(0.85, base_confidence + 0.1)
                    return {
                        'signal': 'BUY', 
                        'confidence': confidence, 
                        'reason': f'Volume breakout up (vol: {last["Volume_Ratio"]:.1f}x, price: {last["Price_Change"]:.2%})',
                        'entry_price': last['close'],
                        'stop_loss': last['Low_20'],
                        'take_profit': last['close'] * (1 + abs(last['Price_Change']) * 2)
                    }
                
                elif price_breakout_down and last['Price_Change'] < 0:
                    confidence = min(0.85, base_confidence + 0.1)
                    return {
                        'signal': 'SELL', 
                        'confidence': confidence, 
                        'reason': f'Volume breakout down (vol: {last["Volume_Ratio"]:.1f}x, price: {last["Price_Change"]:.2%})',
                        'entry_price': last['close'],
                        'stop_loss': last['High_20'],
                        'take_profit': last['close'] * (1 + last['Price_Change'] * 2)
                    }
                
                elif last['Price_Change'] > 0.01:
                    return {
                        'signal': 'BUY', 
                        'confidence': base_confidence, 
                        'reason': f'Volume surge with price up (vol: {last["Volume_Ratio"]:.1f}x)'
                    }
                
                elif last['Price_Change'] < -0.01:
                    return {
                        'signal': 'SELL', 
                        'confidence': base_confidence, 
                        'reason': f'Volume surge with price down (vol: {last["Volume_Ratio"]:.1f}x)'
                    }
            
            # Low volume drift
            elif last['Volume_Ratio'] < 0.7:
                return {
                    'signal': 'HOLD', 
                    'confidence': 0.4, 
                    'reason': f'Low volume drift ({last["Volume_Ratio"]:.1f}x)'
                }
            
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'Normal volume ({last["Volume_Ratio"]:.1f}x)'}
            
        except Exception as e:
            logger.error(f"Volume breakout strategy error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Volume strategy error: {e}'}
    
    def momentum_strategy(self, data: pd.DataFrame) -> Dict:
        """Additional momentum-based strategy"""
        try:
            if len(data) < 15:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data for momentum'}
            
            data = data.copy()
            
            # Calculate momentum indicators
            data['ROC_5'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) * 100
            data['ROC_10'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) * 100
            data['Price_MA_10'] = data['close'].rolling(10).mean()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 1:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid momentum data'}
            
            last = data.iloc[-1]
            
            # Strong momentum conditions
            strong_up_momentum = last['ROC_5'] > 2 and last['ROC_10'] > 1 and last['close'] > last['Price_MA_10']
            strong_down_momentum = last['ROC_5'] < -2 and last['ROC_10'] < -1 and last['close'] < last['Price_MA_10']
            
            if strong_up_momentum:
                confidence = min(0.75, 0.6 + abs(last['ROC_5']) / 100)
                return {
                    'signal': 'BUY', 
                    'confidence': confidence, 
                    'reason': f'Strong upward momentum (ROC5: {last["ROC_5"]:.1f}%)'
                }
            
            elif strong_down_momentum:
                confidence = min(0.75, 0.6 + abs(last['ROC_5']) / 100)
                return {
                    'signal': 'SELL', 
                    'confidence': confidence, 
                    'reason': f'Strong downward momentum (ROC5: {last["ROC_5"]:.1f}%)'
                }
            
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'Moderate momentum (ROC5: {last["ROC_5"]:.1f}%)'}
            
        except Exception as e:
            logger.error(f"Momentum strategy error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Momentum strategy error: {e}'}
    
    def combine_signals(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """Combine multiple strategies for higher confidence with advanced logic"""
        try:
            if len(data) < 50:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data for analysis'}
            
            # Get all strategy signals
            signals = []
            
            ma_signal = self.moving_average_crossover(data)
            signals.append(('moving_average', ma_signal))
            
            rsi_signal = self.rsi_mean_reversion(data)
            signals.append(('rsi', rsi_signal))
            
            bb_signal = self.bollinger_bands_strategy(data)
            signals.append(('bollinger', bb_signal))
            
            vol_signal = self.volume_breakout(data)
            signals.append(('volume', vol_signal))
            
            mom_signal = self.momentum_strategy(data)
            signals.append(('momentum', mom_signal))
            
            # Filter out error signals
            valid_signals = [(name, sig) for name, sig in signals if sig['confidence'] > 0]
            
            if not valid_signals:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid strategy signals'}
            
            # Categorize signals
            buy_signals = []
            sell_signals = []
            hold_signals = []
            
            for name, signal in valid_signals:
                if signal['signal'] in ['BUY', 'WEAK_BUY', 'HOLD_LONG']:
                    buy_signals.append((name, signal))
                elif signal['signal'] in ['SELL', 'WEAK_SELL', 'HOLD_SHORT']:
                    sell_signals.append((name, signal))
                else:
                    hold_signals.append((name, signal))
            
            # Weighted confidence calculation
            total_buy_weight = sum(self.strategy_weights.get(name, 0.2) * sig['confidence'] 
                                 for name, sig in buy_signals)
            total_sell_weight = sum(self.strategy_weights.get(name, 0.2) * sig['confidence'] 
                                  for name, sig in sell_signals)
            
            # Decision logic with consensus requirement
            min_consensus = 0.6  # Require 60% weighted consensus
            
            if total_buy_weight > min_consensus and total_buy_weight > total_sell_weight * 1.5:
                # Strong buy consensus
                combined_confidence = min(0.9, total_buy_weight)
                strongest_buy = max(buy_signals, key=lambda x: x[1]['confidence'])
                
                reasons = [f"{name}: {sig['reason']}" for name, sig in buy_signals[:3]]
                
                result = {
                    'signal': 'BUY',
                    'confidence': combined_confidence,
                    'reason': f'Consensus BUY ({len(buy_signals)}/{len(valid_signals)} strategies)',
                    'details': reasons,
                    'entry_price': strongest_buy[1].get('entry_price', data['close'].iloc[-1]),
                    'stop_loss': strongest_buy[1].get('stop_loss'),
                    'take_profit': strongest_buy[1].get('take_profit'),
                    'strategy_count': len(buy_signals),
                    'total_strategies': len(valid_signals)
                }
                
            elif total_sell_weight > min_consensus and total_sell_weight > total_buy_weight * 1.5:
                # Strong sell consensus
                combined_confidence = min(0.9, total_sell_weight)
                strongest_sell = max(sell_signals, key=lambda x: x[1]['confidence'])
                
                reasons = [f"{name}: {sig['reason']}" for name, sig in sell_signals[:3]]
                
                result = {
                    'signal': 'SELL',
                    'confidence': combined_confidence,
                    'reason': f'Consensus SELL ({len(sell_signals)}/{len(valid_signals)} strategies)',
                    'details': reasons,
                    'entry_price': strongest_sell[1].get('entry_price', data['close'].iloc[-1]),
                    'stop_loss': strongest_sell[1].get('stop_loss'),
                    'take_profit': strongest_sell[1].get('take_profit'),
                    'strategy_count': len(sell_signals),
                    'total_strategies': len(valid_signals)
                }
                
            else:
                # No clear consensus or conflicting signals
                avg_confidence = np.mean([sig['confidence'] for _, sig in valid_signals])
                conflict_level = abs(total_buy_weight - total_sell_weight)
                
                result = {
                    'signal': 'HOLD',
                    'confidence': min(0.6, avg_confidence),
                    'reason': f'Mixed signals (buy: {total_buy_weight:.2f}, sell: {total_sell_weight:.2f})',
                    'details': [f"{name}: {sig['signal']}" for name, sig in valid_signals[:3]],
                    'buy_weight': total_buy_weight,
                    'sell_weight': total_sell_weight,
                    'conflict_level': conflict_level,
                    'total_strategies': len(valid_signals)
                }
            
            # Add symbol and timestamp
            result['symbol'] = symbol
            result['timestamp'] = datetime.now()
            result['data_points'] = len(data)
            
            return result
            
        except Exception as e:
            logger.error(f"Signal combination error: {e}")
            return {
                'signal': 'HOLD', 
                'confidence': 0.0, 
                'reason': f'Signal combination error: {e}',
                'symbol': symbol,
                'timestamp': datetime.now()
            }
    
    def get_strategy_performance(self) -> Dict:
        """Get performance metrics for each strategy"""
        return {
            'strategies': list(self.strategy_weights.keys()),
            'weights': self.strategy_weights,
            'min_confidence': self.min_confidence,
            'status': 'active'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    config = {'min_confidence': 0.6}
    strategies = TradingStrategies(config)
    
    # Get sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="6mo")
    data.columns = data.columns.str.lower()
    
    # Test combined signals
    signal = strategies.combine_signals(data, "AAPL")
    print(f"Signal: {signal}")

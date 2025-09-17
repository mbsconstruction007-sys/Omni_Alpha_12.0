'''Step 8: Advanced Analytics Engine'''

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class AnalyticsEngine:
    def __init__(self, api):
        self.api = api
        self.analysis_cache = {}
        
    def analyze_symbol(self, symbol):
        '''Comprehensive symbol analysis'''
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical': self.technical_analysis(symbol),
                'statistical': self.statistical_analysis(symbol),
                'momentum': self.momentum_analysis(symbol),
                'volume': self.volume_analysis(symbol),
                'risk': self.risk_analysis(symbol)
            }
            
            # Generate score
            analysis['composite_score'] = self.calculate_composite_score(analysis)
            analysis['recommendation'] = self.generate_recommendation(analysis)
            
            return analysis
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def technical_analysis(self, symbol):
        '''Technical indicator analysis'''
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
            ).df
            
            if len(bars) < 20:
                return None
            
            current_price = bars['close'].iloc[-1]
            
            # Moving averages
            sma_20 = bars['close'].rolling(20).mean().iloc[-1]
            sma_50 = bars['close'].rolling(50).mean().iloc[-1] if len(bars) >= 50 else sma_20
            
            # Trend
            trend = 'UPTREND' if current_price > sma_20 > sma_50 else 'DOWNTREND'
            
            # Support/Resistance
            support = bars['low'].rolling(20).min().iloc[-1]
            resistance = bars['high'].rolling(20).max().iloc[-1]
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend': trend,
                'support': support,
                'resistance': resistance,
                'distance_from_support': (current_price - support) / support * 100,
                'distance_from_resistance': (resistance - current_price) / current_price * 100
            }
        except Exception as e:
            print(f"Error in technical analysis for {symbol}: {e}")
            return None
    
    def statistical_analysis(self, symbol):
        '''Statistical analysis'''
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).df
            
            returns = bars['close'].pct_change().dropna()
            
            return {
                'mean_return': returns.mean() * 100,
                'volatility': returns.std() * 100,
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'var_95': np.percentile(returns, 5) * 100,
                'expected_move': returns.std() * np.sqrt(252) * 100
            }
        except Exception as e:
            print(f"Error in statistical analysis for {symbol}: {e}")
            return None
    
    def momentum_analysis(self, symbol):
        '''Momentum indicators'''
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).df
            
            # Rate of change
            roc = (bars['close'].iloc[-1] / bars['close'].iloc[-10] - 1) * 100
            
            # Price acceleration
            recent_change = bars['close'].pct_change().iloc[-5:].mean()
            older_change = bars['close'].pct_change().iloc[-10:-5].mean()
            acceleration = recent_change - older_change
            
            return {
                'rate_of_change_10d': roc,
                'acceleration': acceleration * 100,
                'momentum_score': roc + (acceleration * 100),
                'strength': 'STRONG' if abs(roc) > 5 else 'MODERATE' if abs(roc) > 2 else 'WEAK'
            }
        except Exception as e:
            print(f"Error in momentum analysis for {symbol}: {e}")
            return None
    
    def volume_analysis(self, symbol):
        '''Volume analysis'''
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            ).df
            
            current_volume = bars['volume'].iloc[-1]
            avg_volume = bars['volume'].mean()
            
            # Volume trend
            recent_avg = bars['volume'].iloc[-5:].mean()
            older_avg = bars['volume'].iloc[-10:-5].mean()
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume,
                'volume_trend': 'INCREASING' if recent_avg > older_avg else 'DECREASING',
                'unusual_volume': current_volume > avg_volume * 1.5
            }
        except Exception as e:
            print(f"Error in volume analysis for {symbol}: {e}")
            return None
    
    def risk_analysis(self, symbol):
        '''Risk metrics'''
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            ).df
            
            returns = bars['close'].pct_change().dropna()
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            return {
                'max_drawdown': drawdown.min(),
                'current_drawdown': drawdown.iloc[-1],
                'beta': 1.0,  # Simplified - would calculate vs SPY
                'risk_rating': 'HIGH' if drawdown.min() < -20 else 'MEDIUM' if drawdown.min() < -10 else 'LOW'
            }
        except Exception as e:
            print(f"Error in risk analysis for {symbol}: {e}")
            return None
    
    def calculate_composite_score(self, analysis):
        '''Calculate overall score'''
        try:
            score = 0
            
            # Technical score
            if analysis['technical']:
                if analysis['technical']['trend'] == 'UPTREND':
                    score += 20
                if analysis['technical']['distance_from_support'] > 5:
                    score += 10
            
            # Momentum score
            if analysis['momentum']:
                if analysis['momentum']['momentum_score'] > 5:
                    score += 20
                elif analysis['momentum']['momentum_score'] > 0:
                    score += 10
            
            # Volume score
            if analysis['volume']:
                if analysis['volume']['unusual_volume'] and analysis['volume']['volume_trend'] == 'INCREASING':
                    score += 15
            
            # Statistical score
            if analysis['statistical']:
                if analysis['statistical']['mean_return'] > 0:
                    score += 15
                if analysis['statistical']['volatility'] < 30:
                    score += 10
            
            # Risk adjustment
            if analysis['risk']:
                if analysis['risk']['risk_rating'] == 'HIGH':
                    score -= 20
                elif analysis['risk']['risk_rating'] == 'MEDIUM':
                    score -= 10
            
            return max(0, min(100, score))
        except Exception as e:
            print(f"Error calculating composite score: {e}")
            return 50
    
    def generate_recommendation(self, analysis):
        '''Generate trading recommendation'''
        try:
            score = analysis['composite_score']
            
            if score >= 70:
                return 'STRONG BUY'
            elif score >= 50:
                return 'BUY'
            elif score >= 30:
                return 'HOLD'
            elif score >= 10:
                return 'SELL'
            else:
                return 'STRONG SELL'
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return 'HOLD'

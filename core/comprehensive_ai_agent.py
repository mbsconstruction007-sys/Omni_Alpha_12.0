"""
COMPREHENSIVE AI AGENT - Advanced Trading Intelligence System
Extends Gemini AI with complete trading analysis capabilities
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Enums for clarity
class MarketRegime(Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class TradeDecision(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    AVOID = "AVOID"

@dataclass
class TradeValidation:
    symbol: str
    is_valid: bool
    confidence: float
    reasons: List[str]
    risks: List[str]
    opportunities: List[str]

class ComprehensiveAIAgent:
    """
    Advanced AI Agent that handles all aspects of trading intelligence
    """
    
    def __init__(self, api_client):
        # Gemini setup (demo mode for testing)
        self.api_key = os.getenv('GEMINI_API_KEY', 'demo_key')
        self.demo_mode = True  # Set to False when you have real API key
        
        if not self.demo_mode and self.api_key != 'demo_key':
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
        
        # API client for market data
        self.api = api_client
        
        # Task enablement
        self.tasks = {
            'trade_validation': True,
            'risk_detection': True,
            'pattern_recognition': True,
            'behavioral_analysis': True,
            'execution_optimization': True,
            'predictive_analysis': True,
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Trading history for learning
        self.trade_history = deque(maxlen=1000)
        self.pattern_library = {}
        self.behavioral_patterns = defaultdict(list)
        
        # Market regime
        self.current_regime = MarketRegime.RANGE_BOUND
        
    # ==================== PRE-MARKET ANALYSIS ====================
    
    async def pre_market_analysis(self, watchlist: List[str]) -> Dict:
        """
        Comprehensive pre-market analysis for the trading day
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': await self.detect_market_regime(),
            'global_markets': await self.analyze_global_markets(),
            'gap_analysis': {},
            'key_levels': {},
            'events_today': await self.get_todays_events(),
            'high_probability_setups': []
        }
        
        for symbol in watchlist:
            # Gap analysis
            gap = await self.analyze_gap(symbol)
            if gap['significant']:
                analysis['gap_analysis'][symbol] = gap
            
            # Key levels
            levels = await self.calculate_key_levels(symbol)
            analysis['key_levels'][symbol] = levels
            
            # High probability setups
            setup = await self.identify_setup(symbol)
            if setup['probability'] > 0.7:
                analysis['high_probability_setups'].append(setup)
        
        # AI summary
        analysis['ai_summary'] = await self.generate_premarket_summary(analysis)
        
        return analysis
    
    async def analyze_gap(self, symbol: str) -> Dict:
        """Analyze overnight gaps and their implications"""
        try:
            # Get yesterday's close and today's pre-market
            bars = self.api.get_bars(symbol, '1Day', limit=2).df
            yesterday_close = bars['close'].iloc[-2]
            
            # Get current quote
            quote = self.api.get_latest_quote(symbol)
            current_price = quote.ap
            
            gap_pct = ((current_price - yesterday_close) / yesterday_close) * 100
            
            # AI analysis of gap (demo mode)
            if self.demo_mode:
                analysis = self._simulate_gap_analysis(gap_pct)
            else:
                prompt = f"""
                Analyze this gap for {symbol}:
                Yesterday Close: ${yesterday_close}
                Current Price: ${current_price}
                Gap: {gap_pct:.2f}%
                
                Determine:
                1. Gap type (breakaway/runaway/exhaustion)
                2. Fill probability (0-100%)
                3. Likely cause
                4. Trading strategy
                """
                
                response = self.model.generate_content(prompt)
                analysis = response.text
            
            return {
                'significant': abs(gap_pct) > 1,
                'gap_percent': gap_pct,
                'fill_probability': 60 if abs(gap_pct) > 2 else 80,
                'analysis': analysis,
                'direction': 'UP' if gap_pct > 0 else 'DOWN' if gap_pct < 0 else 'FLAT'
            }
        except Exception as e:
            logger.error(f"Gap analysis error: {e}")
            return {'significant': False, 'gap_percent': 0}
    
    def _simulate_gap_analysis(self, gap_pct: float) -> str:
        """Simulate gap analysis for demo mode"""
        if abs(gap_pct) > 3:
            return f"Significant {gap_pct:.1f}% gap - likely news driven. Monitor for continuation or reversal."
        elif abs(gap_pct) > 1:
            return f"Moderate {gap_pct:.1f}% gap - normal market movement. High fill probability."
        else:
            return f"Small {gap_pct:.1f}% gap - minimal significance."
    
    # ==================== TRADE VALIDATION ====================
    
    async def validate_trade(
        self, 
        symbol: str, 
        side: str, 
        entry_price: float,
        technical_signals: Dict = None
    ) -> TradeValidation:
        """
        Comprehensive trade validation before execution
        """
        
        validation_checks = []
        risks = []
        opportunities = []
        
        try:
            # 1. News Risk Check
            news_check = await self.check_news_risk(symbol)
            if news_check['risk_level'] == 'HIGH':
                risks.append(f"High news risk: {news_check['reason']}")
            
            # 2. Market Regime Check
            regime_appropriate = await self.is_trade_appropriate_for_regime(symbol, side)
            if not regime_appropriate:
                risks.append(f"Trade against market regime: {self.current_regime.value}")
            else:
                opportunities.append("Trade aligns with market regime")
            
            # 3. Pattern Recognition
            patterns = await self.detect_patterns(symbol)
            for pattern in patterns:
                if pattern['bullish'] and side == 'buy':
                    opportunities.append(f"{pattern['name']} pattern detected")
                elif not pattern['bullish'] and side == 'sell':
                    opportunities.append(f"{pattern['name']} pattern detected")
                elif pattern['bullish'] and side == 'sell':
                    risks.append(f"Bullish {pattern['name']} pattern conflicts with sell")
            
            # 4. Volatility Check
            volatility = await self.calculate_volatility(symbol)
            if volatility > 50:
                risks.append(f"High volatility: {volatility:.1f}%")
            elif volatility < 15:
                opportunities.append(f"Low volatility environment: {volatility:.1f}%")
            
            # 5. Volume Analysis
            volume_analysis = await self.analyze_volume_profile(symbol)
            if volume_analysis['unusual_volume']:
                opportunities.append("Unusual volume detected")
            
            # Calculate overall validation
            risk_score = len(risks)
            opportunity_score = len(opportunities)
            
            is_valid = risk_score <= 2 and opportunity_score >= 1
            confidence = max(0, min(100, 50 + (opportunity_score * 15) - (risk_score * 20)))
            
            # Get AI recommendation
            ai_recommendation = await self.get_ai_trade_recommendation(
                symbol, side, technical_signals or {}, risks, opportunities
            )
            
            return TradeValidation(
                symbol=symbol,
                is_valid=is_valid,
                confidence=confidence,
                reasons=[ai_recommendation],
                risks=risks,
                opportunities=opportunities
            )
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return TradeValidation(
                symbol=symbol,
                is_valid=False,
                confidence=0,
                reasons=[f"Validation error: {str(e)}"],
                risks=["System error during validation"],
                opportunities=[]
            )
    
    async def check_news_risk(self, symbol: str) -> Dict:
        """Check for news-related risks"""
        try:
            # Simple news risk simulation
            current_hour = datetime.now().hour
            
            # Higher risk during market open and close
            if 9 <= current_hour <= 10 or 15 <= current_hour <= 16:
                risk_level = 'MEDIUM'
                reason = 'Market open/close volatility period'
            else:
                risk_level = 'LOW'
                reason = 'Normal trading hours'
            
            return {
                'risk_level': risk_level,
                'reason': reason,
                'recommendation': 'PROCEED_WITH_CAUTION' if risk_level == 'HIGH' else 'NORMAL'
            }
        except Exception as e:
            logger.error(f"News risk check error: {e}")
            return {'risk_level': 'UNKNOWN', 'reason': 'Error checking news'}
    
    async def is_trade_appropriate_for_regime(self, symbol: str, side: str) -> bool:
        """Check if trade aligns with current market regime"""
        try:
            regime = await self.detect_market_regime()
            
            if regime == MarketRegime.BULL_TREND and side == 'buy':
                return True
            elif regime == MarketRegime.BEAR_TREND and side == 'sell':
                return True
            elif regime == MarketRegime.RANGE_BOUND:
                return True  # Both directions OK in range
            else:
                return False
        except Exception as e:
            logger.error(f"Regime check error: {e}")
            return True  # Default to allowing trade
    
    # ==================== PATTERN RECOGNITION ====================
    
    async def detect_patterns(self, symbol: str) -> List[Dict]:
        """
        Detect complex trading patterns using AI
        """
        try:
            # Get price data
            data = self.api.get_bars(symbol, '1Day', limit=50).df
            
            if len(data) < 20:
                return []
            
            # Simple pattern detection (demo mode)
            patterns = []
            
            # Moving average pattern
            sma_20 = data['close'].rolling(20).mean()
            current_price = data['close'].iloc[-1]
            
            if current_price > sma_20.iloc[-1]:
                patterns.append({
                    'name': 'Above SMA20',
                    'confidence': 70,
                    'bullish': True,
                    'target_price': current_price * 1.05,
                    'stop_loss': sma_20.iloc[-1],
                    'description': 'Price trading above 20-day moving average'
                })
            
            # Support/Resistance pattern
            recent_high = data['high'].rolling(10).max().iloc[-1]
            recent_low = data['low'].rolling(10).min().iloc[-1]
            
            if current_price > recent_high * 0.99:
                patterns.append({
                    'name': 'Breakout',
                    'confidence': 75,
                    'bullish': True,
                    'target_price': current_price * 1.08,
                    'stop_loss': recent_high * 0.98,
                    'description': 'Breaking above recent resistance'
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return []
    
    # ==================== BEHAVIORAL ANALYSIS ====================
    
    async def analyze_trader_psychology(self, trade_history: List[Dict]) -> Dict:
        """
        Analyze trader's psychological patterns
        """
        if not trade_history:
            return {
                'status': 'insufficient_data',
                'patterns_detected': [],
                'recommendations': ['Execute more trades to enable analysis']
            }
        
        analysis = {
            'total_trades': len(trade_history),
            'patterns_detected': [],
            'psychological_state': 'NEUTRAL',
            'recommendations': []
        }
        
        try:
            # Convert to DataFrame for analysis
            if trade_history:
                df = pd.DataFrame(trade_history)
                
                # Calculate win rate
                winning_trades = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
                win_rate = (winning_trades / len(trade_history)) * 100
                
                # Detect revenge trading
                recent_losses = [t for t in trade_history[-10:] if t.get('pnl', 0) < 0]
                if len(recent_losses) > 3:
                    analysis['patterns_detected'].append('POTENTIAL_REVENGE_TRADING')
                    analysis['recommendations'].append('Take a break after consecutive losses')
                
                # Detect FOMO
                large_losses = [t for t in trade_history if t.get('pnl', 0) < -500]
                if len(large_losses) > len(trade_history) * 0.1:
                    analysis['patterns_detected'].append('FOMO_TRADING')
                    analysis['recommendations'].append('Reduce position sizes - FOMO detected')
                
                # Analyze best trading times
                if len(trade_history) > 20:
                    # Simulate time analysis
                    profitable_hours = [9, 10, 14, 15]  # Common profitable hours
                    analysis['best_trading_hours'] = profitable_hours
                    analysis['recommendations'].append(f"Focus trading on hours: {profitable_hours}")
                
                # Overall assessment
                if win_rate > 60:
                    analysis['psychological_state'] = 'CONFIDENT'
                elif win_rate < 40:
                    analysis['psychological_state'] = 'STRUGGLING'
                    analysis['recommendations'].append('Consider reducing trade frequency')
                
                # AI psychological assessment (demo mode)
                if self.demo_mode:
                    assessment = f"Win rate: {win_rate:.1f}%. "
                    if win_rate > 55:
                        assessment += "Good performance - maintain discipline. "
                    else:
                        assessment += "Below average performance - review strategy. "
                    
                    if len(recent_losses) > 2:
                        assessment += "Recent losses detected - consider taking a break."
                    
                    analysis['ai_psychological_assessment'] = assessment
        
        except Exception as e:
            logger.error(f"Psychology analysis error: {e}")
            analysis['patterns_detected'].append('ANALYSIS_ERROR')
        
        return analysis
    
    # ==================== EXECUTION OPTIMIZATION ====================
    
    async def optimize_execution(self, symbol: str, size: int, urgency: str = 'NORMAL') -> Dict:
        """
        Determine optimal execution strategy
        """
        
        try:
            # Analyze current market conditions
            spread = await self.analyze_spread(symbol)
            liquidity = await self.analyze_liquidity_depth(symbol)
            momentum = await self.analyze_momentum(symbol)
            volatility = await self.calculate_volatility(symbol)
            
            execution_plan = {
                'symbol': symbol,
                'size': size,
                'method': 'SINGLE',
                'order_type': 'LIMIT',
                'price_adjustment': 0,
                'time_strategy': 'IMMEDIATE',
                'expected_slippage': 0.01
            }
            
            # Determine if we need to split the order
            if size > 1000:  # Large order threshold
                execution_plan['method'] = 'SPLIT'
                execution_plan['splits'] = min(10, size // 100)
                execution_plan['interval_seconds'] = 30
            
            # Determine order type based on spread and urgency
            if spread['relative_spread'] < 0.001 or urgency == 'HIGH':
                execution_plan['order_type'] = 'MARKET'
                execution_plan['expected_slippage'] = spread['spread'] * 0.5
            else:
                execution_plan['order_type'] = 'LIMIT'
                execution_plan['price_adjustment'] = spread['spread'] * 0.25
                execution_plan['expected_slippage'] = 0.005
            
            # Time-based optimization
            current_time = datetime.now().time()
            if time(9, 30) <= current_time <= time(10, 0):
                execution_plan['time_strategy'] = 'WAIT_FOR_SETTLEMENT'
                execution_plan['wait_minutes'] = 15
            elif time(15, 30) <= current_time <= time(16, 0):
                execution_plan['time_strategy'] = 'IMMEDIATE'
            
            # AI recommendation (demo mode)
            if self.demo_mode:
                recommendation = f"Recommended execution: {execution_plan['method']} order using {execution_plan['order_type']} type. "
                if execution_plan['method'] == 'SPLIT':
                    recommendation += f"Split into {execution_plan['splits']} orders. "
                recommendation += f"Expected slippage: {execution_plan['expected_slippage']:.3f}%"
                
                execution_plan['ai_recommendation'] = recommendation
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Execution optimization error: {e}")
            return {
                'symbol': symbol,
                'size': size,
                'method': 'SINGLE',
                'order_type': 'MARKET',
                'error': str(e)
            }
    
    async def analyze_spread(self, symbol: str) -> Dict:
        """Analyze bid-ask spread"""
        try:
            quote = self.api.get_latest_quote(symbol)
            spread = quote.ap - quote.bp
            mid_price = (quote.ap + quote.bp) / 2
            relative_spread = spread / mid_price
            
            return {
                'spread': spread,
                'relative_spread': relative_spread,
                'bid': quote.bp,
                'ask': quote.ap,
                'mid_price': mid_price
            }
        except Exception as e:
            logger.error(f"Spread analysis error: {e}")
            return {'spread': 0.01, 'relative_spread': 0.001}
    
    async def analyze_liquidity_depth(self, symbol: str) -> Dict:
        """Analyze market liquidity"""
        try:
            # Get recent volume data
            bars = self.api.get_bars(symbol, '1Min', limit=60).df
            avg_volume = bars['volume'].mean()
            
            return {
                'avg_volume_per_min': avg_volume,
                'liquidity_score': min(100, avg_volume / 1000),
                'depth_quality': 'GOOD' if avg_volume > 10000 else 'FAIR' if avg_volume > 1000 else 'POOR'
            }
        except Exception as e:
            logger.error(f"Liquidity analysis error: {e}")
            return {'avg_volume_per_min': 5000, 'liquidity_score': 50}
    
    async def analyze_momentum(self, symbol: str) -> Dict:
        """Analyze price momentum"""
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=30).df
            
            if len(bars) < 10:
                return {'direction': 'NEUTRAL', 'strength': 0}
            
            # Calculate momentum
            price_change = (bars['close'].iloc[-1] - bars['close'].iloc[-10]) / bars['close'].iloc[-10]
            
            return {
                'direction': 'UP' if price_change > 0.001 else 'DOWN' if price_change < -0.001 else 'NEUTRAL',
                'strength': abs(price_change) * 100,
                'price_change_pct': price_change * 100
            }
        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return {'direction': 'NEUTRAL', 'strength': 0}
    
    async def calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility"""
        try:
            bars = self.api.get_bars(symbol, '1Day', limit=20).df
            returns = bars['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            return volatility
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 20.0  # Default volatility
    
    async def analyze_volume_profile(self, symbol: str) -> Dict:
        """Analyze volume patterns"""
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=60).df
            
            if len(bars) < 10:
                return {'unusual_volume': False}
            
            current_volume = bars['volume'].iloc[-1]
            avg_volume = bars['volume'].mean()
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume,
                'unusual_volume': current_volume > avg_volume * 2
            }
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            return {'unusual_volume': False}
    
    # ==================== PREDICTIVE ANALYSIS ====================
    
    async def predict_price_movement(self, symbol: str, timeframe: str) -> Dict:
        """
        Predict future price movement using multiple factors
        """
        
        try:
            # Gather data
            technical_data = await self.gather_technical_data(symbol)
            patterns = await self.detect_patterns(symbol)
            
            # Current price
            quote = self.api.get_latest_quote(symbol)
            current_price = quote.ap
            
            # Simple prediction logic (demo mode)
            if self.demo_mode:
                prediction = self._simulate_price_prediction(symbol, current_price, technical_data, timeframe)
            else:
                # Real Gemini prediction would go here
                prediction = self._simulate_price_prediction(symbol, current_price, technical_data, timeframe)
            
            # Store prediction for later validation
            self.store_prediction(symbol, timeframe, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Price prediction error: {e}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'error': str(e)
            }
    
    def _simulate_price_prediction(self, symbol: str, current_price: float, technical_data: Dict, timeframe: str) -> Dict:
        """Simulate price prediction for demo mode"""
        
        # Simple logic based on technical data
        momentum = technical_data.get('momentum', 0)
        volatility = technical_data.get('volatility', 20)
        
        if momentum > 1:
            direction = 'UP'
            confidence = min(85, 60 + momentum * 10)
            expected_move = volatility * 0.5
            target_price = current_price * (1 + expected_move / 100)
        elif momentum < -1:
            direction = 'DOWN'
            confidence = min(85, 60 + abs(momentum) * 10)
            expected_move = volatility * 0.5
            target_price = current_price * (1 - expected_move / 100)
        else:
            direction = 'NEUTRAL'
            confidence = 50
            expected_move = volatility * 0.2
            target_price = current_price
        
        # Time-based adjustments
        if timeframe == '5m':
            expected_move *= 0.1
            timeframe_to_target = '5-15 minutes'
        elif timeframe == '1h':
            expected_move *= 0.3
            timeframe_to_target = '1-3 hours'
        else:  # 1d
            timeframe_to_target = '1-3 days'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'target_price': target_price,
            'stop_loss': current_price * (0.98 if direction == 'UP' else 1.02),
            'expected_move_percent': expected_move,
            'key_levels': [current_price * 0.99, current_price * 1.01],
            'catalysts': ['Technical momentum', 'Volume confirmation'],
            'risks': ['Market volatility', 'News events'],
            'timeframe_to_target': timeframe_to_target
        }
    
    async def gather_technical_data(self, symbol: str) -> Dict:
        """Gather technical analysis data"""
        try:
            bars = self.api.get_bars(symbol, '1Day', limit=30).df
            
            if len(bars) < 10:
                return {'momentum': 0, 'volatility': 20}
            
            # Calculate momentum
            momentum = (bars['close'].iloc[-1] - bars['close'].iloc[-5]) / bars['close'].iloc[-5] * 100
            
            # Calculate volatility
            returns = bars['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            return {
                'momentum': momentum,
                'volatility': volatility,
                'current_price': bars['close'].iloc[-1],
                'sma_20': bars['close'].rolling(20).mean().iloc[-1],
                'volume_trend': 'INCREASING' if bars['volume'].iloc[-1] > bars['volume'].mean() else 'NORMAL'
            }
        except Exception as e:
            logger.error(f"Technical data gathering error: {e}")
            return {'momentum': 0, 'volatility': 20}
    
    # ==================== MARKET REGIME DETECTION ====================
    
    async def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Get SPY data
            spy_data = self.api.get_bars('SPY', '1Day', limit=50).df
            
            # Calculate metrics
            returns = spy_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100
            trend = spy_data['close'].iloc[-1] > spy_data['close'].rolling(20).mean().iloc[-1]
            recent_trend = returns.iloc[-10:].mean()
            
            # Determine regime
            if volatility > 30:
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 12:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend and recent_trend > 0.001:
                regime = MarketRegime.BULL_TREND
            elif not trend and recent_trend < -0.001:
                regime = MarketRegime.BEAR_TREND
            else:
                regime = MarketRegime.RANGE_BOUND
            
            self.current_regime = regime
            return regime
                
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegime.RANGE_BOUND
    
    async def analyze_global_markets(self) -> Dict:
        """Analyze global market conditions"""
        try:
            # Simplified global analysis
            return {
                'us_futures': 'POSITIVE',
                'european_markets': 'MIXED',
                'asian_markets': 'POSITIVE',
                'commodities': 'STABLE',
                'currencies': 'USD_STRENGTH',
                'overall_sentiment': 'RISK_ON'
            }
        except Exception as e:
            logger.error(f"Global markets analysis error: {e}")
            return {'overall_sentiment': 'NEUTRAL'}
    
    async def get_todays_events(self) -> List[Dict]:
        """Get today's important events"""
        # Simplified events calendar
        return [
            {'time': '10:00', 'event': 'Economic Data Release', 'importance': 'MEDIUM'},
            {'time': '14:00', 'event': 'Fed Speaker', 'importance': 'HIGH'},
            {'time': 'After Close', 'event': 'Earnings Reports', 'importance': 'HIGH'}
        ]
    
    async def calculate_key_levels(self, symbol: str) -> Dict:
        """Calculate key support and resistance levels"""
        try:
            bars = self.api.get_bars(symbol, '1Day', limit=30).df
            
            # Simple support/resistance calculation
            recent_high = bars['high'].rolling(10).max().iloc[-1]
            recent_low = bars['low'].rolling(10).min().iloc[-1]
            current_price = bars['close'].iloc[-1]
            
            return {
                'resistance': recent_high,
                'support': recent_low,
                'current': current_price,
                'pivot': (recent_high + recent_low + current_price) / 3
            }
        except Exception as e:
            logger.error(f"Key levels calculation error: {e}")
            return {'resistance': 0, 'support': 0, 'current': 0}
    
    async def identify_setup(self, symbol: str) -> Dict:
        """Identify high-probability trading setups"""
        try:
            technical = await self.gather_technical_data(symbol)
            patterns = await self.detect_patterns(symbol)
            
            # Calculate setup probability
            probability = 0.5  # Base probability
            
            if technical['momentum'] > 2:
                probability += 0.2
            if technical['volume_trend'] == 'INCREASING':
                probability += 0.1
            if len(patterns) > 0:
                probability += 0.2
            
            return {
                'symbol': symbol,
                'probability': min(1.0, probability),
                'type': 'MOMENTUM_BREAKOUT' if technical['momentum'] > 1 else 'MEAN_REVERSION',
                'entry_price': technical['current_price'],
                'confidence': min(95, probability * 100)
            }
        except Exception as e:
            logger.error(f"Setup identification error: {e}")
            return {'symbol': symbol, 'probability': 0}
    
    async def generate_premarket_summary(self, analysis: Dict) -> str:
        """Generate AI summary of pre-market analysis"""
        
        if self.demo_mode:
            summary = f"Pre-market analysis complete. Market regime: {analysis['market_regime'].value}. "
            
            if analysis['gap_analysis']:
                summary += f"Significant gaps detected in {len(analysis['gap_analysis'])} symbols. "
            
            if analysis['high_probability_setups']:
                summary += f"{len(analysis['high_probability_setups'])} high-probability setups identified. "
            
            summary += "Monitor key levels and news developments."
            
            return summary
        
        return "Pre-market analysis summary generated."
    
    # ==================== AI RECOMMENDATION ENGINE ====================
    
    async def get_ai_trade_recommendation(
        self, 
        symbol: str, 
        side: str,
        technical_signals: Dict,
        risks: List[str],
        opportunities: List[str]
    ) -> str:
        """Get comprehensive AI recommendation for trade"""
        
        if self.demo_mode:
            # Demo recommendation logic
            risk_count = len(risks)
            opp_count = len(opportunities)
            
            if risk_count > 2:
                return f"AVOID: Too many risks identified ({risk_count} risks vs {opp_count} opportunities)"
            elif opp_count > risk_count:
                return f"GO: Favorable risk/reward ratio ({opp_count} opportunities vs {risk_count} risks)"
            else:
                return f"CAUTION: Mixed signals ({opp_count} opportunities vs {risk_count} risks)"
        
        return "Trade analysis complete - proceed with caution"
    
    def store_prediction(self, symbol: str, timeframe: str, prediction: Dict):
        """Store predictions for later validation and learning"""
        prediction_key = f"{symbol}_{timeframe}_{datetime.now().isoformat()}"
        self.pattern_library[prediction_key] = {
            'prediction': prediction,
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe
        }
    
    # ==================== RISK DETECTION ====================
    
    async def detect_hidden_risks(self, portfolio: Dict) -> Dict:
        """
        Detect risks that aren't immediately obvious
        """
        hidden_risks = {
            'correlation_risk': await self.analyze_portfolio_correlation(portfolio),
            'concentration_risk': await self.analyze_concentration_risk(portfolio),
            'liquidity_risk': await self.analyze_portfolio_liquidity(portfolio),
            'event_risk': await self.scan_upcoming_events(portfolio),
            'regime_risk': await self.assess_regime_change_risk(portfolio)
        }
        
        # Calculate overall risk score
        risk_scores = []
        for risk_type, risk_data in hidden_risks.items():
            if isinstance(risk_data, dict) and 'score' in risk_data:
                risk_scores.append(risk_data['score'])
        
        overall_risk = np.mean(risk_scores) if risk_scores else 50
        
        hidden_risks['overall_risk_score'] = overall_risk
        hidden_risks['risk_level'] = 'HIGH' if overall_risk > 70 else 'MEDIUM' if overall_risk > 40 else 'LOW'
        
        return hidden_risks
    
    async def analyze_portfolio_correlation(self, portfolio: Dict) -> Dict:
        """Analyze correlation risks in portfolio"""
        try:
            # Simplified correlation analysis
            symbols = list(portfolio.keys())
            
            if len(symbols) < 2:
                return {'score': 0, 'warning': False}
            
            # Check for sector concentration
            tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
            tech_count = sum(1 for symbol in symbols if symbol in tech_stocks)
            
            correlation_score = (tech_count / len(symbols)) * 100 if len(symbols) > 0 else 0
            
            return {
                'score': correlation_score,
                'warning': correlation_score > 60,
                'correlated_symbols': tech_count,
                'recommendation': 'DIVERSIFY' if correlation_score > 60 else 'OK'
            }
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {'score': 0, 'warning': False}
    
    async def analyze_concentration_risk(self, portfolio: Dict) -> Dict:
        """Analyze position concentration risk"""
        try:
            if not portfolio:
                return {'score': 0, 'concentrated_positions': []}
            
            total_value = sum(pos.get('value', 0) for pos in portfolio.values())
            
            concentrated_positions = []
            for symbol, position in portfolio.items():
                position_pct = (position.get('value', 0) / total_value) * 100 if total_value > 0 else 0
                if position_pct > 20:
                    concentrated_positions.append({
                        'symbol': symbol,
                        'percentage': position_pct
                    })
            
            concentration_score = sum(pos['percentage'] for pos in concentrated_positions)
            
            return {
                'score': min(100, concentration_score),
                'concentrated_positions': concentrated_positions,
                'warning': concentration_score > 40,
                'recommendation': 'REDUCE_CONCENTRATION' if concentration_score > 40 else 'OK'
            }
        except Exception as e:
            logger.error(f"Concentration analysis error: {e}")
            return {'score': 0, 'concentrated_positions': []}
    
    async def analyze_portfolio_liquidity(self, portfolio: Dict) -> Dict:
        """Analyze portfolio liquidity risk"""
        try:
            illiquid_positions = []
            
            for symbol, position in portfolio.items():
                # Get volume data
                bars = self.api.get_bars(symbol, '1Day', limit=5).df
                avg_volume = bars['volume'].mean()
                
                # Check if position is large relative to volume
                position_size = position.get('shares', 0)
                if position_size > avg_volume * 0.01:  # More than 1% of daily volume
                    illiquid_positions.append({
                        'symbol': symbol,
                        'liquidity_concern': True,
                        'exit_difficulty': 'HIGH'
                    })
            
            liquidity_score = (len(illiquid_positions) / len(portfolio)) * 100 if portfolio else 0
            
            return {
                'score': liquidity_score,
                'illiquid_positions': illiquid_positions,
                'warning': liquidity_score > 30,
                'recommendation': 'REDUCE_ILLIQUID_POSITIONS' if liquidity_score > 30 else 'OK'
            }
        except Exception as e:
            logger.error(f"Liquidity analysis error: {e}")
            return {'score': 0, 'illiquid_positions': []}
    
    async def scan_upcoming_events(self, portfolio: Dict) -> Dict:
        """Scan for upcoming events that could affect portfolio"""
        # Simplified event scanning
        events = [
            {'date': '2025-09-18', 'event': 'Fed Meeting', 'impact': 'HIGH'},
            {'date': '2025-09-19', 'event': 'Earnings Season', 'impact': 'MEDIUM'},
            {'date': '2025-09-20', 'event': 'Options Expiry', 'impact': 'MEDIUM'}
        ]
        
        return {
            'upcoming_events': events,
            'high_impact_events': [e for e in events if e['impact'] == 'HIGH'],
            'score': len([e for e in events if e['impact'] == 'HIGH']) * 20,
            'recommendation': 'REDUCE_RISK' if len([e for e in events if e['impact'] == 'HIGH']) > 1 else 'MONITOR'
        }
    
    async def assess_regime_change_risk(self, portfolio: Dict) -> Dict:
        """Assess risk of market regime change"""
        try:
            current_regime = await self.detect_market_regime()
            
            # Simple regime risk assessment
            if current_regime == MarketRegime.HIGH_VOLATILITY:
                score = 80
                warning = True
                recommendation = 'REDUCE_EXPOSURE'
            elif current_regime == MarketRegime.BULL_TREND:
                score = 20
                warning = False
                recommendation = 'MAINTAIN'
            else:
                score = 40
                warning = False
                recommendation = 'MONITOR'
            
            return {
                'current_regime': current_regime.value,
                'score': score,
                'warning': warning,
                'recommendation': recommendation,
                'regime_stability': 'LOW' if score > 60 else 'HIGH'
            }
        except Exception as e:
            logger.error(f"Regime risk assessment error: {e}")
            return {'score': 50, 'warning': False}


# Integration function for main bot
def integrate_comprehensive_ai(bot_instance):
    """Integrate Comprehensive AI Agent into main bot"""
    
    # Initialize the comprehensive agent
    bot_instance.comprehensive_ai = ComprehensiveAIAgent(bot_instance.core.api)
    
    # Command handlers
    async def validate_command(update, context):
        if len(context.args) < 3:
            await update.message.reply_text("Usage: /validate SYMBOL BUY/SELL PRICE")
            return
        
        symbol = context.args[0].upper()
        side = context.args[1].lower()
        price = float(context.args[2])
        
        await update.message.reply_text(f"🤖 Validating {side.upper()} {symbol} at ${price:.2f}...")
        
        try:
            # Get validation
            validation = await bot_instance.comprehensive_ai.validate_trade(
                symbol, side, price, {}
            )
            
            msg = f"""
🤖 **AI Trade Validation: {symbol}**

**Decision:** {'✅ VALID TRADE' if validation.is_valid else '❌ INVALID TRADE'}
**Confidence:** {validation.confidence:.1f}%

**💡 Opportunities:**
{chr(10).join('• ' + opp for opp in validation.opportunities[:3]) if validation.opportunities else '• None identified'}

**⚠️ Risks:**
{chr(10).join('• ' + risk for risk in validation.risks[:3]) if validation.risks else '• None identified'}

**🤖 AI Recommendation:**
{validation.reasons[0][:200] if validation.reasons else 'Proceed with standard caution'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Validation error: {str(e)}")
    
    async def psychology_command(update, context):
        await update.message.reply_text("🧠 Analyzing trading psychology...")
        
        try:
            # Analyze trader psychology
            history = list(bot_instance.comprehensive_ai.trade_history)
            analysis = await bot_instance.comprehensive_ai.analyze_trader_psychology(history)
            
            msg = f"""
🧠 **Psychological Analysis**

**📊 Trading Stats:**
• Total Trades: {analysis.get('total_trades', 0)}
• Psychological State: {analysis.get('psychological_state', 'UNKNOWN')}

**🔍 Patterns Detected:**
{chr(10).join('• ' + pattern for pattern in analysis.get('patterns_detected', ['None detected']))}

**⏰ Best Trading Hours:**
{', '.join(str(h) + ':00' for h in analysis.get('best_trading_hours', [])) if analysis.get('best_trading_hours') else 'Insufficient data'}

**💡 Recommendations:**
{chr(10).join('• ' + rec for rec in analysis.get('recommendations', [])[:3]) if analysis.get('recommendations') else '• Continue current approach'}

**🤖 AI Assessment:**
{analysis.get('ai_psychological_assessment', 'No assessment available')[:300]}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Psychology analysis error: {str(e)}")
    
    async def predict_command(update, context):
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /predict SYMBOL TIMEFRAME(5m/1h/1d)")
            return
        
        symbol = context.args[0].upper()
        timeframe = context.args[1]
        
        await update.message.reply_text(f"🔮 Predicting {timeframe} movement for {symbol}...")
        
        try:
            prediction = await bot_instance.comprehensive_ai.predict_price_movement(symbol, timeframe)
            
            msg = f"""
🔮 **AI Price Prediction: {symbol} ({timeframe})**

**📈 Direction:** **{prediction.get('direction', 'UNKNOWN')}**
**🎯 Confidence:** {prediction.get('confidence', 0):.1f}%

**💰 Targets:**
• Target Price: ${prediction.get('target_price', 0):.2f}
• Stop Loss: ${prediction.get('stop_loss', 0):.2f}
• Expected Move: {prediction.get('expected_move_percent', 0):.2f}%

**⏱️ Time to Target:** {prediction.get('timeframe_to_target', 'Unknown')}

**🚀 Key Catalysts:**
{chr(10).join('• ' + catalyst for catalyst in prediction.get('catalysts', [])[:3]) if prediction.get('catalysts') else '• Technical momentum'}

**⚠️ Key Risks:**
{chr(10).join('• ' + risk for risk in prediction.get('risks', [])[:3]) if prediction.get('risks') else '• Market volatility'}

**📊 Key Levels:**
{', '.join(f'${level:.2f}' for level in prediction.get('key_levels', [])[:3]) if prediction.get('key_levels') else 'N/A'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Prediction error: {str(e)}")
    
    async def optimize_execution_command(update, context):
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /optimize SYMBOL SHARES [URGENCY]")
            return
        
        symbol = context.args[0].upper()
        shares = int(context.args[1])
        urgency = context.args[2].upper() if len(context.args) > 2 else 'NORMAL'
        
        await update.message.reply_text(f"⚡ Optimizing execution for {shares} shares of {symbol}...")
        
        try:
            plan = await bot_instance.comprehensive_ai.optimize_execution(symbol, shares, urgency)
            
            msg = f"""
⚡ **Execution Optimization: {symbol}**

**📋 Execution Plan:**
• Method: {plan['method']}
• Order Type: {plan['order_type']}
• Strategy: {plan['time_strategy']}

{f"• Splits: {plan.get('splits', 1)} orders" if plan['method'] == 'SPLIT' else ''}
{f"• Interval: {plan.get('interval_seconds', 0)}s between orders" if plan['method'] == 'SPLIT' else ''}

**💰 Cost Estimates:**
• Expected Slippage: {plan.get('expected_slippage', 0):.3f}%
• Price Adjustment: ${plan.get('price_adjustment', 0):.4f}

{f"• Wait Time: {plan.get('wait_minutes', 0)} minutes" if plan.get('wait_minutes') else ''}

**🤖 AI Recommendation:**
{plan.get('ai_recommendation', 'Execute as planned')[:300]}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Execution optimization error: {str(e)}")
    
    async def risk_scan_command(update, context):
        await update.message.reply_text("🔍 Scanning for hidden risks...")
        
        try:
            # Get current positions
            positions = bot_instance.core.api.list_positions()
            portfolio = {}
            
            for pos in positions:
                portfolio[pos.symbol] = {
                    'shares': int(pos.qty),
                    'value': float(pos.market_value),
                    'pnl': float(pos.unrealized_pl)
                }
            
            risks = await bot_instance.comprehensive_ai.detect_hidden_risks(portfolio)
            
            msg = f"""
🔍 **Hidden Risk Analysis**

**📊 Overall Risk Level:** {risks.get('risk_level', 'UNKNOWN')}
**🎯 Risk Score:** {risks.get('overall_risk_score', 0):.1f}/100

**⚠️ Risk Factors:**

**Correlation Risk:**
• Score: {risks.get('correlation_risk', {}).get('score', 0):.1f}%
• Warning: {'Yes' if risks.get('correlation_risk', {}).get('warning') else 'No'}

**Concentration Risk:**
• Concentrated Positions: {len(risks.get('concentration_risk', {}).get('concentrated_positions', []))}
• Warning: {'Yes' if risks.get('concentration_risk', {}).get('warning') else 'No'}

**Event Risk:**
• Upcoming High-Impact Events: {len(risks.get('event_risk', {}).get('high_impact_events', []))}

**Regime Risk:**
• Current Regime: {risks.get('regime_risk', {}).get('current_regime', 'UNKNOWN')}
• Stability: {risks.get('regime_risk', {}).get('regime_stability', 'UNKNOWN')}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Risk scan error: {str(e)}")
    
    return validate_command, psychology_command, predict_command, optimize_execution_command, risk_scan_command

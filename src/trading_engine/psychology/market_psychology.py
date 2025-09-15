"""
Market Psychology Engine
Analyzes market sentiment, crowd behavior, and manipulation patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MarketPsychologyEngine:
    """
    Detects and analyzes market psychology patterns
    Including fear/greed, manipulation, and crowd behavior
    """
    
    def __init__(self, market_data_manager, config: Dict[str, Any]):
        self.market_data = market_data_manager
        self.config = config
        
        # Psychology parameters
        self.fear_threshold = config.get('fear_greed_extreme_threshold', 20)
        self.greed_threshold = config.get('greed_extreme_threshold', 80)
        
        # Current state
        self.fear_greed_index = 50
        self.market_sentiment = "neutral"
        self.manipulation_detected = False
        
        # Tracking
        self.sentiment_history = []
        self.manipulation_events = []
        self.psychology_metrics = defaultdict(dict)
        
        # Wyckoff phases
        self.wyckoff_phase = None
        self.accumulation_score = 0
        self.distribution_score = 0
        
        logger.info("Market Psychology Engine initialized")
    
    async def get_market_psychology(self) -> Dict:
        """Get comprehensive market psychology analysis"""
        psychology = {
            'fear_greed_index': await self._calculate_fear_greed_index(),
            'sentiment': await self._analyze_sentiment(),
            'crowd_behavior': await self._analyze_crowd_behavior(),
            'smart_money': await self._track_smart_money(),
            'wyckoff_phase': await self._identify_wyckoff_phase(),
            'manipulation_risk': await self._assess_manipulation_risk()
        }
        
        return psychology
    
    async def _calculate_fear_greed_index(self) -> float:
        """Calculate custom fear/greed index"""
        components = {}
        
        # 1. Price Momentum (25%)
        spy_data = await self.market_data.get_historical_data('SPY', '1d', 125)
        if spy_data is not None and len(spy_data) > 0:
            current_price = spy_data['close'].iloc[-1]
            ma_125 = spy_data['close'].rolling(125).mean().iloc[-1]
            
            momentum_score = 50 + ((current_price / ma_125 - 1) * 100)
            components['momentum'] = np.clip(momentum_score, 0, 100)
        
        # 2. Market Volatility (25%)
        vix = await self._get_vix_level()
        if vix:
            # Inverse VIX - high VIX = fear
            vol_score = 100 - (vix / 50 * 100)
            components['volatility'] = np.clip(vol_score, 0, 100)
        
        # 3. Market Breadth (20%)
        breadth = await self._calculate_market_breadth()
        components['breadth'] = breadth * 100
        
        # 4. Put/Call Ratio (15%)
        pcr = await self._get_put_call_ratio()
        if pcr:
            # Inverse PCR - high PCR = fear
            pcr_score = 100 - (pcr * 50)
            components['pcr'] = np.clip(pcr_score, 0, 100)
        
        # 5. Safe Haven Demand (15%)
        safe_haven = await self._analyze_safe_haven_flows()
        components['safe_haven'] = 100 - safe_haven  # Inverse - high safe haven = fear
        
        # Calculate weighted average
        weights = {
            'momentum': 0.25,
            'volatility': 0.25,
            'breadth': 0.20,
            'pcr': 0.15,
            'safe_haven': 0.15
        }
        
        fear_greed = 0
        total_weight = 0
        
        for component, value in components.items():
            if component in weights:
                fear_greed += value * weights[component]
                total_weight += weights[component]
        
        if total_weight > 0:
            fear_greed = fear_greed / total_weight
        
        self.fear_greed_index = fear_greed
        
        # Determine sentiment
        if fear_greed < self.fear_threshold:
            self.market_sentiment = "extreme_fear"
        elif fear_greed < 40:
            self.market_sentiment = "fear"
        elif fear_greed > self.greed_threshold:
            self.market_sentiment = "extreme_greed"
        elif fear_greed > 60:
            self.market_sentiment = "greed"
        else:
            self.market_sentiment = "neutral"
        
        logger.info(f"Fear/Greed Index: {fear_greed:.1f} ({self.market_sentiment})")
        
        return fear_greed
    
    async def _analyze_sentiment(self) -> Dict:
        """Analyze overall market sentiment"""
        sentiment = {
            'overall': self.market_sentiment,
            'fear_greed_index': self.fear_greed_index,
            'retail_sentiment': await self._analyze_retail_sentiment(),
            'institutional_sentiment': await self._analyze_institutional_sentiment(),
            'options_sentiment': await self._analyze_options_sentiment()
        }
        
        return sentiment
    
    async def _analyze_crowd_behavior(self) -> Dict:
        """Analyze crowd psychology patterns"""
        patterns = {
            'herding': await self._detect_herding(),
            'panic_selling': await self._detect_panic_selling(),
            'euphoria': await self._detect_euphoria(),
            'capitulation': await self._detect_capitulation()
        }
        
        return patterns
    
    async def _track_smart_money(self) -> Dict:
        """Track smart money movements"""
        smart_money = {
            'accumulation': await self._detect_accumulation(),
            'distribution': await self._detect_distribution(),
            'institutional_flows': await self._track_institutional_flows(),
            'dark_pool_activity': await self._analyze_dark_pools()
        }
        
        return smart_money
    
    async def _identify_wyckoff_phase(self) -> str:
        """Identify current Wyckoff phase"""
        # Get market data
        spy_data = await self.market_data.get_historical_data('SPY', '1d', 60)
        
        if spy_data is None or len(spy_data) < 60:
            return "unknown"
        
        close = spy_data['close'].values
        volume = spy_data['volume'].values
        high = spy_data['high'].values
        low = spy_data['low'].values
        
        # Analyze price and volume patterns
        phases = {
            'accumulation': 0,
            'markup': 0,
            'distribution': 0,
            'markdown': 0
        }
        
        # Check for accumulation patterns
        if self._check_accumulation_patterns(close, volume, high, low):
            phases['accumulation'] += 1
        
        # Check for markup patterns
        if self._check_markup_patterns(close, volume):
            phases['markup'] += 1
        
        # Check for distribution patterns
        if self._check_distribution_patterns(close, volume, high, low):
            phases['distribution'] += 1
        
        # Check for markdown patterns
        if self._check_markdown_patterns(close, volume):
            phases['markdown'] += 1
        
        # Determine phase
        self.wyckoff_phase = max(phases, key=phases.get)
        
        return self.wyckoff_phase
    
    async def detect_manipulation(self) -> Optional[Dict]:
        """Detect market manipulation patterns"""
        manipulation_patterns = {}
        
        # Stop hunting
        stop_hunt = await self._detect_stop_hunting()
        if stop_hunt:
            manipulation_patterns['stop_hunting'] = stop_hunt
        
        # Spoofing
        spoofing = await self._detect_spoofing()
        if spoofing:
            manipulation_patterns['spoofing'] = spoofing
        
        # Pump and dump
        pump_dump = await self._detect_pump_and_dump()
        if pump_dump:
            manipulation_patterns['pump_and_dump'] = pump_dump
        
        # Bear raid
        bear_raid = await self._detect_bear_raid()
        if bear_raid:
            manipulation_patterns['bear_raid'] = bear_raid
        
        if manipulation_patterns:
            self.manipulation_detected = True
            self.manipulation_events.append({
                'timestamp': datetime.utcnow(),
                'patterns': manipulation_patterns
            })
            
            return manipulation_patterns
        
        return None
    
    async def analyze_signal(self, signal) -> float:
        """Analyze signal from psychology perspective"""
        psychology_score = 0.5
        
        # Check if signal aligns with sentiment
        if self.market_sentiment == "extreme_fear" and signal.action == "BUY":
            psychology_score += 0.2  # Contrarian
        elif self.market_sentiment == "extreme_greed" and signal.action == "SELL":
            psychology_score += 0.2  # Contrarian
        
        # Check Wyckoff alignment
        if self.wyckoff_phase == "accumulation" and signal.action == "BUY":
            psychology_score += 0.15
        elif self.wyckoff_phase == "distribution" and signal.action == "SELL":
            psychology_score += 0.15
        
        # Check for manipulation
        if not self.manipulation_detected:
            psychology_score += 0.1
        
        return min(1.0, psychology_score)
    
    # Helper methods
    
    async def _get_vix_level(self) -> Optional[float]:
        """Get VIX level"""
        try:
            vix_data = await self.market_data.get_quote('VIX')
            if vix_data:
                return vix_data.get('last', 20)
        except:
            pass
        return 20  # Default
    
    async def _calculate_market_breadth(self) -> float:
        """Calculate market breadth"""
        # Simplified - would use advance/decline line in production
        return 0.5
    
    async def _get_put_call_ratio(self) -> Optional[float]:
        """Get put/call ratio"""
        # Would fetch from options data provider
        return 1.0  # Neutral default
    
    async def _analyze_safe_haven_flows(self) -> float:
        """Analyze flows to safe haven assets"""
        # Check gold, bonds, dollar strength
        return 50  # Neutral default
    
    async def _analyze_retail_sentiment(self) -> str:
        """Analyze retail trader sentiment"""
        # Would analyze social media, retail broker data
        return "neutral"
    
    async def _analyze_institutional_sentiment(self) -> str:
        """Analyze institutional sentiment"""
        # Would analyze 13F filings, large trades
        return "neutral"
    
    async def _analyze_options_sentiment(self) -> str:
        """Analyze options market sentiment"""
        # Would analyze options flow, skew, term structure
        return "neutral"
    
    async def _detect_herding(self) -> bool:
        """Detect herding behavior"""
        # Check for coordinated movements
        return False
    
    async def _detect_panic_selling(self) -> bool:
        """Detect panic selling"""
        # High volume + sharp decline + high VIX
        return self.market_sentiment == "extreme_fear"
    
    async def _detect_euphoria(self) -> bool:
        """Detect market euphoria"""
        # Low volatility + persistent buying + extreme greed
        return self.market_sentiment == "extreme_greed"
    
    async def _detect_capitulation(self) -> bool:
        """Detect capitulation"""
        # Extreme volume + extreme decline + sentiment washout
        return False
    
    async def _detect_accumulation(self) -> float:
        """Detect smart money accumulation"""
        # Volume analysis + price stability
        return self.accumulation_score
    
    async def _detect_distribution(self) -> float:
        """Detect smart money distribution"""
        # Volume analysis + price weakness
        return self.distribution_score
    
    async def _track_institutional_flows(self) -> Dict:
        """Track institutional money flows"""
        return {}
    
    async def _analyze_dark_pools(self) -> Dict:
        """Analyze dark pool activity"""
        return {}
    
    async def _assess_manipulation_risk(self) -> float:
        """Assess overall manipulation risk"""
        if self.manipulation_detected:
            return 0.8
        return 0.2
    
    async def _detect_stop_hunting(self) -> Optional[Dict]:
        """Detect stop hunting patterns"""
        # Quick spike beyond key levels, then reversal
        return None
    
    async def _detect_spoofing(self) -> Optional[Dict]:
        """Detect spoofing in order book"""
        # Large orders that disappear
        return None
    
    async def _detect_pump_and_dump(self) -> Optional[Dict]:
        """Detect pump and dump schemes"""
        # Unusual volume + price spike + social media activity
        return None
    
    async def _detect_bear_raid(self) -> Optional[Dict]:
        """Detect bear raid attacks"""
        # Coordinated selling + negative news
        return None
    
    def _check_accumulation_patterns(self, close, volume, high, low) -> bool:
        """Check for Wyckoff accumulation patterns"""
        # Simplified check - would be more complex in production
        recent_range = np.max(high[-20:]) - np.min(low[-20:])
        previous_range = np.max(high[-40:-20]) - np.min(low[-40:-20])
        
        # Narrowing range with steady volume = accumulation
        if recent_range < previous_range * 0.8 and np.mean(volume[-20:]) > np.mean(volume[-40:-20]) * 0.9:
            self.accumulation_score = 0.7
            return True
        
        return False
    
    def _check_markup_patterns(self, close, volume) -> bool:
        """Check for markup phase patterns"""
        # Rising prices with increasing volume
        price_trend = np.polyfit(range(len(close[-20:])), close[-20:], 1)[0]
        volume_trend = np.polyfit(range(len(volume[-20:])), volume[-20:], 1)[0]
        
        return price_trend > 0 and volume_trend > 0
    
    def _check_distribution_patterns(self, close, volume, high, low) -> bool:
        """Check for distribution patterns"""
        # High volume with limited price progress
        recent_range = np.max(high[-20:]) - np.min(low[-20:])
        
        if np.mean(volume[-20:]) > np.mean(volume[-40:-20]) * 1.3 and recent_range < np.mean(close[-20:]) * 0.05:
            self.distribution_score = 0.7
            return True
        
        return False
    
    def _check_markdown_patterns(self, close, volume) -> bool:
        """Check for markdown phase patterns"""
        # Declining prices with increasing volume
        price_trend = np.polyfit(range(len(close[-20:])), close[-20:], 1)[0]
        volume_trend = np.polyfit(range(len(volume[-20:])), volume[-20:], 1)[0]
        
        return price_trend < 0 and volume_trend > 0

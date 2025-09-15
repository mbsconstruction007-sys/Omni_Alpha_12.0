"""
Market Regime Detector
Identifies current market conditions using multiple methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import talib
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Models,
    volatility clustering, and trend analysis
    """
    
    def __init__(self, market_data_manager, config: Dict[str, Any]):
        self.market_data = market_data_manager
        self.config = config
        
        # Regime parameters
        self.lookback_period = config.get('regime_lookback_periods', 60)
        self.update_frequency = config.get('regime_update_frequency', 300)
        
        # Thresholds
        self.bull_threshold = config.get('bull_regime_threshold', 0.6)
        self.bear_threshold = config.get('bear_regime_threshold', -0.6)
        self.high_vol_threshold = config.get('high_volatility_threshold', 30)
        self.trend_strength_threshold = config.get('trend_strength_threshold', 25)
        
        # Current state
        self.current_regime = "neutral"
        self.regime_confidence = 0.5
        self.regime_history = []
        
        # Models
        self.hmm_model = None
        self.volatility_model = None
        self.scaler = StandardScaler()
        
        # Regime characteristics
        self.regime_characteristics = {
            'bull': {
                'volatility': 'low',
                'trend': 'up',
                'momentum': 'positive',
                'breadth': 'expanding'
            },
            'bear': {
                'volatility': 'high',
                'trend': 'down',
                'momentum': 'negative',
                'breadth': 'contracting'
            },
            'neutral': {
                'volatility': 'medium',
                'trend': 'sideways',
                'momentum': 'mixed',
                'breadth': 'stable'
            },
            'volatile': {
                'volatility': 'very_high',
                'trend': 'choppy',
                'momentum': 'whipsaw',
                'breadth': 'divergent'
            }
        }
        
        logger.info("Regime Detector initialized")
    
    async def detect_regime(self) -> str:
        """Main regime detection method"""
        try:
            # Get market data
            spy_data = await self._get_market_data('SPY')
            
            if spy_data is None or len(spy_data) < self.lookback_period:
                return self.current_regime
            
            # Method 1: Trend-based regime
            trend_regime = await self._detect_trend_regime(spy_data)
            
            # Method 2: Volatility-based regime
            vol_regime = await self._detect_volatility_regime(spy_data)
            
            # Method 3: HMM-based regime
            hmm_regime = await self._detect_hmm_regime(spy_data)
            
            # Method 4: Market breadth regime
            breadth_regime = await self._detect_breadth_regime()
            
            # Method 5: Momentum regime
            momentum_regime = await self._detect_momentum_regime(spy_data)
            
            # Combine all methods
            final_regime = self._combine_regime_signals({
                'trend': trend_regime,
                'volatility': vol_regime,
                'hmm': hmm_regime,
                'breadth': breadth_regime,
                'momentum': momentum_regime
            })
            
            # Update state
            self._update_regime_state(final_regime)
            
            return self.current_regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self.current_regime
    
    async def _detect_trend_regime(self, data: pd.DataFrame) -> Dict:
        """Detect regime based on trend analysis"""
        close = data['close'].values
        
        # Calculate moving averages
        sma_20 = talib.SMA(close, timeperiod=20)[-1]
        sma_50 = talib.SMA(close, timeperiod=50)[-1]
        sma_200 = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else sma_50
        
        # ADX for trend strength
        high = data['high'].values
        low = data['low'].values
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]
        
        # Linear regression slope
        x = np.arange(len(close[-20:]))
        slope = np.polyfit(x, close[-20:], 1)[0]
        
        # Determine regime
        regime = "neutral"
        confidence = 0.5
        
        if close[-1] > sma_20 > sma_50 > sma_200:
            regime = "bull"
            confidence = min(0.9, adx / 40)
        elif close[-1] < sma_20 < sma_50 < sma_200:
            regime = "bear"
            confidence = min(0.9, adx / 40)
        elif adx < 20:
            regime = "neutral"
            confidence = 0.6
        else:
            # Mixed signals
            if slope > 0:
                regime = "bull"
                confidence = 0.4
            else:
                regime = "bear"
                confidence = 0.4
        
        return {
            'regime': regime,
            'confidence': confidence,
            'adx': adx,
            'slope': slope
        }
    
    async def _detect_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """Detect regime based on volatility patterns"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Historical volatility
        returns = np.diff(np.log(close))
        hvol = np.std(returns) * np.sqrt(252) * 100
        
        # ATR-based volatility
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        atr_pct = (atr / close[-1]) * 100
        
        # Bollinger Band width
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        bb_width = ((upper[-1] - lower[-1]) / middle[-1]) * 100
        
        # VIX proxy (if available)
        vix_level = await self._get_vix_level()
        
        # Determine volatility regime
        if vix_level:
            if vix_level > 30:
                vol_regime = "high_volatility"
            elif vix_level > 20:
                vol_regime = "moderate_volatility"
            else:
                vol_regime = "low_volatility"
        else:
            if hvol > 30 or atr_pct > 3:
                vol_regime = "high_volatility"
            elif hvol > 20 or atr_pct > 2:
                vol_regime = "moderate_volatility"
            else:
                vol_regime = "low_volatility"
        
        # Map to regime
        regime_map = {
            'low_volatility': 'bull',
            'moderate_volatility': 'neutral',
            'high_volatility': 'volatile'
        }
        
        return {
            'regime': regime_map[vol_regime],
            'confidence': 0.7,
            'hvol': hvol,
            'atr_pct': atr_pct,
            'bb_width': bb_width,
            'vix': vix_level
        }
    
    async def _detect_hmm_regime(self, data: pd.DataFrame) -> Dict:
        """Detect regime using Hidden Markov Model"""
        try:
            # Prepare features
            close = data['close'].values
            returns = np.diff(np.log(close))
            
            # Calculate features for HMM
            features = []
            
            for i in range(20, len(returns)):
                feature_vector = [
                    returns[i],  # Current return
                    np.mean(returns[i-5:i]),  # 5-day mean return
                    np.std(returns[i-5:i]),  # 5-day volatility
                    np.mean(returns[i-20:i]),  # 20-day mean return
                    np.std(returns[i-20:i])  # 20-day volatility
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Train or update HMM
            if self.hmm_model is None:
                self.hmm_model = GaussianMixture(
                    n_components=4,  # 4 regimes
                    covariance_type='full',
                    max_iter=100
                )
                self.hmm_model.fit(self.scaler.fit_transform(features))
            
            # Predict current regime
            current_features = features[-1].reshape(1, -1)
            scaled_features = self.scaler.transform(current_features)
            
            regime_probs = self.hmm_model.predict_proba(scaled_features)[0]
            regime_idx = np.argmax(regime_probs)
            
            # Map to regime names
            regime_map = {
                0: 'bull',
                1: 'bear',
                2: 'neutral',
                3: 'volatile'
            }
            
            return {
                'regime': regime_map[regime_idx],
                'confidence': regime_probs[regime_idx],
                'probabilities': regime_probs
            }
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            return {
                'regime': 'neutral',
                'confidence': 0.3
            }
    
    async def _detect_breadth_regime(self) -> Dict:
        """Detect regime based on market breadth"""
        try:
            # Get breadth indicators (simplified - would use advance/decline in production)
            # For now, use sector performance
            sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU']
            
            advancing = 0
            declining = 0
            
            for sector in sectors:
                data = await self._get_market_data(sector)
                if data is not None and len(data) > 1:
                    if data['close'].iloc[-1] > data['close'].iloc[-2]:
                        advancing += 1
                    else:
                        declining += 1
            
            # Calculate breadth
            if advancing + declining > 0:
                breadth_ratio = advancing / (advancing + declining)
            else:
                breadth_ratio = 0.5
            
            # Determine regime
            if breadth_ratio > 0.7:
                regime = 'bull'
                confidence = breadth_ratio
            elif breadth_ratio < 0.3:
                regime = 'bear'
                confidence = 1 - breadth_ratio
            else:
                regime = 'neutral'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'advancing': advancing,
                'declining': declining,
                'breadth_ratio': breadth_ratio
            }
            
        except Exception as e:
            logger.error(f"Breadth regime detection failed: {e}")
            return {
                'regime': 'neutral',
                'confidence': 0.3
            }
    
    async def _detect_momentum_regime(self, data: pd.DataFrame) -> Dict:
        """Detect regime based on momentum indicators"""
        close = data['close'].values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(close)
        macd_histogram = hist[-1]
        
        # Rate of Change
        roc_10 = talib.ROC(close, timeperiod=10)[-1]
        roc_20 = talib.ROC(close, timeperiod=20)[-1]
        
        # Momentum score
        momentum_score = 0
        
        if rsi > 50:
            momentum_score += 0.25
        if rsi > 60:
            momentum_score += 0.25
            
        if macd_histogram > 0:
            momentum_score += 0.25
            
        if roc_10 > 0 and roc_20 > 0:
            momentum_score += 0.25
        
        # Determine regime
        if momentum_score >= 0.75:
            regime = 'bull'
        elif momentum_score <= 0.25:
            regime = 'bear'
        else:
            regime = 'neutral'
        
        return {
            'regime': regime,
            'confidence': abs(momentum_score - 0.5) * 2,  # Convert to confidence
            'rsi': rsi,
            'macd_histogram': macd_histogram,
            'momentum_score': momentum_score
        }
    
    def _combine_regime_signals(self, signals: Dict) -> str:
        """Combine multiple regime signals into final determination"""
        # Weight each method
        weights = {
            'trend': 0.25,
            'volatility': 0.20,
            'hmm': 0.25,
            'breadth': 0.15,
            'momentum': 0.15
        }
        
        # Score each regime
        regime_scores = {
            'bull': 0,
            'bear': 0,
            'neutral': 0,
            'volatile': 0
        }
        
        # Accumulate weighted scores
        for method, signal in signals.items():
            if isinstance(signal, dict) and 'regime' in signal:
                regime = signal['regime']
                confidence = signal.get('confidence', 0.5)
                weight = weights.get(method, 0.2)
                
                regime_scores[regime] += weight * confidence
        
        # Normalize scores
        total_score = sum(regime_scores.values())
        if total_score > 0:
            for regime in regime_scores:
                regime_scores[regime] /= total_score
        
        # Get regime with highest score
        final_regime = max(regime_scores, key=regime_scores.get)
        self.regime_confidence = regime_scores[final_regime]
        
        # Log regime probabilities
        logger.info(f"Regime probabilities: {regime_scores}")
        
        return final_regime
    
    def _update_regime_state(self, new_regime: str):
        """Update regime state and history"""
        if new_regime != self.current_regime:
            logger.info(f"Regime change: {self.current_regime} -> {new_regime}")
            
            # Add to history
            self.regime_history.append({
                'from': self.current_regime,
                'to': new_regime,
                'timestamp': datetime.utcnow(),
                'confidence': self.regime_confidence
            })
            
            # Update current regime
            self.current_regime = new_regime
        
        # Limit history size
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            data = await self.market_data.get_historical_data(
                symbol,
                interval='1d',
                limit=self.lookback_period + 50
            )
            return data
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _get_vix_level(self) -> Optional[float]:
        """Get current VIX level"""
        try:
            vix_data = await self.market_data.get_quote('VIX')
            if vix_data:
                return vix_data.get('last', None)
        except:
            pass
        return None
    
    async def update(self, market_update: Dict):
        """Update regime detector with new market data"""
        # This would be called periodically or on market updates
        pass
    
    def get_regime_characteristics(self) -> Dict:
        """Get characteristics of current regime"""
        return self.regime_characteristics.get(
            self.current_regime,
            self.regime_characteristics['neutral']
        )
    
    def get_regime_history(self) -> List:
        """Get regime change history"""
        return self.regime_history
    
    def adjust_parameters_for_regime(self, parameters: Dict) -> Dict:
        """Adjust strategy parameters based on current regime"""
        adjustments = {
            'bull': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.2,
                'confidence_threshold_multiplier': 0.9
            },
            'bear': {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.9,
                'confidence_threshold_multiplier': 1.2
            },
            'volatile': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.3,
                'confidence_threshold_multiplier': 1.3
            },
            'neutral': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'confidence_threshold_multiplier': 1.0
            }
        }
        
        regime_adjustments = adjustments.get(self.current_regime, adjustments['neutral'])
        
        # Apply adjustments
        adjusted_params = parameters.copy()
        for key, multiplier in regime_adjustments.items():
            param_key = key.replace('_multiplier', '')
            if param_key in adjusted_params:
                adjusted_params[param_key] *= multiplier
        
        return adjusted_params

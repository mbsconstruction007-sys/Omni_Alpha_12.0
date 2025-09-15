"""
Signal Processor - Advanced signal filtering and validation
Implements institutional-grade signal processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
import logging

from ..strategies.base_strategy import Signal

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Processes, filters, and validates trading signals
    Removes noise and ensures only high-quality signals pass through
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Processing parameters
        self.min_signal_strength = config.get('min_signal_strength', 50)
        self.confirmation_sources = config.get('signal_confirmation_sources', 3)
        self.correlation_threshold = config.get('correlation_filter_threshold', 0.8)
        
        # Kalman filter parameters
        self.use_kalman = config.get('kalman_filter_enabled', True)
        self.process_noise = config.get('kalman_process_noise', 0.01)
        self.measurement_noise = config.get('kalman_measurement_noise', 0.1)
        
        # Signal history for analysis
        self.signal_history = defaultdict(list)
        self.processed_signals = []
        
        # Correlation matrix
        self.correlation_matrix = {}
        
        # Signal quality metrics
        self.quality_metrics = defaultdict(lambda: {
            'total_signals': 0,
            'passed_signals': 0,
            'filtered_signals': 0,
            'success_rate': 0.0
        })
        
        # Initialize filters
        self.kalman_filters = {}
        self.scaler = StandardScaler()
        
    async def process(self, signal: Signal) -> Optional[Signal]:
        """Main signal processing pipeline"""
        try:
            # Stage 1: Basic validation
            if not self._validate_basic(signal):
                return None
            
            # Stage 2: Noise filtering
            filtered_signal = await self._filter_noise(signal)
            
            # Stage 3: Correlation check
            if not await self._check_correlation(filtered_signal):
                return None
            
            # Stage 4: Signal enhancement
            enhanced_signal = await self._enhance_signal(filtered_signal)
            
            # Stage 5: Final scoring
            final_signal = await self._score_signal(enhanced_signal)
            
            # Stage 6: Quality check
            if not self._quality_check(final_signal):
                return None
            
            # Track processed signal
            self._track_signal(final_signal)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    def _validate_basic(self, signal: Signal) -> bool:
        """Basic signal validation"""
        # Check signal strength
        if signal.strength < self.min_signal_strength:
            logger.debug(f"Signal strength too low: {signal.strength}")
            self.quality_metrics[signal.symbol]['filtered_signals'] += 1
            return False
        
        # Check confidence
        min_confidence = self.config.get('min_signal_confidence', 0.5)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal confidence too low: {signal.confidence}")
            self.quality_metrics[signal.symbol]['filtered_signals'] += 1
            return False
        
        # Check for required fields
        if not signal.entry_price or not signal.stop_loss:
            logger.debug("Signal missing required fields")
            return False
        
        # Check risk/reward ratio
        if signal.take_profit and signal.entry_price and signal.stop_loss:
            reward = abs(float(signal.take_profit - signal.entry_price))
            risk = abs(float(signal.entry_price - signal.stop_loss))
            
            if risk > 0:
                rr_ratio = reward / risk
                min_rr = self.config.get('min_risk_reward_ratio', 1.5)
                
                if rr_ratio < min_rr:
                    logger.debug(f"Risk/reward ratio too low: {rr_ratio:.2f}")
                    return False
        
        return True
    
    async def _filter_noise(self, signal: Signal) -> Signal:
        """Apply noise filtering to signal"""
        if not self.use_kalman:
            return signal
        
        # Get or create Kalman filter for this symbol
        if signal.symbol not in self.kalman_filters:
            self.kalman_filters[signal.symbol] = self._create_kalman_filter()
        
        kf = self.kalman_filters[signal.symbol]
        
        # Apply Kalman filtering to signal strength
        filtered_strength = self._apply_kalman(
            kf,
            signal.strength,
            self.process_noise,
            self.measurement_noise
        )
        
        # Update signal with filtered values
        signal.strength = max(0, min(100, filtered_strength))
        
        # Apply smoothing to confidence
        if signal.symbol in self.signal_history:
            recent_confidences = [s.confidence for s in self.signal_history[signal.symbol][-5:]]
            if recent_confidences:
                # Exponential weighted average
                weights = np.exp(np.linspace(-1, 0, len(recent_confidences) + 1))
                weights = weights / weights.sum()
                
                all_confidences = recent_confidences + [signal.confidence]
                signal.confidence = np.average(all_confidences, weights=weights[1:])
        
        return signal
    
    async def _check_correlation(self, signal: Signal) -> bool:
        """Check if signal is too correlated with recent signals"""
        # Get recent signals for this symbol
        recent_signals = self.signal_history[signal.symbol][-10:]
        
        if len(recent_signals) < 2:
            return True
        
        # Check time correlation (too many signals in short time)
        time_window = timedelta(minutes=5)
        recent_time_signals = [
            s for s in recent_signals
            if datetime.utcnow() - s.timestamp < time_window
        ]
        
        if len(recent_time_signals) >= 3:
            logger.debug(f"Too many signals in time window: {len(recent_time_signals)}")
            return False
        
        # Check feature correlation
        if signal.indicators:
            for recent_signal in recent_signals[-3:]:
                if recent_signal.indicators:
                    correlation = self._calculate_indicator_correlation(
                        signal.indicators,
                        recent_signal.indicators
                    )
                    
                    if correlation > self.correlation_threshold:
                        logger.debug(f"Signal too correlated with recent: {correlation:.2f}")
                        return False
        
        return True
    
    async def _enhance_signal(self, signal: Signal) -> Signal:
        """Enhance signal with additional analysis"""
        # Add confirmation count
        confirmations = 0
        
        # Check multiple confirmation sources
        if signal.indicators:
            # Technical confirmations
            if signal.indicators.get('rsi'):
                if signal.action == 'BUY' and signal.indicators['rsi'] < 70:
                    confirmations += 1
                elif signal.action == 'SELL' and signal.indicators['rsi'] > 30:
                    confirmations += 1
            
            if signal.indicators.get('macd'):
                if signal.action == 'BUY' and signal.indicators['macd'] > signal.indicators.get('macd_signal', 0):
                    confirmations += 1
                elif signal.action == 'SELL' and signal.indicators['macd'] < signal.indicators.get('macd_signal', 0):
                    confirmations += 1
            
            if signal.indicators.get('volume_ratio'):
                if signal.indicators['volume_ratio'] > 1.2:
                    confirmations += 1
            
            if signal.indicators.get('adx'):
                if signal.indicators['adx'] > 25:
                    confirmations += 1
        
        # Add regime confirmation
        if signal.regime_alignment > 0.7:
            confirmations += 1
        
        # Add psychology confirmation
        if signal.psychology_score > 0.6:
            confirmations += 1
        
        # Update signal confidence based on confirmations
        confirmation_boost = confirmations / self.confirmation_sources * 0.2
        signal.confidence = min(1.0, signal.confidence + confirmation_boost)
        
        # Add metadata
        signal.indicators['confirmations'] = confirmations
        signal.indicators['quality_score'] = self._calculate_quality_score(signal)
        
        return signal
    
    async def _score_signal(self, signal: Signal) -> Signal:
        """Calculate final signal score"""
        # Multi-factor scoring
        scores = {
            'strength': signal.strength / 100 * 0.25,
            'confidence': signal.confidence * 0.25,
            'regime': signal.regime_alignment * 0.15,
            'psychology': signal.psychology_score * 0.15,
            'confirmations': signal.indicators.get('confirmations', 0) / self.confirmation_sources * 0.10,
            'quality': signal.indicators.get('quality_score', 0.5) * 0.10
        }
        
        # Calculate composite score
        final_score = sum(scores.values()) * 100
        
        # Adjust strength based on final score
        signal.strength = min(100, (signal.strength + final_score) / 2)
        
        # Add score breakdown to indicators
        signal.indicators['score_breakdown'] = scores
        signal.indicators['final_score'] = final_score
        
        return signal
    
    def _quality_check(self, signal: Signal) -> bool:
        """Final quality check before passing signal"""
        # Check minimum score
        min_score = self.config.get('min_final_score', 60)
        if signal.indicators.get('final_score', 0) < min_score:
            logger.debug(f"Signal failed final score check: {signal.indicators.get('final_score', 0):.1f}")
            return False
        
        # Check confirmation requirements
        min_confirmations = self.config.get('min_confirmations', 2)
        if signal.indicators.get('confirmations', 0) < min_confirmations:
            logger.debug(f"Insufficient confirmations: {signal.indicators.get('confirmations', 0)}")
            return False
        
        # Pass quality check
        self.quality_metrics[signal.symbol]['passed_signals'] += 1
        return True
    
    def _track_signal(self, signal: Signal):
        """Track processed signal for analysis"""
        # Add to history
        self.signal_history[signal.symbol].append(signal)
        self.processed_signals.append(signal)
        
        # Limit history size
        max_history = 100
        if len(self.signal_history[signal.symbol]) > max_history:
            self.signal_history[signal.symbol] = self.signal_history[signal.symbol][-max_history:]
        
        # Update metrics
        self.quality_metrics[signal.symbol]['total_signals'] += 1
        
        # Calculate success rate (would need trade results in production)
        total = self.quality_metrics[signal.symbol]['total_signals']
        passed = self.quality_metrics[signal.symbol]['passed_signals']
        
        if total > 0:
            self.quality_metrics[signal.symbol]['success_rate'] = passed / total
    
    def _create_kalman_filter(self) -> Dict:
        """Create a new Kalman filter"""
        return {
            'x': np.array([[0], [0]]),  # Initial state (position, velocity)
            'P': np.eye(2),  # Initial covariance
            'F': np.array([[1, 1], [0, 1]]),  # State transition
            'H': np.array([[1, 0]]),  # Measurement function
            'R': self.measurement_noise,  # Measurement noise
            'Q': np.array([[self.process_noise, 0], [0, self.process_noise]])  # Process noise
        }
    
    def _apply_kalman(self, kf: Dict, measurement: float, process_noise: float, measurement_noise: float) -> float:
        """Apply Kalman filtering"""
        # Prediction step
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        
        # Update step
        y = measurement - kf['H'] @ kf['x']  # Innovation
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']  # Innovation covariance
        K = kf['P'] @ kf['H'].T / S  # Kalman gain
        
        kf['x'] = kf['x'] + K * y
        kf['P'] = (np.eye(2) - K @ kf['H']) @ kf['P']
        
        return float(kf['x'][0])
    
    def _calculate_indicator_correlation(self, indicators1: Dict, indicators2: Dict) -> float:
        """Calculate correlation between two sets of indicators"""
        common_keys = set(indicators1.keys()) & set(indicators2.keys())
        
        if len(common_keys) < 3:
            return 0.0
        
        values1 = []
        values2 = []
        
        for key in common_keys:
            if isinstance(indicators1[key], (int, float)) and isinstance(indicators2[key], (int, float)):
                values1.append(indicators1[key])
                values2.append(indicators2[key])
        
        if len(values1) < 3:
            return 0.0
        
        # Normalize values
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Calculate correlation
        correlation = np.corrcoef(values1, values2)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_quality_score(self, signal: Signal) -> float:
        """Calculate overall signal quality score"""
        quality_factors = []
        
        # Historical success rate for this symbol
        if signal.symbol in self.quality_metrics:
            quality_factors.append(self.quality_metrics[signal.symbol]['success_rate'])
        
        # Signal clarity (how decisive the signal is)
        if signal.action in ['BUY', 'SELL']:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.3)
        
        # Risk/reward quality
        if signal.take_profit and signal.stop_loss and signal.entry_price:
            reward = abs(float(signal.take_profit - signal.entry_price))
            risk = abs(float(signal.entry_price - signal.stop_loss))
            
            if risk > 0:
                rr_ratio = reward / risk
                quality_factors.append(min(1.0, rr_ratio / 3))  # Cap at 3:1
        
        # Indicator agreement
        if signal.indicators and signal.indicators.get('confirmations'):
            confirmation_rate = signal.indicators['confirmations'] / self.confirmation_sources
            quality_factors.append(confirmation_rate)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def get_statistics(self) -> Dict:
        """Get signal processing statistics"""
        stats = {
            'total_processed': len(self.processed_signals),
            'symbols_tracked': len(self.signal_history),
            'quality_metrics': dict(self.quality_metrics),
            'average_strength': np.mean([s.strength for s in self.processed_signals]) if self.processed_signals else 0,
            'average_confidence': np.mean([s.confidence for s in self.processed_signals]) if self.processed_signals else 0
        }
        
        return stats

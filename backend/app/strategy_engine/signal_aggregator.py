"""
Signal Aggregator - Advanced Signal Fusion System
Step 8: World's #1 Strategy Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .core.strategy_config import StrategyConfig
from .models.strategy_models import Signal, SignalType, SignalSource, TradingSignal

logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """Signal fusion methods"""
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    ML_FUSION = "ml_fusion"
    BAYESIAN = "bayesian"
    FUZZY_LOGIC = "fuzzy_logic"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    SIMPLE_AVERAGE = "simple_average"

@dataclass
class SignalWeight:
    """Signal weight configuration"""
    signal_id: str
    weight: float
    confidence: float
    source: SignalSource
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """Signal fusion result"""
    aggregated_signal: TradingSignal
    fusion_method: FusionMethod
    input_signals: List[Signal]
    weights: List[SignalWeight]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class SignalAggregator:
    """
    Advanced Signal Aggregator - Multi-Method Signal Fusion System
    
    Features:
    - Weighted ensemble fusion
    - ML-based signal fusion
    - Bayesian signal combination
    - Fuzzy logic aggregation
    - Quantum superposition fusion
    - Simple average aggregation
    - Dynamic weight adjustment
    - Confidence-based filtering
    - Real-time signal processing
    - Performance monitoring
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Signal storage
        self.signals: Dict[str, Signal] = {}
        self.signal_weights: Dict[str, SignalWeight] = {}
        self.fusion_history: List[FusionResult] = []
        
        # Fusion methods
        self.fusion_methods = {
            FusionMethod.WEIGHTED_ENSEMBLE: self._weighted_ensemble_fusion,
            FusionMethod.ML_FUSION: self._ml_fusion,
            FusionMethod.BAYESIAN: self._bayesian_fusion,
            FusionMethod.FUZZY_LOGIC: self._fuzzy_logic_fusion,
            FusionMethod.QUANTUM_SUPERPOSITION: self._quantum_superposition_fusion,
            FusionMethod.SIMPLE_AVERAGE: self._simple_average_fusion
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals_processed': 0,
            'total_fusions_performed': 0,
            'avg_fusion_time': 0.0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'avg_confidence': 0.0,
            'method_usage': {method.value: 0 for method in FusionMethod}
        }
        
        # ML models for fusion
        self.ml_models = {}
        self.model_weights = {}
        
        # Bayesian parameters
        self.bayesian_priors = {}
        self.bayesian_likelihoods = {}
        
        # Fuzzy logic parameters
        self.fuzzy_rules = {}
        self.fuzzy_membership_functions = {}
        
        # Quantum parameters
        self.quantum_states = {}
        self.quantum_amplitudes = {}
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🔗 Signal Aggregator initialized successfully")
    
    def _initialize_components(self):
        """Initialize signal aggregator components"""
        try:
            # Initialize ML models
            self._initialize_ml_models()
            
            # Initialize Bayesian parameters
            self._initialize_bayesian_parameters()
            
            # Initialize fuzzy logic parameters
            self._initialize_fuzzy_logic_parameters()
            
            # Initialize quantum parameters
            self._initialize_quantum_parameters()
            
            self.logger.info("✅ Signal Aggregator components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize signal aggregator components: {e}")
            raise
    
    def _initialize_ml_models(self):
        """Initialize ML models for signal fusion"""
        try:
            # Initialize ensemble models
            self.ml_models['ensemble'] = {
                'random_forest': None,  # Will be loaded from file
                'gradient_boosting': None,  # Will be loaded from file
                'neural_network': None,  # Will be loaded from file
                'svm': None  # Will be loaded from file
            }
            
            # Initialize model weights
            self.model_weights = {
                'random_forest': 0.3,
                'gradient_boosting': 0.3,
                'neural_network': 0.2,
                'svm': 0.2
            }
            
            self.logger.info("✅ ML models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML models: {e}")
            raise
    
    def _initialize_bayesian_parameters(self):
        """Initialize Bayesian fusion parameters"""
        try:
            # Initialize priors for different signal sources
            self.bayesian_priors = {
                SignalSource.TECHNICAL: 0.3,
                SignalSource.ML: 0.25,
                SignalSource.ALTERNATIVE_DATA: 0.2,
                SignalSource.SENTIMENT: 0.15,
                SignalSource.QUANTUM: 0.1
            }
            
            # Initialize likelihood functions
            self.bayesian_likelihoods = {
                SignalSource.TECHNICAL: self._technical_likelihood,
                SignalSource.ML: self._ml_likelihood,
                SignalSource.ALTERNATIVE_DATA: self._alternative_data_likelihood,
                SignalSource.SENTIMENT: self._sentiment_likelihood,
                SignalSource.QUANTUM: self._quantum_likelihood
            }
            
            self.logger.info("✅ Bayesian parameters initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Bayesian parameters: {e}")
            raise
    
    def _initialize_fuzzy_logic_parameters(self):
        """Initialize fuzzy logic parameters"""
        try:
            # Initialize fuzzy rules
            self.fuzzy_rules = {
                'strong_buy': {
                    'conditions': ['high_confidence', 'positive_sentiment', 'technical_bullish'],
                    'output': 'strong_buy'
                },
                'buy': {
                    'conditions': ['medium_confidence', 'positive_sentiment'],
                    'output': 'buy'
                },
                'hold': {
                    'conditions': ['low_confidence', 'mixed_signals'],
                    'output': 'hold'
                },
                'sell': {
                    'conditions': ['medium_confidence', 'negative_sentiment'],
                    'output': 'sell'
                },
                'strong_sell': {
                    'conditions': ['high_confidence', 'negative_sentiment', 'technical_bearish'],
                    'output': 'strong_sell'
                }
            }
            
            # Initialize membership functions
            self.fuzzy_membership_functions = {
                'confidence': {
                    'low': lambda x: max(0, 1 - x / 0.3) if x <= 0.3 else 0,
                    'medium': lambda x: max(0, 1 - abs(x - 0.5) / 0.2) if 0.3 <= x <= 0.7 else 0,
                    'high': lambda x: max(0, (x - 0.7) / 0.3) if x >= 0.7 else 0
                },
                'sentiment': {
                    'negative': lambda x: max(0, 1 - (x + 1) / 0.5) if x <= -0.5 else 0,
                    'neutral': lambda x: max(0, 1 - abs(x) / 0.5) if -0.5 <= x <= 0.5 else 0,
                    'positive': lambda x: max(0, (x - 0.5) / 0.5) if x >= 0.5 else 0
                }
            }
            
            self.logger.info("✅ Fuzzy logic parameters initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize fuzzy logic parameters: {e}")
            raise
    
    def _initialize_quantum_parameters(self):
        """Initialize quantum superposition parameters"""
        try:
            # Initialize quantum states
            self.quantum_states = {
                'buy': np.array([1, 0], dtype=complex),
                'sell': np.array([0, 1], dtype=complex),
                'hold': np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            }
            
            # Initialize quantum amplitudes
            self.quantum_amplitudes = {
                SignalSource.TECHNICAL: 0.3 + 0.1j,
                SignalSource.ML: 0.25 + 0.05j,
                SignalSource.ALTERNATIVE_DATA: 0.2 + 0.02j,
                SignalSource.SENTIMENT: 0.15 + 0.01j,
                SignalSource.QUANTUM: 0.1 + 0.005j
            }
            
            self.logger.info("✅ Quantum parameters initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize quantum parameters: {e}")
            raise
    
    async def aggregate_signals(self, signals: List[Signal], method: FusionMethod = FusionMethod.WEIGHTED_ENSEMBLE) -> List[TradingSignal]:
        """Aggregate multiple signals into trading signals"""
        try:
            if not signals:
                return []
            
            start_time = time.time()
            
            # Filter signals by confidence
            filtered_signals = self._filter_signals_by_confidence(signals)
            
            if not filtered_signals:
                self.logger.warning("No signals passed confidence filter")
                return []
            
            # Group signals by symbol
            signals_by_symbol = self._group_signals_by_symbol(filtered_signals)
            
            # Aggregate signals for each symbol
            aggregated_signals = []
            for symbol, symbol_signals in signals_by_symbol.items():
                if len(symbol_signals) > 1:
                    # Multiple signals for symbol - perform fusion
                    fusion_result = await self._perform_fusion(symbol_signals, method)
                    if fusion_result:
                        aggregated_signals.append(fusion_result.aggregated_signal)
                else:
                    # Single signal for symbol - convert to trading signal
                    trading_signal = self._convert_to_trading_signal(symbol_signals[0])
                    aggregated_signals.append(trading_signal)
            
            # Update performance metrics
            fusion_time = time.time() - start_time
            self._update_performance_metrics(len(signals), len(aggregated_signals), fusion_time, method)
            
            self.logger.info(f"✅ Aggregated {len(signals)} signals into {len(aggregated_signals)} trading signals")
            return aggregated_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating signals: {e}")
            return []
    
    def _filter_signals_by_confidence(self, signals: List[Signal], min_confidence: float = 0.3) -> List[Signal]:
        """Filter signals by confidence threshold"""
        try:
            filtered_signals = [s for s in signals if s.confidence >= min_confidence]
            
            self.logger.debug(f"Filtered {len(signals)} signals to {len(filtered_signals)} by confidence >= {min_confidence}")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering signals by confidence: {e}")
            return signals
    
    def _group_signals_by_symbol(self, signals: List[Signal]) -> Dict[str, List[Signal]]:
        """Group signals by symbol"""
        try:
            signals_by_symbol = {}
            
            for signal in signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []
                signals_by_symbol[signal.symbol].append(signal)
            
            return signals_by_symbol
            
        except Exception as e:
            self.logger.error(f"❌ Error grouping signals by symbol: {e}")
            return {}
    
    async def _perform_fusion(self, signals: List[Signal], method: FusionMethod) -> Optional[FusionResult]:
        """Perform signal fusion using specified method"""
        try:
            if method not in self.fusion_methods:
                self.logger.error(f"Unknown fusion method: {method}")
                return None
            
            # Get fusion function
            fusion_function = self.fusion_methods[method]
            
            # Perform fusion
            fusion_result = await fusion_function(signals)
            
            if fusion_result:
                # Store fusion result
                self.fusion_history.append(fusion_result)
                
                # Update method usage
                self.performance_metrics['method_usage'][method.value] += 1
            
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"❌ Error performing fusion: {e}")
            return None
    
    async def _weighted_ensemble_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform weighted ensemble fusion"""
        try:
            # Calculate signal weights
            weights = self._calculate_signal_weights(signals)
            
            # Calculate weighted average
            total_weight = sum(w.weight for w in weights)
            if total_weight == 0:
                return None
            
            # Calculate weighted signal strength
            weighted_strength = sum(s.strength * w.weight for s, w in zip(signals, weights)) / total_weight
            
            # Calculate weighted confidence
            weighted_confidence = sum(s.confidence * w.weight for s, w in zip(signals, weights)) / total_weight
            
            # Determine signal type based on weighted strength
            if weighted_strength > 0.6:
                signal_type = SignalType.BUY
            elif weighted_strength < -0.6:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=signal_type,
                strength=weighted_strength,
                confidence=weighted_confidence,
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'weighted_ensemble',
                    'input_signals_count': len(signals),
                    'total_weight': total_weight
                }
            )
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.WEIGHTED_ENSEMBLE,
                input_signals=signals,
                weights=weights,
                confidence=weighted_confidence,
                timestamp=datetime.now(),
                metadata={'total_weight': total_weight}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in weighted ensemble fusion: {e}")
            return None
    
    async def _ml_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform ML-based fusion"""
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(signals)
            
            # Get ML predictions
            predictions = await self._get_ml_predictions(features)
            
            # Calculate ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(predictions)
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=ensemble_prediction['signal_type'],
                strength=ensemble_prediction['strength'],
                confidence=ensemble_prediction['confidence'],
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'ml_fusion',
                    'input_signals_count': len(signals),
                    'ml_predictions': predictions
                }
            )
            
            # Create dummy weights for ML fusion
            weights = [SignalWeight(
                signal_id=s.id,
                weight=1.0 / len(signals),
                confidence=s.confidence,
                source=s.source,
                timestamp=s.timestamp
            ) for s in signals]
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.ML_FUSION,
                input_signals=signals,
                weights=weights,
                confidence=ensemble_prediction['confidence'],
                timestamp=datetime.now(),
                metadata={'ml_predictions': predictions}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in ML fusion: {e}")
            return None
    
    async def _bayesian_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform Bayesian fusion"""
        try:
            # Calculate Bayesian probabilities
            probabilities = self._calculate_bayesian_probabilities(signals)
            
            # Determine signal type based on probabilities
            if probabilities['buy'] > probabilities['sell'] and probabilities['buy'] > probabilities['hold']:
                signal_type = SignalType.BUY
                confidence = probabilities['buy']
            elif probabilities['sell'] > probabilities['buy'] and probabilities['sell'] > probabilities['hold']:
                signal_type = SignalType.SELL
                confidence = probabilities['sell']
            else:
                signal_type = SignalType.HOLD
                confidence = probabilities['hold']
            
            # Calculate strength based on probability difference
            strength = (probabilities['buy'] - probabilities['sell']) * 2 - 1
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'bayesian',
                    'input_signals_count': len(signals),
                    'probabilities': probabilities
                }
            )
            
            # Create weights based on Bayesian probabilities
            weights = [SignalWeight(
                signal_id=s.id,
                weight=probabilities.get(s.source.value, 0.1),
                confidence=s.confidence,
                source=s.source,
                timestamp=s.timestamp
            ) for s in signals]
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.BAYESIAN,
                input_signals=signals,
                weights=weights,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={'probabilities': probabilities}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in Bayesian fusion: {e}")
            return None
    
    async def _fuzzy_logic_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform fuzzy logic fusion"""
        try:
            # Calculate fuzzy membership values
            membership_values = self._calculate_fuzzy_membership(signals)
            
            # Apply fuzzy rules
            rule_outputs = self._apply_fuzzy_rules(membership_values)
            
            # Defuzzify to get final signal
            final_signal = self._defuzzify(rule_outputs)
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=final_signal['signal_type'],
                strength=final_signal['strength'],
                confidence=final_signal['confidence'],
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'fuzzy_logic',
                    'input_signals_count': len(signals),
                    'membership_values': membership_values,
                    'rule_outputs': rule_outputs
                }
            )
            
            # Create weights based on fuzzy membership
            weights = [SignalWeight(
                signal_id=s.id,
                weight=membership_values.get(s.id, {}).get('overall', 0.1),
                confidence=s.confidence,
                source=s.source,
                timestamp=s.timestamp
            ) for s in signals]
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.FUZZY_LOGIC,
                input_signals=signals,
                weights=weights,
                confidence=final_signal['confidence'],
                timestamp=datetime.now(),
                metadata={'membership_values': membership_values, 'rule_outputs': rule_outputs}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in fuzzy logic fusion: {e}")
            return None
    
    async def _quantum_superposition_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform quantum superposition fusion"""
        try:
            # Create quantum superposition state
            superposition_state = self._create_quantum_superposition(signals)
            
            # Calculate quantum probabilities
            quantum_probabilities = self._calculate_quantum_probabilities(superposition_state)
            
            # Determine signal type based on quantum probabilities
            if quantum_probabilities['buy'] > quantum_probabilities['sell'] and quantum_probabilities['buy'] > quantum_probabilities['hold']:
                signal_type = SignalType.BUY
                confidence = quantum_probabilities['buy']
            elif quantum_probabilities['sell'] > quantum_probabilities['buy'] and quantum_probabilities['sell'] > quantum_probabilities['hold']:
                signal_type = SignalType.SELL
                confidence = quantum_probabilities['sell']
            else:
                signal_type = SignalType.HOLD
                confidence = quantum_probabilities['hold']
            
            # Calculate strength based on quantum probability difference
            strength = (quantum_probabilities['buy'] - quantum_probabilities['sell']) * 2 - 1
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'quantum_superposition',
                    'input_signals_count': len(signals),
                    'quantum_probabilities': quantum_probabilities
                }
            )
            
            # Create weights based on quantum amplitudes
            weights = [SignalWeight(
                signal_id=s.id,
                weight=abs(self.quantum_amplitudes.get(s.source, 0.1)),
                confidence=s.confidence,
                source=s.source,
                timestamp=s.timestamp
            ) for s in signals]
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.QUANTUM_SUPERPOSITION,
                input_signals=signals,
                weights=weights,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={'quantum_probabilities': quantum_probabilities}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in quantum superposition fusion: {e}")
            return None
    
    async def _simple_average_fusion(self, signals: List[Signal]) -> Optional[FusionResult]:
        """Perform simple average fusion"""
        try:
            # Calculate simple averages
            avg_strength = sum(s.strength for s in signals) / len(signals)
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            
            # Determine signal type based on average strength
            if avg_strength > 0.3:
                signal_type = SignalType.BUY
            elif avg_strength < -0.3:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signals[0].symbol,
                signal_type=signal_type,
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=datetime.now(),
                source=SignalSource.AGGREGATED,
                metadata={
                    'fusion_method': 'simple_average',
                    'input_signals_count': len(signals)
                }
            )
            
            # Create equal weights
            weights = [SignalWeight(
                signal_id=s.id,
                weight=1.0 / len(signals),
                confidence=s.confidence,
                source=s.source,
                timestamp=s.timestamp
            ) for s in signals]
            
            return FusionResult(
                aggregated_signal=aggregated_signal,
                fusion_method=FusionMethod.SIMPLE_AVERAGE,
                input_signals=signals,
                weights=weights,
                confidence=avg_confidence,
                timestamp=datetime.now(),
                metadata={'avg_strength': avg_strength, 'avg_confidence': avg_confidence}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in simple average fusion: {e}")
            return None
    
    def _calculate_signal_weights(self, signals: List[Signal]) -> List[SignalWeight]:
        """Calculate weights for signals"""
        try:
            weights = []
            
            for signal in signals:
                # Base weight from confidence
                base_weight = signal.confidence
                
                # Adjust weight based on source
                source_multiplier = self._get_source_weight_multiplier(signal.source)
                
                # Adjust weight based on recency
                recency_multiplier = self._get_recency_multiplier(signal.timestamp)
                
                # Calculate final weight
                final_weight = base_weight * source_multiplier * recency_multiplier
                
                weight = SignalWeight(
                    signal_id=signal.id,
                    weight=final_weight,
                    confidence=signal.confidence,
                    source=signal.source,
                    timestamp=signal.timestamp
                )
                weights.append(weight)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating signal weights: {e}")
            return []
    
    def _get_source_weight_multiplier(self, source: SignalSource) -> float:
        """Get weight multiplier for signal source"""
        multipliers = {
            SignalSource.TECHNICAL: 1.0,
            SignalSource.ML: 1.2,
            SignalSource.ALTERNATIVE_DATA: 0.8,
            SignalSource.SENTIMENT: 0.6,
            SignalSource.QUANTUM: 1.5
        }
        return multipliers.get(source, 1.0)
    
    def _get_recency_multiplier(self, timestamp: datetime) -> float:
        """Get weight multiplier based on signal recency"""
        try:
            now = datetime.now()
            age_seconds = (now - timestamp).total_seconds()
            
            # Decay weight over time
            if age_seconds < 60:  # Less than 1 minute
                return 1.0
            elif age_seconds < 300:  # Less than 5 minutes
                return 0.9
            elif age_seconds < 900:  # Less than 15 minutes
                return 0.7
            elif age_seconds < 3600:  # Less than 1 hour
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating recency multiplier: {e}")
            return 1.0
    
    def _prepare_ml_features(self, signals: List[Signal]) -> np.ndarray:
        """Prepare features for ML model"""
        try:
            features = []
            
            for signal in signals:
                feature_vector = [
                    signal.strength,
                    signal.confidence,
                    float(signal.source.value),
                    signal.timestamp.timestamp(),
                    len(signal.metadata) if signal.metadata else 0
                ]
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing ML features: {e}")
            return np.array([])
    
    async def _get_ml_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Get ML model predictions"""
        try:
            # Simplified ML prediction (in production, this would use real models)
            predictions = {
                'random_forest': {
                    'signal_type': SignalType.BUY if np.mean(features[:, 0]) > 0 else SignalType.SELL,
                    'strength': np.mean(features[:, 0]),
                    'confidence': np.mean(features[:, 1])
                },
                'gradient_boosting': {
                    'signal_type': SignalType.BUY if np.mean(features[:, 0]) > 0.1 else SignalType.SELL,
                    'strength': np.mean(features[:, 0]) * 1.1,
                    'confidence': np.mean(features[:, 1]) * 0.9
                },
                'neural_network': {
                    'signal_type': SignalType.BUY if np.mean(features[:, 0]) > -0.1 else SignalType.SELL,
                    'strength': np.mean(features[:, 0]) * 0.8,
                    'confidence': np.mean(features[:, 1]) * 1.1
                },
                'svm': {
                    'signal_type': SignalType.BUY if np.mean(features[:, 0]) > 0.2 else SignalType.SELL,
                    'strength': np.mean(features[:, 0]) * 1.2,
                    'confidence': np.mean(features[:, 1]) * 0.8
                }
            }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting ML predictions: {e}")
            return {}
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble prediction from ML models"""
        try:
            # Calculate weighted ensemble
            total_weight = sum(self.model_weights.values())
            
            weighted_strength = 0
            weighted_confidence = 0
            signal_types = []
            
            for model_name, prediction in predictions.items():
                weight = self.model_weights.get(model_name, 0.25)
                weighted_strength += prediction['strength'] * weight
                weighted_confidence += prediction['confidence'] * weight
                signal_types.append(prediction['signal_type'])
            
            # Normalize weights
            weighted_strength /= total_weight
            weighted_confidence /= total_weight
            
            # Determine final signal type
            buy_count = sum(1 for st in signal_types if st == SignalType.BUY)
            sell_count = sum(1 for st in signal_types if st == SignalType.SELL)
            
            if buy_count > sell_count:
                final_signal_type = SignalType.BUY
            elif sell_count > buy_count:
                final_signal_type = SignalType.SELL
            else:
                final_signal_type = SignalType.HOLD
            
            return {
                'signal_type': final_signal_type,
                'strength': weighted_strength,
                'confidence': weighted_confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ensemble prediction: {e}")
            return {
                'signal_type': SignalType.HOLD,
                'strength': 0.0,
                'confidence': 0.0
            }
    
    def _calculate_bayesian_probabilities(self, signals: List[Signal]) -> Dict[str, float]:
        """Calculate Bayesian probabilities for signal types"""
        try:
            # Initialize probabilities
            probabilities = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            
            # Calculate likelihood for each signal
            for signal in signals:
                likelihood = self.bayesian_likelihoods.get(signal.source, lambda x: 0.1)(signal)
                prior = self.bayesian_priors.get(signal.source, 0.1)
                
                # Update probabilities based on signal type
                if signal.signal_type == SignalType.BUY:
                    probabilities['buy'] += likelihood * prior
                elif signal.signal_type == SignalType.SELL:
                    probabilities['sell'] += likelihood * prior
                else:
                    probabilities['hold'] += likelihood * prior
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                for key in probabilities:
                    probabilities[key] /= total
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Bayesian probabilities: {e}")
            return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
    
    def _technical_likelihood(self, signal: Signal) -> float:
        """Calculate likelihood for technical signals"""
        return signal.confidence * 0.8
    
    def _ml_likelihood(self, signal: Signal) -> float:
        """Calculate likelihood for ML signals"""
        return signal.confidence * 0.9
    
    def _alternative_data_likelihood(self, signal: Signal) -> float:
        """Calculate likelihood for alternative data signals"""
        return signal.confidence * 0.7
    
    def _sentiment_likelihood(self, signal: Signal) -> float:
        """Calculate likelihood for sentiment signals"""
        return signal.confidence * 0.6
    
    def _quantum_likelihood(self, signal: Signal) -> float:
        """Calculate likelihood for quantum signals"""
        return signal.confidence * 1.0
    
    def _calculate_fuzzy_membership(self, signals: List[Signal]) -> Dict[str, Dict[str, float]]:
        """Calculate fuzzy membership values"""
        try:
            membership_values = {}
            
            for signal in signals:
                signal_membership = {}
                
                # Calculate confidence membership
                confidence_membership = {}
                for level, func in self.fuzzy_membership_functions['confidence'].items():
                    confidence_membership[level] = func(signal.confidence)
                signal_membership['confidence'] = confidence_membership
                
                # Calculate sentiment membership (simplified)
                sentiment_membership = {}
                for level, func in self.fuzzy_membership_functions['sentiment'].items():
                    sentiment_membership[level] = func(signal.strength)
                signal_membership['sentiment'] = sentiment_membership
                
                # Calculate overall membership
                signal_membership['overall'] = max(confidence_membership.values()) * max(sentiment_membership.values())
                
                membership_values[signal.id] = signal_membership
            
            return membership_values
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating fuzzy membership: {e}")
            return {}
    
    def _apply_fuzzy_rules(self, membership_values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Apply fuzzy rules to get outputs"""
        try:
            rule_outputs = {}
            
            for rule_name, rule in self.fuzzy_rules.items():
                # Calculate rule activation
                activation = 1.0
                
                for condition in rule['conditions']:
                    if condition == 'high_confidence':
                        activation *= max(mv.get('confidence', {}).get('high', 0) for mv in membership_values.values())
                    elif condition == 'medium_confidence':
                        activation *= max(mv.get('confidence', {}).get('medium', 0) for mv in membership_values.values())
                    elif condition == 'low_confidence':
                        activation *= max(mv.get('confidence', {}).get('low', 0) for mv in membership_values.values())
                    elif condition == 'positive_sentiment':
                        activation *= max(mv.get('sentiment', {}).get('positive', 0) for mv in membership_values.values())
                    elif condition == 'negative_sentiment':
                        activation *= max(mv.get('sentiment', {}).get('negative', 0) for mv in membership_values.values())
                    elif condition == 'mixed_signals':
                        activation *= 0.5  # Simplified
                    elif condition == 'technical_bullish':
                        activation *= 0.8  # Simplified
                    elif condition == 'technical_bearish':
                        activation *= 0.8  # Simplified
                
                rule_outputs[rule_name] = activation
            
            return rule_outputs
            
        except Exception as e:
            self.logger.error(f"❌ Error applying fuzzy rules: {e}")
            return {}
    
    def _defuzzify(self, rule_outputs: Dict[str, float]) -> Dict[str, Any]:
        """Defuzzify fuzzy outputs to get final signal"""
        try:
            # Find rule with highest activation
            best_rule = max(rule_outputs.items(), key=lambda x: x[1])
            rule_name, activation = best_rule
            
            # Map rule to signal type
            signal_type_mapping = {
                'strong_buy': SignalType.BUY,
                'buy': SignalType.BUY,
                'hold': SignalType.HOLD,
                'sell': SignalType.SELL,
                'strong_sell': SignalType.SELL
            }
            
            signal_type = signal_type_mapping.get(rule_name, SignalType.HOLD)
            
            # Calculate strength and confidence
            strength = activation * 2 - 1  # Convert to [-1, 1] range
            confidence = activation
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error defuzzifying: {e}")
            return {
                'signal_type': SignalType.HOLD,
                'strength': 0.0,
                'confidence': 0.0
            }
    
    def _create_quantum_superposition(self, signals: List[Signal]) -> np.ndarray:
        """Create quantum superposition state"""
        try:
            # Initialize superposition state
            superposition = np.zeros(3, dtype=complex)  # [buy, sell, hold]
            
            # Add each signal's contribution
            for signal in signals:
                amplitude = self.quantum_amplitudes.get(signal.source, 0.1)
                
                if signal.signal_type == SignalType.BUY:
                    superposition[0] += amplitude * signal.confidence
                elif signal.signal_type == SignalType.SELL:
                    superposition[1] += amplitude * signal.confidence
                else:
                    superposition[2] += amplitude * signal.confidence
            
            # Normalize superposition
            norm = np.linalg.norm(superposition)
            if norm > 0:
                superposition /= norm
            
            return superposition
            
        except Exception as e:
            self.logger.error(f"❌ Error creating quantum superposition: {e}")
            return np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex)
    
    def _calculate_quantum_probabilities(self, superposition: np.ndarray) -> Dict[str, float]:
        """Calculate quantum probabilities from superposition state"""
        try:
            # Calculate probabilities as |amplitude|^2
            probabilities = {
                'buy': abs(superposition[0]) ** 2,
                'sell': abs(superposition[1]) ** 2,
                'hold': abs(superposition[2]) ** 2
            }
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                for key in probabilities:
                    probabilities[key] /= total
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quantum probabilities: {e}")
            return {'buy': 0.33, 'sell': 0.33, 'hold': 0.34}
    
    def _convert_to_trading_signal(self, signal: Signal) -> TradingSignal:
        """Convert Signal to TradingSignal"""
        try:
            return TradingSignal(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                timestamp=datetime.now(),
                source=signal.source,
                metadata=signal.metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error converting signal to trading signal: {e}")
            return None
    
    def _update_performance_metrics(self, input_signals: int, output_signals: int, fusion_time: float, method: FusionMethod):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_signals_processed'] += input_signals
            self.performance_metrics['total_fusions_performed'] += 1
            
            # Update average fusion time
            total_fusions = self.performance_metrics['total_fusions_performed']
            current_avg = self.performance_metrics['avg_fusion_time']
            self.performance_metrics['avg_fusion_time'] = (current_avg * (total_fusions - 1) + fusion_time) / total_fusions
            
            # Update success/failure counts
            if output_signals > 0:
                self.performance_metrics['successful_fusions'] += 1
            else:
                self.performance_metrics['failed_fusions'] += 1
            
            # Update method usage
            self.performance_metrics['method_usage'][method.value] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")
    
    # Utility Methods
    
    def get_fusion_history(self, limit: int = 100) -> List[FusionResult]:
        """Get fusion history"""
        return self.fusion_history[-limit:] if self.fusion_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def get_signal_weights(self, signal_id: str) -> Optional[SignalWeight]:
        """Get signal weight"""
        return self.signal_weights.get(signal_id)
    
    def get_all_signal_weights(self) -> Dict[str, SignalWeight]:
        """Get all signal weights"""
        return self.signal_weights.copy()
    
    # Health Check
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now(),
                'performance_metrics': self.performance_metrics,
                'fusion_methods_available': len(self.fusion_methods),
                'ml_models_loaded': len([m for m in self.ml_models.values() if m is not None]),
                'bayesian_parameters_loaded': len(self.bayesian_priors),
                'fuzzy_rules_loaded': len(self.fuzzy_rules),
                'quantum_parameters_loaded': len(self.quantum_states)
            }
            
            # Check if any component is missing
            if health_status['ml_models_loaded'] == 0:
                health_status['status'] = 'degraded'
                health_status['warnings'] = ['ML models not loaded']
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Error performing health check: {e}")
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now(),
                'error': str(e)
            }

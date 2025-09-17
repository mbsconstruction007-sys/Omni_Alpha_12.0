"""
Step 13: Market Microstructure Signals
Advanced signal generation from order flow and microstructure analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MicrostructureSignals:
    """Generates sophisticated trading signals from market microstructure data"""
    
    def __init__(self, order_book_analyzer, volume_analyzer, flow_tracker):
        self.order_book = order_book_analyzer
        self.volume = volume_analyzer
        self.flow = flow_tracker
        
        # Signal weights and thresholds
        self.weights = {
            'order_book_imbalance': 0.25,
            'vpin_toxicity': 0.15,
            'large_orders': 0.20,
            'volume_profile': 0.15,
            'order_flow': 0.25
        }
        
        self.thresholds = {
            'strong_signal': 80,
            'medium_signal': 60,
            'weak_signal': 40,
            'toxicity_override': 70
        }
        
        # Signal history for trend analysis
        self.signal_history = []
        
    def generate_comprehensive_signal(self, symbol: str) -> Optional[Dict]:
        """Generate comprehensive trading signal from all microstructure components"""
        try:
            # Collect all microstructure data
            microstructure_data = self._collect_microstructure_data(symbol)
            
            if not microstructure_data:
                return None
            
            # Generate individual signals
            signals = self._generate_individual_signals(microstructure_data)
            
            # Calculate composite signal
            composite_signal = self._calculate_composite_signal(signals)
            
            # Apply risk filters
            filtered_signal = self._apply_risk_filters(composite_signal, microstructure_data)
            
            # Generate final recommendation
            final_signal = self._generate_final_recommendation(filtered_signal, microstructure_data)
            
            # Store in history
            self.signal_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': final_signal,
                'components': signals
            })
            
            # Keep only recent history
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _collect_microstructure_data(self, symbol: str) -> Optional[Dict]:
        """Collect all microstructure data for analysis"""
        try:
            data = {}
            
            # Order book analysis
            data['imbalance'] = self.order_book.get_order_book_imbalance(symbol)
            data['toxicity'] = self.order_book.calculate_vpin_toxicity(symbol)
            data['large_orders'] = self.order_book.detect_large_orders(symbol)
            data['spread'] = self.order_book.analyze_spread_dynamics(symbol)
            
            # Volume profile analysis
            data['volume_profile'] = self.volume.calculate_volume_profile(symbol)
            data['poc_levels'] = self.volume.find_poc_levels(symbol)
            data['hvn_lvn'] = self.volume.identify_hvn_lvn(symbol)
            
            # Order flow analysis
            data['aggressor_flow'] = self.flow.classify_aggressor_side(symbol)
            data['institutional_flow'] = self.flow.track_institutional_flow(symbol)
            data['flow_metrics'] = self.flow.calculate_flow_metrics(symbol)
            
            return data
            
        except Exception as e:
            logger.error(f"Data collection error for {symbol}: {e}")
            return None
    
    def _generate_individual_signals(self, data: Dict) -> Dict:
        """Generate individual signals from each component"""
        signals = {}
        
        # Order book imbalance signal
        if data['imbalance']:
            imbalance = data['imbalance']['imbalance']
            if imbalance > 0.4:
                signals['imbalance'] = {'signal': 'STRONG_BUY', 'confidence': 90}
            elif imbalance > 0.2:
                signals['imbalance'] = {'signal': 'BUY', 'confidence': 70}
            elif imbalance < -0.4:
                signals['imbalance'] = {'signal': 'STRONG_SELL', 'confidence': 90}
            elif imbalance < -0.2:
                signals['imbalance'] = {'signal': 'SELL', 'confidence': 70}
            else:
                signals['imbalance'] = {'signal': 'NEUTRAL', 'confidence': 50}
        
        # VPIN toxicity signal (risk filter)
        if data['toxicity']:
            vpin = data['toxicity']['vpin']
            if vpin > 0.8:
                signals['toxicity'] = {'signal': 'AVOID', 'confidence': 95}
            elif vpin > 0.6:
                signals['toxicity'] = {'signal': 'REDUCE_SIZE', 'confidence': 80}
            else:
                signals['toxicity'] = {'signal': 'NORMAL', 'confidence': 60}
        
        # Large orders signal
        if data['large_orders'] and data['large_orders']['detected']:
            direction = data['large_orders']['direction']
            confidence = data['large_orders']['confidence']
            participation = data['large_orders']['participation_rate']
            
            if participation > 30:
                signal_strength = 'STRONG_' + direction
            else:
                signal_strength = direction
                
            signals['large_orders'] = {'signal': signal_strength, 'confidence': confidence}
        else:
            signals['large_orders'] = {'signal': 'NEUTRAL', 'confidence': 50}
        
        # Volume profile signal
        if data['volume_profile']:
            position = data['volume_profile']['position_in_profile']
            if position == 'BELOW_VALUE':
                signals['volume_profile'] = {'signal': 'BUY', 'confidence': 65}
            elif position == 'ABOVE_VALUE':
                signals['volume_profile'] = {'signal': 'SELL', 'confidence': 65}
            else:
                signals['volume_profile'] = {'signal': 'NEUTRAL', 'confidence': 50}
        
        # Order flow signal
        if data['flow_metrics']:
            flow_signal = data['flow_metrics']['combined_signal']
            flow_strength = data['flow_metrics']['signal_strength']
            
            signals['order_flow'] = {'signal': flow_signal, 'confidence': flow_strength}
        
        # Institutional flow signal
        if data['institutional_flow'] and data['institutional_flow']['institutional_detected']:
            inst_direction = data['institutional_flow']['institutional_direction']
            inst_score = data['institutional_flow']['institutional_score']
            
            if inst_score > 80:
                signal_strength = 'STRONG_' + inst_direction if inst_direction != 'NEUTRAL' else 'NEUTRAL'
            else:
                signal_strength = inst_direction if inst_direction != 'NEUTRAL' else 'NEUTRAL'
                
            signals['institutional'] = {'signal': signal_strength, 'confidence': inst_score}
        
        return signals
    
    def _calculate_composite_signal(self, signals: Dict) -> Dict:
        """Calculate weighted composite signal from individual components"""
        
        # Signal scoring system
        signal_scores = {
            'STRONG_BUY': 100,
            'BUY': 75,
            'WEAK_BUY': 60,
            'NEUTRAL': 50,
            'WEAK_SELL': 40,
            'SELL': 25,
            'STRONG_SELL': 0,
            'AVOID': -50,
            'REDUCE_SIZE': 30
        }
        
        weighted_score = 0
        total_weight = 0
        confidence_sum = 0
        valid_signals = 0
        
        for signal_type, signal_data in signals.items():
            if signal_type in self.weights:
                weight = self.weights.get(signal_type, 0.1)
                signal_name = signal_data['signal']
                confidence = signal_data['confidence']
                
                # Handle special signals
                if signal_name == 'AVOID':
                    return {'signal': 'AVOID', 'confidence': 95, 'reason': 'High toxicity detected'}
                
                # Map signal to score
                if signal_name in signal_scores:
                    score = signal_scores[signal_name]
                    weighted_score += score * weight * (confidence / 100)
                    total_weight += weight
                    confidence_sum += confidence
                    valid_signals += 1
        
        if total_weight == 0 or valid_signals == 0:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'reason': 'Insufficient data'}
        
        # Calculate final score and confidence
        final_score = weighted_score / total_weight
        avg_confidence = confidence_sum / valid_signals
        
        # Map score back to signal
        if final_score >= 85:
            signal = 'STRONG_BUY'
        elif final_score >= 70:
            signal = 'BUY'
        elif final_score >= 55:
            signal = 'WEAK_BUY'
        elif final_score >= 45:
            signal = 'NEUTRAL'
        elif final_score >= 30:
            signal = 'WEAK_SELL'
        elif final_score >= 15:
            signal = 'SELL'
        else:
            signal = 'STRONG_SELL'
        
        return {
            'signal': signal,
            'confidence': min(95, avg_confidence),
            'score': final_score,
            'supporting_signals': valid_signals
        }
    
    def _apply_risk_filters(self, composite_signal: Dict, data: Dict) -> Dict:
        """Apply risk filters to the composite signal"""
        
        # Toxicity override
        if data['toxicity'] and data['toxicity']['vpin'] > 0.7:
            return {
                'signal': 'AVOID',
                'confidence': 95,
                'reason': f"High VPIN toxicity: {data['toxicity']['vpin']:.2f}",
                'original_signal': composite_signal['signal']
            }
        
        # Spread filter - wide spreads reduce signal confidence
        if data['spread']:
            spread_bps = data['spread']['relative_spread_bps']
            if spread_bps > 100:  # Very wide spread
                composite_signal['confidence'] *= 0.5
                composite_signal['reason'] = f"Wide spread reduces confidence: {spread_bps:.1f} bps"
            elif spread_bps > 50:  # Wide spread
                composite_signal['confidence'] *= 0.7
        
        # Volume filter - low volume reduces confidence
        if data['volume_profile']:
            total_volume = data['volume_profile']['total_volume']
            if total_volume < 100000:  # Low volume threshold
                composite_signal['confidence'] *= 0.6
                composite_signal['reason'] = composite_signal.get('reason', '') + ' Low volume'
        
        return composite_signal
    
    def _generate_final_recommendation(self, signal: Dict, data: Dict) -> Dict:
        """Generate final trading recommendation with entry/exit guidance"""
        
        recommendation = {
            'timestamp': datetime.now(),
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'reason': signal.get('reason', 'Microstructure analysis'),
            'supporting_signals': signal.get('supporting_signals', 0),
            'entry_timing': self._calculate_entry_timing(signal, data),
            'position_sizing': self._calculate_position_sizing(signal, data),
            'risk_level': self._assess_risk_level(signal, data),
            'time_horizon': self._estimate_time_horizon(signal, data),
            'stop_loss_guidance': self._generate_stop_loss_guidance(signal, data),
            'take_profit_guidance': self._generate_take_profit_guidance(signal, data)
        }
        
        return recommendation
    
    def _calculate_entry_timing(self, signal: Dict, data: Dict) -> str:
        """Calculate optimal entry timing"""
        
        if signal['signal'] == 'AVOID':
            return 'DO_NOT_ENTER'
        
        confidence = signal['confidence']
        
        # Check for institutional flow alignment
        institutional_aligned = False
        if data['institutional_flow'] and data['institutional_flow']['institutional_detected']:
            inst_direction = data['institutional_flow']['institutional_direction']
            if ('BUY' in signal['signal'] and inst_direction == 'BUYING') or \
               ('SELL' in signal['signal'] and inst_direction == 'SELLING'):
                institutional_aligned = True
        
        # Check order flow momentum
        flow_momentum = False
        if data['flow_metrics']:
            flow_signal = data['flow_metrics']['combined_signal']
            if flow_signal == signal['signal'] or \
               ('BUY' in flow_signal and 'BUY' in signal['signal']) or \
               ('SELL' in flow_signal and 'SELL' in signal['signal']):
                flow_momentum = True
        
        # Determine timing
        if confidence > 80 and institutional_aligned and flow_momentum:
            return 'IMMEDIATE'
        elif confidence > 70 and (institutional_aligned or flow_momentum):
            return 'SOON'
        elif confidence > 60:
            return 'WAIT_FOR_CONFIRMATION'
        else:
            return 'WAIT_FOR_BETTER_SETUP'
    
    def _calculate_position_sizing(self, signal: Dict, data: Dict) -> str:
        """Calculate recommended position sizing"""
        
        confidence = signal['confidence']
        
        # Base sizing on confidence
        if confidence > 85:
            base_size = 'LARGE'
        elif confidence > 70:
            base_size = 'NORMAL'
        elif confidence > 55:
            base_size = 'SMALL'
        else:
            base_size = 'VERY_SMALL'
        
        # Adjust for toxicity
        if data['toxicity'] and data['toxicity']['vpin'] > 0.5:
            if base_size == 'LARGE':
                base_size = 'NORMAL'
            elif base_size == 'NORMAL':
                base_size = 'SMALL'
            elif base_size == 'SMALL':
                base_size = 'VERY_SMALL'
        
        # Adjust for spread
        if data['spread'] and data['spread']['relative_spread_bps'] > 50:
            if base_size == 'LARGE':
                base_size = 'NORMAL'
            elif base_size == 'NORMAL':
                base_size = 'SMALL'
        
        return base_size
    
    def _assess_risk_level(self, signal: Dict, data: Dict) -> str:
        """Assess overall risk level of the trade"""
        
        risk_factors = 0
        
        # Toxicity risk
        if data['toxicity'] and data['toxicity']['vpin'] > 0.6:
            risk_factors += 2
        elif data['toxicity'] and data['toxicity']['vpin'] > 0.4:
            risk_factors += 1
        
        # Spread risk
        if data['spread'] and data['spread']['relative_spread_bps'] > 100:
            risk_factors += 2
        elif data['spread'] and data['spread']['relative_spread_bps'] > 50:
            risk_factors += 1
        
        # Volume risk
        if data['volume_profile'] and data['volume_profile']['total_volume'] < 50000:
            risk_factors += 1
        
        # Signal confidence risk
        if signal['confidence'] < 60:
            risk_factors += 1
        
        # Map to risk level
        if risk_factors >= 4:
            return 'VERY_HIGH'
        elif risk_factors >= 3:
            return 'HIGH'
        elif risk_factors >= 2:
            return 'MEDIUM'
        elif risk_factors >= 1:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _estimate_time_horizon(self, signal: Dict, data: Dict) -> str:
        """Estimate optimal holding time horizon"""
        
        # Base on signal strength and institutional involvement
        if 'STRONG' in signal['signal']:
            if data['institutional_flow'] and data['institutional_flow']['institutional_detected']:
                return 'MEDIUM_TERM'  # 1-5 days
            else:
                return 'SHORT_TERM'   # Hours to 1 day
        elif signal['confidence'] > 70:
            return 'SHORT_TERM'
        else:
            return 'VERY_SHORT_TERM'  # Minutes to hours
    
    def _generate_stop_loss_guidance(self, signal: Dict, data: Dict) -> Dict:
        """Generate stop loss guidance"""
        
        guidance = {'type': 'PERCENTAGE', 'value': 2.0}  # Default 2%
        
        # Adjust based on volatility and spread
        if data['spread']:
            spread_bps = data['spread']['relative_spread_bps']
            # Wider stops for wider spreads
            if spread_bps > 100:
                guidance['value'] = 4.0
            elif spread_bps > 50:
                guidance['value'] = 3.0
        
        # Use volume profile levels if available
        if data['volume_profile'] and data['hvn_lvn']:
            current_price = data['volume_profile']['current_price']
            
            if 'BUY' in signal['signal'] and data['hvn_lvn']['nearest_hvn']:
                hvn_price = data['hvn_lvn']['nearest_hvn']['price']
                if hvn_price < current_price:
                    # Use HVN as support
                    stop_distance = (current_price - hvn_price) / current_price * 100
                    if stop_distance < 5:  # Within reasonable range
                        guidance = {'type': 'LEVEL', 'value': hvn_price}
            
            elif 'SELL' in signal['signal'] and data['hvn_lvn']['nearest_hvn']:
                hvn_price = data['hvn_lvn']['nearest_hvn']['price']
                if hvn_price > current_price:
                    # Use HVN as resistance
                    stop_distance = (hvn_price - current_price) / current_price * 100
                    if stop_distance < 5:  # Within reasonable range
                        guidance = {'type': 'LEVEL', 'value': hvn_price}
        
        return guidance
    
    def _generate_take_profit_guidance(self, signal: Dict, data: Dict) -> Dict:
        """Generate take profit guidance"""
        
        guidance = {'type': 'PERCENTAGE', 'value': 4.0}  # Default 4%
        
        # Adjust based on signal strength
        if 'STRONG' in signal['signal']:
            guidance['value'] = 6.0
        elif signal['confidence'] < 60:
            guidance['value'] = 2.0
        
        # Use volume profile levels if available
        if data['volume_profile'] and data['hvn_lvn']:
            current_price = data['volume_profile']['current_price']
            
            if 'BUY' in signal['signal'] and data['hvn_lvn']['nearest_hvn']:
                # Look for resistance levels above current price
                resistance_levels = [hvn for hvn in data['hvn_lvn']['hvn_levels'] 
                                   if hvn['price'] > current_price]
                
                if resistance_levels:
                    nearest_resistance = min(resistance_levels, key=lambda x: x['price'])
                    profit_distance = (nearest_resistance['price'] - current_price) / current_price * 100
                    
                    if 2 <= profit_distance <= 10:  # Reasonable range
                        guidance = {'type': 'LEVEL', 'value': nearest_resistance['price']}
            
            elif 'SELL' in signal['signal'] and data['hvn_lvn']['nearest_hvn']:
                # Look for support levels below current price
                support_levels = [hvn for hvn in data['hvn_lvn']['hvn_levels'] 
                                if hvn['price'] < current_price]
                
                if support_levels:
                    nearest_support = max(support_levels, key=lambda x: x['price'])
                    profit_distance = (current_price - nearest_support['price']) / current_price * 100
                    
                    if 2 <= profit_distance <= 10:  # Reasonable range
                        guidance = {'type': 'LEVEL', 'value': nearest_support['price']}
        
        return guidance
    
    def get_signal_history(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get recent signal history"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_signals = [
            s for s in self.signal_history 
            if s['timestamp'] > cutoff_time and (symbol is None or s['symbol'] == symbol)
        ]
        
        return recent_signals
    
    def analyze_signal_performance(self, symbol: str, hours: int = 24) -> Dict:
        """Analyze recent signal performance"""
        
        history = self.get_signal_history(symbol, hours)
        
        if not history:
            return {'error': 'No signal history available'}
        
        # Count signal types
        signal_counts = {}
        for h in history:
            signal = h['signal']['signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([h['signal']['confidence'] for h in history])
        
        # Identify trends
        recent_signals = [h['signal']['signal'] for h in history[-10:]]  # Last 10 signals
        
        trend = 'NEUTRAL'
        if recent_signals.count('BUY') + recent_signals.count('STRONG_BUY') > 6:
            trend = 'BULLISH'
        elif recent_signals.count('SELL') + recent_signals.count('STRONG_SELL') > 6:
            trend = 'BEARISH'
        
        return {
            'total_signals': len(history),
            'signal_distribution': signal_counts,
            'average_confidence': avg_confidence,
            'recent_trend': trend,
            'most_common_signal': max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else 'NONE'
        }

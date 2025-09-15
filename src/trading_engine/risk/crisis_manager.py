"""
Crisis Manager - Handles market crises and black swan events
Implements defensive protocols and emergency procedures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

class CrisisManager:
    """
    Manages crisis situations and implements defensive strategies
    Protects capital during extreme market conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Crisis thresholds
        self.vix_crisis_threshold = config.get('crisis_vix_threshold', 40)
        self.drawdown_crisis_threshold = config.get('crisis_drawdown_threshold', 10)
        self.correlation_crisis_threshold = config.get('crisis_correlation_threshold', 0.9)
        self.volume_spike_threshold = config.get('crisis_volume_spike_threshold', 5.0)
        
        # Crisis response parameters
        self.position_reduction_factor = config.get('crisis_position_reduction', 0.5)
        self.stop_loss_tightening_factor = config.get('crisis_stop_loss_tightening', 0.5)
        self.cash_allocation_target = config.get('crisis_cash_allocation', 0.5)
        
        # Black swan hedges
        self.hedge_allocation = Decimal(str(config.get('tail_risk_hedge_percent', 1.0)))
        self.put_protection_enabled = config.get('put_protection_enabled', True)
        self.vix_hedge_enabled = config.get('vix_hedge_enabled', True)
        
        # Crisis state
        self.crisis_level = 0.0  # 0 = normal, 1 = maximum crisis
        self.crisis_indicators = {}
        self.active_hedges = {}
        self.crisis_history = []
        
        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reset_time = None
        
        logger.info("Crisis Manager initialized")
    
    async def assess_crisis_level(self) -> float:
        """Assess current crisis level from 0 to 1"""
        crisis_scores = {}
        
        # 1. VIX level check
        vix_level = await self._get_vix_level()
        if vix_level:
            vix_score = min(1.0, vix_level / self.vix_crisis_threshold)
            crisis_scores['vix'] = vix_score * 0.25
        
        # 2. Market drawdown check
        market_drawdown = await self._calculate_market_drawdown()
        dd_score = min(1.0, abs(market_drawdown) / self.drawdown_crisis_threshold)
        crisis_scores['drawdown'] = dd_score * 0.25
        
        # 3. Correlation breakdown check
        correlation = await self._check_correlation_breakdown()
        crisis_scores['correlation'] = correlation * 0.20
        
        # 4. Volume spike check
        volume_spike = await self._check_volume_spike()
        crisis_scores['volume'] = volume_spike * 0.15
        
        # 5. Credit stress check
        credit_stress = await self._check_credit_stress()
        crisis_scores['credit'] = credit_stress * 0.15
        
        # Calculate weighted crisis level
        self.crisis_level = sum(crisis_scores.values())
        self.crisis_indicators = crisis_scores
        
        # Log if crisis level is elevated
        if self.crisis_level > 0.5:
            logger.warning(f"Elevated crisis level: {self.crisis_level:.2f}")
            logger.warning(f"Crisis indicators: {crisis_scores}")
        
        # Track crisis history
        self.crisis_history.append({
            'timestamp': datetime.now(),
            'level': self.crisis_level,
            'indicators': crisis_scores.copy()
        })
        
        # Limit history
        if len(self.crisis_history) > 1000:
            self.crisis_history = self.crisis_history[-1000:]
        
        return self.crisis_level
    
    async def execute_crisis_protocols(self, positions: Dict):
        """Execute crisis management protocols"""
        if self.crisis_level < 0.3:
            return  # No action needed
        
        logger.warning(f"Executing crisis protocols - Level: {self.crisis_level:.2f}")
        
        if self.crisis_level >= 0.7:
            # Severe crisis - aggressive defensive measures
            await self._execute_severe_crisis_protocol(positions)
        elif self.crisis_level >= 0.5:
            # Moderate crisis - standard defensive measures
            await self._execute_moderate_crisis_protocol(positions)
        else:
            # Mild crisis - cautionary measures
            await self._execute_mild_crisis_protocol(positions)
    
    async def _execute_severe_crisis_protocol(self, positions: Dict):
        """Execute severe crisis protocol"""
        logger.critical("SEVERE CRISIS PROTOCOL ACTIVATED")
        
        # 1. Immediately reduce all positions by 70%
        await self._reduce_all_positions(0.7)
        
        # 2. Activate maximum hedges
        await self._activate_crisis_hedges('maximum')
        
        # 3. Move to cash
        await self._increase_cash_allocation(0.8)
        
        # 4. Tighten all stops to 1%
        await self._tighten_stop_losses(0.01)
        
        # 5. Disable new position opening
        await self._disable_new_positions()
        
        # 6. Activate circuit breaker
        self._trigger_circuit_breaker()
    
    async def _execute_moderate_crisis_protocol(self, positions: Dict):
        """Execute moderate crisis protocol"""
        logger.warning("Moderate crisis protocol activated")
        
        # 1. Reduce positions by 50%
        await self._reduce_all_positions(0.5)
        
        # 2. Activate standard hedges
        await self._activate_crisis_hedges('standard')
        
        # 3. Increase cash to 50%
        await self._increase_cash_allocation(0.5)
        
        # 4. Tighten stops to 2%
        await self._tighten_stop_losses(0.02)
    
    async def _execute_mild_crisis_protocol(self, positions: Dict):
        """Execute mild crisis protocol"""
        logger.info("Mild crisis protocol activated")
        
        # 1. Reduce positions by 25%
        await self._reduce_all_positions(0.25)
        
        # 2. Activate basic hedges
        await self._activate_crisis_hedges('basic')
        
        # 3. Increase cash to 30%
        await self._increase_cash_allocation(0.3)
    
    async def activate_hedges(self):
        """Activate defensive hedges"""
        hedges_activated = []
        
        # 1. Put protection
        if self.put_protection_enabled:
            put_hedge = await self._activate_put_protection()
            if put_hedge:
                hedges_activated.append(put_hedge)
                self.active_hedges['puts'] = put_hedge
        
        # 2. VIX hedge
        if self.vix_hedge_enabled:
            vix_hedge = await self._activate_vix_hedge()
            if vix_hedge:
                hedges_activated.append(vix_hedge)
                self.active_hedges['vix'] = vix_hedge
        
        # 3. Gold allocation
        gold_hedge = await self._activate_gold_hedge()
        if gold_hedge:
            hedges_activated.append(gold_hedge)
            self.active_hedges['gold'] = gold_hedge
        
        # 4. Treasury bonds
        bond_hedge = await self._activate_bond_hedge()
        if bond_hedge:
            hedges_activated.append(bond_hedge)
            self.active_hedges['bonds'] = bond_hedge
        
        logger.info(f"Activated {len(hedges_activated)} hedges")
        return hedges_activated
    
    async def _activate_put_protection(self) -> Optional[Dict]:
        """Activate put option protection"""
        try:
            # Calculate put strike (10% OTM)
            spy_price = await self._get_spy_price()
            if not spy_price:
                return None
            
            put_strike = float(spy_price) * 0.9
            
            hedge = {
                'type': 'put_option',
                'symbol': 'SPY',
                'strike': put_strike,
                'expiry': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'quantity': self._calculate_hedge_size('puts'),
                'activated': datetime.now()
            }
            
            logger.info(f"Put protection activated: Strike {put_strike}")
            return hedge
            
        except Exception as e:
            logger.error(f"Failed to activate put protection: {e}")
            return None
    
    async def _activate_vix_hedge(self) -> Optional[Dict]:
        """Activate VIX hedge"""
        try:
            hedge = {
                'type': 'vix_calls',
                'symbol': 'VXX',
                'quantity': self._calculate_hedge_size('vix'),
                'activated': datetime.now()
            }
            
            logger.info("VIX hedge activated")
            return hedge
            
        except Exception as e:
            logger.error(f"Failed to activate VIX hedge: {e}")
            return None
    
    async def _activate_gold_hedge(self) -> Optional[Dict]:
        """Activate gold hedge"""
        try:
            hedge = {
                'type': 'gold',
                'symbol': 'GLD',
                'allocation': 0.05,  # 5% allocation
                'activated': datetime.now()
            }
            
            logger.info("Gold hedge activated")
            return hedge
            
        except Exception as e:
            logger.error(f"Failed to activate gold hedge: {e}")
            return None
    
    async def _activate_bond_hedge(self) -> Optional[Dict]:
        """Activate treasury bond hedge"""
        try:
            hedge = {
                'type': 'bonds',
                'symbol': 'TLT',
                'allocation': 0.10,  # 10% allocation
                'activated': datetime.now()
            }
            
            logger.info("Bond hedge activated")
            return hedge
            
        except Exception as e:
            logger.error(f"Failed to activate bond hedge: {e}")
            return None
    
    def _calculate_hedge_size(self, hedge_type: str) -> int:
        """Calculate appropriate hedge size"""
        # Simplified calculation - would be more complex in production
        base_size = 100
        
        if self.crisis_level > 0.7:
            multiplier = 3
        elif self.crisis_level > 0.5:
            multiplier = 2
        else:
            multiplier = 1
        
        return base_size * multiplier
    
    async def _activate_crisis_hedges(self, level: str):
        """Activate crisis hedges based on level"""
        if level == 'maximum':
            # Activate all hedges at maximum size
            await self.activate_hedges()
        elif level == 'standard':
            # Activate put and VIX hedges
            await self._activate_put_protection()
            await self._activate_vix_hedge()
        else:  # basic
            # Activate only put protection
            await self._activate_put_protection()
    
    def _trigger_circuit_breaker(self):
        """Trigger circuit breaker"""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_reset_time = datetime.now() + timedelta(hours=1)
        logger.critical("CIRCUIT BREAKER TRIGGERED - Trading halted for 1 hour")
    
    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active"""
        if self.circuit_breaker_triggered:
            if datetime.now() > self.circuit_breaker_reset_time:
                self.circuit_breaker_triggered = False
                logger.info("Circuit breaker reset")
                return False
            return True
        return False
    
    # Helper methods for crisis detection
    
    async def _get_vix_level(self) -> Optional[float]:
        """Get current VIX level"""
        # Would fetch from market data
        return 25.0  # Placeholder
    
    async def _calculate_market_drawdown(self) -> float:
        """Calculate current market drawdown"""
        # Would calculate from SPY data
        return -5.0  # Placeholder
    
    async def _check_correlation_breakdown(self) -> float:
        """Check for correlation breakdown"""
        # Would analyze correlation matrix
        return 0.3  # Placeholder
    
    async def _check_volume_spike(self) -> float:
        """Check for unusual volume spikes"""
        # Would analyze volume data
        return 0.2  # Placeholder
    
    async def _check_credit_stress(self) -> float:
        """Check credit market stress indicators"""
        # Would check credit spreads, etc.
        return 0.1  # Placeholder
    
    async def _get_spy_price(self) -> Optional[float]:
        """Get current SPY price"""
        return 450.0  # Placeholder
    
    async def _reduce_all_positions(self, reduction_factor: float):
        """Reduce all positions by specified factor"""
        logger.info(f"Reducing all positions by {reduction_factor:.0%}")
    
    async def _increase_cash_allocation(self, target_cash: float):
        """Increase cash allocation to target percentage"""
        logger.info(f"Increasing cash allocation to {target_cash:.0%}")
    
    async def _tighten_stop_losses(self, stop_percentage: float):
        """Tighten all stop losses to specified percentage"""
        logger.info(f"Tightening stop losses to {stop_percentage:.1%}")
    
    async def _disable_new_positions(self):
        """Disable opening of new positions"""
        logger.warning("New position opening disabled")
    
    def get_crisis_report(self) -> Dict:
        """Get comprehensive crisis report"""
        return {
            'crisis_level': self.crisis_level,
            'indicators': self.crisis_indicators,
            'active_hedges': len(self.active_hedges),
            'circuit_breaker': self.circuit_breaker_triggered,
            'last_assessment': self.crisis_history[-1] if self.crisis_history else None
        }

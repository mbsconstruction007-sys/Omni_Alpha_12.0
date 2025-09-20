"""
OMNI ALPHA 5.0 - PRODUCTION RISK MANAGEMENT SYSTEM
==================================================
Complete risk management system for live trading
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    daily_pnl: float = 0.0
    daily_trades: int = 0
    max_drawdown: float = 0.0
    current_positions: int = 0
    portfolio_value: float = 0.0
    buying_power: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class PositionRisk:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    position_value: float
    risk_percent: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class RiskManager:
    """Production risk management system with comprehensive controls"""
    
    def __init__(self, config):
        self.config = config
        
        # Risk limits
        self.max_daily_loss = config.get('MAX_DAILY_LOSS', 1000)
        self.max_position_size = config.get('MAX_POSITION_SIZE', 10000)
        self.max_positions = config.get('MAX_POSITIONS', 5)
        self.max_portfolio_risk = config.get('MAX_PORTFOLIO_RISK', 0.02)  # 2%
        self.max_single_position_risk = config.get('MAX_SINGLE_POSITION_RISK', 0.01)  # 1%
        self.max_sector_concentration = config.get('MAX_SECTOR_CONCENTRATION', 0.3)  # 30%
        self.max_daily_trades = config.get('MAX_DAILY_TRADES', 20)
        
        # Risk tracking
        self.risk_metrics = RiskMetrics()
        self.position_risks = {}
        self.daily_trades = []
        self.risk_events = []
        
        # Kelly Criterion parameters
        self.kelly_lookback = 100
        self.max_kelly_fraction = 0.25
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        
        # Sector mappings
        self.sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
            'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'VTI': 'ETF'
        }
        
    def check_pre_trade_risk(self, order: Dict, account: Dict, positions: Dict) -> Tuple[bool, str, Dict]:
        """Comprehensive pre-trade risk checks"""
        
        try:
            risk_details = {
                'checks_passed': [],
                'checks_failed': [],
                'warnings': [],
                'risk_score': 0.0
            }
            
            # Update current metrics
            self._update_risk_metrics(account, positions)
            
            # Check 1: Circuit breaker
            if self.circuit_breaker_active:
                risk_details['checks_failed'].append('Circuit breaker active')
                return False, f"Circuit breaker active: {self.circuit_breaker_reason}", risk_details
            
            # Check 2: Daily loss limit
            if self.risk_metrics.daily_pnl <= -self.max_daily_loss:
                risk_details['checks_failed'].append('Daily loss limit reached')
                self._trigger_circuit_breaker("Daily loss limit exceeded")
                return False, f"Daily loss limit reached: ${self.risk_metrics.daily_pnl:.2f}", risk_details
            
            risk_details['checks_passed'].append('Daily loss limit OK')
            
            # Check 3: Daily trade limit
            if self.risk_metrics.daily_trades >= self.max_daily_trades:
                risk_details['checks_failed'].append('Daily trade limit reached')
                return False, f"Daily trade limit reached: {self.risk_metrics.daily_trades}", risk_details
            
            risk_details['checks_passed'].append('Daily trade limit OK')
            
            # Check 4: Position count
            if len(positions) >= self.max_positions:
                risk_details['checks_failed'].append('Maximum positions reached')
                return False, f"Maximum {self.max_positions} positions allowed", risk_details
            
            risk_details['checks_passed'].append('Position count OK')
            
            # Check 5: Position size
            order_value = abs(order['quantity']) * order['price']
            if order_value > self.max_position_size:
                risk_details['checks_failed'].append('Position size too large')
                return False, f"Position size ${order_value:.2f} exceeds limit ${self.max_position_size}", risk_details
            
            risk_details['checks_passed'].append('Position size OK')
            
            # Check 6: Buying power
            required_buying_power = order_value * 1.1  # 10% buffer
            if required_buying_power > account.get('buying_power', 0):
                risk_details['checks_failed'].append('Insufficient buying power')
                return False, f"Insufficient buying power: need ${required_buying_power:.2f}, have ${account.get('buying_power', 0):.2f}", risk_details
            
            risk_details['checks_passed'].append('Buying power OK')
            
            # Check 7: Portfolio risk
            portfolio_value = account.get('portfolio_value', 0)
            if portfolio_value > 0:
                position_risk = order_value / portfolio_value
                if position_risk > self.max_single_position_risk:
                    risk_details['checks_failed'].append('Single position risk too high')
                    return False, f"Single position risk {position_risk:.2%} exceeds limit {self.max_single_position_risk:.2%}", risk_details
            
            risk_details['checks_passed'].append('Portfolio risk OK')
            
            # Check 8: Concentration risk
            symbol = order['symbol']
            sector = self.sector_map.get(symbol, 'Other')
            sector_exposure = self._calculate_sector_exposure(positions, sector, order_value)
            
            if sector_exposure > self.max_sector_concentration and portfolio_value > 0:
                risk_details['warnings'].append(f'High sector concentration: {sector} {sector_exposure:.1%}')
                if sector_exposure > self.max_sector_concentration * 1.5:  # Hard limit at 1.5x
                    risk_details['checks_failed'].append('Sector concentration too high')
                    return False, f"Sector concentration {sector_exposure:.1%} exceeds limit", risk_details
            
            risk_details['checks_passed'].append('Concentration risk OK')
            
            # Check 9: Existing position risk
            if symbol in positions:
                existing_value = abs(positions[symbol].get('market_value', 0))
                total_exposure = existing_value + order_value
                if total_exposure > self.max_position_size * 1.2:  # 20% buffer
                    risk_details['checks_failed'].append('Combined position size too large')
                    return False, f"Combined position size ${total_exposure:.2f} exceeds safe limit", risk_details
            
            risk_details['checks_passed'].append('Existing position risk OK')
            
            # Check 10: Volatility risk
            volatility_risk = self._assess_volatility_risk(symbol, order_value)
            if volatility_risk > 0.8:  # High volatility
                risk_details['warnings'].append(f'High volatility risk: {volatility_risk:.2f}')
                if volatility_risk > 0.95:  # Extreme volatility
                    risk_details['checks_failed'].append('Volatility too high')
                    return False, f"Volatility risk too high: {volatility_risk:.2f}", risk_details
            
            risk_details['checks_passed'].append('Volatility risk OK')
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(order, account, positions, risk_details)
            risk_details['risk_score'] = risk_score
            
            # Final decision
            if risk_score > 0.8:
                risk_details['warnings'].append(f'High risk score: {risk_score:.2f}')
            
            return True, "All risk checks passed", risk_details
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, f"Risk check error: {e}", {'error': str(e)}
    
    def calculate_position_size(self, signal: Dict, account: Dict, price: float, 
                              historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """Advanced position sizing using multiple methods"""
        
        try:
            portfolio_value = account.get('portfolio_value', 0)
            buying_power = account.get('buying_power', 0)
            
            if portfolio_value <= 0:
                return {'quantity': 0, 'reason': 'No portfolio value', 'method': 'none'}
            
            # Method 1: Fixed fractional
            fixed_fraction = 0.05  # 5% of portfolio per position
            fixed_size = portfolio_value * fixed_fraction
            
            # Method 2: Kelly Criterion (if historical data available)
            kelly_size = fixed_size  # Default to fixed if no historical data
            if historical_data is not None and len(historical_data) > 20:
                kelly_fraction = self._calculate_kelly_fraction(historical_data, signal)
                kelly_size = portfolio_value * min(kelly_fraction, self.max_kelly_fraction)
            
            # Method 3: Volatility-adjusted sizing
            volatility_multiplier = self._get_volatility_multiplier(signal.get('symbol', ''))
            vol_adjusted_size = fixed_size * volatility_multiplier
            
            # Method 4: Confidence-based sizing
            confidence = signal.get('confidence', 0.5)
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale confidence 0.5-1.0 to 1.0-2.0, cap at 1.0
            confidence_size = fixed_size * confidence_multiplier
            
            # Combine methods (weighted average)
            weights = {
                'fixed': 0.3,
                'kelly': 0.3,
                'volatility': 0.2,
                'confidence': 0.2
            }
            
            combined_size = (
                weights['fixed'] * fixed_size +
                weights['kelly'] * kelly_size +
                weights['volatility'] * vol_adjusted_size +
                weights['confidence'] * confidence_size
            )
            
            # Apply limits
            max_size = min(
                self.max_position_size,
                portfolio_value * self.max_single_position_risk / 0.02,  # Assume 2% risk per position
                buying_power * 0.9  # Leave 10% buying power buffer
            )
            
            final_size = min(combined_size, max_size)
            
            # Convert to shares
            shares = int(final_size / price)
            
            # Minimum position check
            min_shares = max(1, int(1000 / price))  # Minimum $1000 position or 1 share
            if shares < min_shares and final_size > 500:  # Only if we have enough capital
                shares = min_shares
            
            # Final validation
            final_value = shares * price
            if final_value > buying_power:
                shares = int(buying_power * 0.95 / price)  # Use 95% of buying power
            
            return {
                'quantity': max(0, shares),
                'position_value': shares * price,
                'portfolio_percent': (shares * price) / portfolio_value if portfolio_value > 0 else 0,
                'method': 'combined',
                'components': {
                    'fixed': fixed_size,
                    'kelly': kelly_size,
                    'volatility': vol_adjusted_size,
                    'confidence': confidence_size,
                    'combined': combined_size,
                    'final': final_size
                },
                'limits_applied': final_size != combined_size,
                'volatility_multiplier': volatility_multiplier,
                'confidence_multiplier': confidence_multiplier
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return {'quantity': 0, 'reason': f'Sizing error: {e}', 'method': 'error'}
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, 
                          symbol: str = None, atr_data: pd.DataFrame = None) -> float:
        """Dynamic stop loss calculation using multiple methods"""
        
        try:
            # Method 1: Fixed percentage
            base_stop_pct = 0.02  # 2% base stop
            
            # Method 2: ATR-based stop (if data available)
            atr_stop_pct = base_stop_pct
            if atr_data is not None and len(atr_data) > 14:
                atr = self._calculate_atr(atr_data)
                current_price = atr_data['close'].iloc[-1]
                atr_stop_pct = min(atr / current_price, 0.05)  # Cap at 5%
            
            # Method 3: Volatility-based adjustment
            vol_multiplier = self._get_volatility_multiplier(symbol) if symbol else 1.0
            vol_adjusted_stop = base_stop_pct * (2.0 - vol_multiplier)  # Higher vol = tighter stop
            
            # Use the most conservative (tightest) stop
            final_stop_pct = min(base_stop_pct, atr_stop_pct, vol_adjusted_stop)
            final_stop_pct = max(final_stop_pct, 0.005)  # Minimum 0.5% stop
            
            if position_type.upper() == 'LONG':
                stop_price = entry_price * (1 - final_stop_pct)
            else:  # SHORT
                stop_price = entry_price * (1 + final_stop_pct)
            
            return round(stop_price, 2)
            
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            # Fallback to simple percentage
            if position_type.upper() == 'LONG':
                return round(entry_price * 0.98, 2)
            else:
                return round(entry_price * 1.02, 2)
    
    def calculate_take_profit(self, entry_price: float, position_type: str, 
                            signal_confidence: float = 0.6, risk_reward_ratio: float = 2.5) -> float:
        """Dynamic take profit calculation"""
        
        try:
            # Base risk (stop loss distance)
            base_risk_pct = 0.02
            
            # Adjust risk-reward ratio based on confidence
            adjusted_ratio = risk_reward_ratio * min(signal_confidence / 0.6, 1.5)  # Scale with confidence
            
            # Calculate take profit distance
            take_profit_pct = base_risk_pct * adjusted_ratio
            
            if position_type.upper() == 'LONG':
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # SHORT
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            return round(take_profit_price, 2)
            
        except Exception as e:
            logger.error(f"Take profit calculation error: {e}")
            # Fallback
            if position_type.upper() == 'LONG':
                return round(entry_price * 1.05, 2)
            else:
                return round(entry_price * 0.95, 2)
    
    def update_daily_pnl(self, pnl: float, trade_count: int = 1):
        """Update daily P&L and check for circuit breaker conditions"""
        
        try:
            self.risk_metrics.daily_pnl += pnl
            self.risk_metrics.daily_trades += trade_count
            self.risk_metrics.last_update = datetime.now()
            
            # Record trade
            self.daily_trades.append({
                'timestamp': datetime.now(),
                'pnl': pnl,
                'cumulative_pnl': self.risk_metrics.daily_pnl
            })
            
            # Check circuit breaker conditions
            if self.risk_metrics.daily_pnl <= -self.max_daily_loss:
                self._trigger_circuit_breaker("Daily loss limit exceeded")
            
            # Update risk level
            self._update_risk_level()
            
            logger.info(f"Daily P&L updated: ${self.risk_metrics.daily_pnl:.2f}, Trades: {self.risk_metrics.daily_trades}")
            
        except Exception as e:
            logger.error(f"P&L update error: {e}")
    
    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status"""
        
        return {
            'risk_metrics': {
                'daily_pnl': self.risk_metrics.daily_pnl,
                'daily_trades': self.risk_metrics.daily_trades,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'current_positions': self.risk_metrics.current_positions,
                'risk_level': self.risk_metrics.risk_level.value
            },
            'limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_position_size': self.max_position_size,
                'max_positions': self.max_positions,
                'max_daily_trades': self.max_daily_trades
            },
            'circuit_breaker': {
                'active': self.circuit_breaker_active,
                'reason': self.circuit_breaker_reason
            },
            'utilization': {
                'daily_loss_used': abs(self.risk_metrics.daily_pnl) / self.max_daily_loss,
                'trades_used': self.risk_metrics.daily_trades / self.max_daily_trades,
                'positions_used': self.risk_metrics.current_positions / self.max_positions
            },
            'last_update': self.risk_metrics.last_update.isoformat()
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (called at market open)"""
        
        self.risk_metrics.daily_pnl = 0.0
        self.risk_metrics.daily_trades = 0
        self.daily_trades = []
        
        # Reset circuit breaker if it was daily-loss related
        if self.circuit_breaker_active and "daily loss" in (self.circuit_breaker_reason or "").lower():
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = None
        
        logger.info("Daily risk metrics reset")
    
    def _update_risk_metrics(self, account: Dict, positions: Dict):
        """Update risk metrics from current account state"""
        
        self.risk_metrics.current_positions = len(positions)
        self.risk_metrics.portfolio_value = account.get('portfolio_value', 0)
        self.risk_metrics.buying_power = account.get('buying_power', 0)
        
        # Calculate current drawdown
        if 'equity' in account and 'last_equity' in account:
            current_drawdown = (account['last_equity'] - account['equity']) / account['last_equity']
            self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)
    
    def _calculate_sector_exposure(self, positions: Dict, sector: str, additional_value: float = 0) -> float:
        """Calculate sector concentration"""
        
        total_portfolio_value = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
        sector_value = sum(
            abs(pos.get('market_value', 0)) 
            for symbol, pos in positions.items() 
            if self.sector_map.get(symbol, 'Other') == sector
        ) + additional_value
        
        return sector_value / total_portfolio_value if total_portfolio_value > 0 else 0
    
    def _assess_volatility_risk(self, symbol: str, position_value: float) -> float:
        """Assess volatility-based risk (simplified)"""
        
        # Simplified volatility risk based on symbol
        high_vol_symbols = ['TSLA', 'NVDA', 'MEME_STOCKS']
        medium_vol_symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        base_risk = position_value / 10000  # Base risk from position size
        
        if symbol in high_vol_symbols:
            return min(base_risk * 1.5, 1.0)
        elif symbol in medium_vol_symbols:
            return min(base_risk * 1.0, 1.0)
        else:
            return min(base_risk * 0.8, 1.0)
    
    def _calculate_risk_score(self, order: Dict, account: Dict, positions: Dict, risk_details: Dict) -> float:
        """Calculate overall risk score for the trade"""
        
        try:
            score_components = []
            
            # Portfolio utilization
            portfolio_value = account.get('portfolio_value', 1)
            position_size = abs(order['quantity']) * order['price']
            portfolio_util = position_size / portfolio_value
            score_components.append(min(portfolio_util / 0.1, 1.0))  # 0.1 = 10% target
            
            # Daily loss utilization
            daily_loss_util = abs(self.risk_metrics.daily_pnl) / self.max_daily_loss
            score_components.append(daily_loss_util)
            
            # Position count utilization
            position_util = len(positions) / self.max_positions
            score_components.append(position_util)
            
            # Trade frequency
            trade_util = self.risk_metrics.daily_trades / self.max_daily_trades
            score_components.append(trade_util)
            
            # Volatility risk
            vol_risk = self._assess_volatility_risk(order['symbol'], position_size)
            score_components.append(vol_risk)
            
            # Calculate weighted average
            return np.mean(score_components)
            
        except Exception as e:
            logger.error(f"Risk score calculation error: {e}")
            return 0.5  # Medium risk default
    
    def _calculate_kelly_fraction(self, historical_data: pd.DataFrame, signal: Dict) -> float:
        """Calculate Kelly Criterion fraction from historical data"""
        
        try:
            if len(historical_data) < 20:
                return 0.05  # Default 5%
            
            # Calculate returns
            returns = historical_data['close'].pct_change().dropna()
            
            # Estimate win rate and average win/loss based on signal type
            confidence = signal.get('confidence', 0.6)
            
            # Simple Kelly approximation
            win_rate = confidence
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.02
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.02
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                return max(0, min(kelly_fraction, self.max_kelly_fraction))
            
            return 0.05
            
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return 0.05
    
    def _get_volatility_multiplier(self, symbol: str) -> float:
        """Get volatility-based position size multiplier"""
        
        # Simplified volatility mapping
        volatility_map = {
            'SPY': 1.2, 'QQQ': 1.1, 'IWM': 1.0,
            'AAPL': 1.0, 'MSFT': 1.0, 'GOOGL': 0.9,
            'TSLA': 0.6, 'NVDA': 0.7, 'META': 0.8
        }
        
        return volatility_map.get(symbol, 1.0)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        
        try:
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['close'].iloc[-1] * 0.02
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return data['close'].iloc[-1] * 0.02  # 2% fallback
    
    def _update_risk_level(self):
        """Update overall risk level based on current metrics"""
        
        risk_factors = []
        
        # Daily loss factor
        loss_factor = abs(self.risk_metrics.daily_pnl) / self.max_daily_loss
        risk_factors.append(loss_factor)
        
        # Position count factor
        position_factor = self.risk_metrics.current_positions / self.max_positions
        risk_factors.append(position_factor)
        
        # Trade frequency factor
        trade_factor = self.risk_metrics.daily_trades / self.max_daily_trades
        risk_factors.append(trade_factor)
        
        # Calculate overall risk
        overall_risk = max(risk_factors)
        
        if overall_risk >= 0.8:
            self.risk_metrics.risk_level = RiskLevel.CRITICAL
        elif overall_risk >= 0.6:
            self.risk_metrics.risk_level = RiskLevel.HIGH
        elif overall_risk >= 0.3:
            self.risk_metrics.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_metrics.risk_level = RiskLevel.LOW
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to stop trading"""
        
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        
        self.risk_events.append({
            'timestamp': datetime.now(),
            'event_type': 'circuit_breaker',
            'reason': reason,
            'metrics': self.risk_metrics.__dict__.copy()
        })
        
        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def reset_circuit_breaker(self, manual_override: bool = False):
        """Reset circuit breaker (manual intervention required)"""
        
        if manual_override or datetime.now().hour >= 9:  # Auto-reset at market open
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = None
            logger.info("Circuit breaker reset")
        else:
            logger.warning("Circuit breaker reset requires manual override or market open")

# Example usage
if __name__ == "__main__":
    config = {
        'MAX_DAILY_LOSS': 1000,
        'MAX_POSITION_SIZE': 10000,
        'MAX_POSITIONS': 5
    }
    
    risk_manager = RiskManager(config)
    
    # Test order
    order = {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0}
    account = {'portfolio_value': 100000, 'buying_power': 50000}
    positions = {}
    
    can_trade, reason, details = risk_manager.check_pre_trade_risk(order, account, positions)
    print(f"Can trade: {can_trade}, Reason: {reason}")
    print(f"Risk details: {details}")

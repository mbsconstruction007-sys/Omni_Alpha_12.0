"""
OMNI ALPHA 5.0 - RISK MANAGEMENT ENGINE
=======================================
Comprehensive risk management with real-time monitoring and automated controls
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

# Set decimal precision for financial calculations
getcontext().prec = 10

from config.settings import get_settings
from config.logging_config import get_logger, log_risk
from infrastructure.monitoring import get_metrics_collector
from infrastructure.circuit_breaker import circuit_breaker, ErrorSeverity

# ===================== RISK DATA STRUCTURES =====================

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    side: PositionSide
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    entry_time: datetime
    last_updated: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.avg_entry_price == 0:
            return 0.0
        return float((self.current_price - self.avg_entry_price) / self.avg_entry_price * 100)
    
    @property
    def position_value(self) -> Decimal:
        """Calculate position value"""
        return abs(self.quantity) * self.current_price
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        return self.unrealized_pnl > 0

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    portfolio_value: Decimal
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    leverage: float
    var_95: Decimal  # Value at Risk 95%
    var_99: Decimal  # Value at Risk 99%
    max_drawdown: Decimal
    current_drawdown: Decimal
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    timestamp: datetime

@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    limit_type: str  # 'absolute', 'percentage', 'count'
    limit_value: Decimal
    current_value: Decimal
    utilization_percent: float
    is_breached: bool
    warning_threshold: float = 0.8  # Warn at 80% of limit

# ===================== RISK CALCULATIONS =====================

class RiskCalculator:
    """Risk calculation utilities"""
    
    @staticmethod
    def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not returns or len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        return float(np.percentile(returns_array, (1 - confidence) * 100))
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
    
    @staticmethod
    def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
        """Calculate beta vs market"""
        if not asset_returns or not market_returns or len(asset_returns) != len(market_returns):
            return 1.0
        
        asset_array = np.array(asset_returns)
        market_array = np.array(market_returns)
        
        covariance = np.cov(asset_array, market_array)[0, 1]
        market_variance = np.var(market_array)
        
        if market_variance == 0:
            return 1.0
        
        return float(covariance / market_variance)
    
    @staticmethod
    def calculate_correlation_matrix(returns_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets"""
        symbols = list(returns_dict.keys())
        correlation_matrix = {}
        
        for symbol1 in symbols:
            correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    returns1 = np.array(returns_dict[symbol1])
                    returns2 = np.array(returns_dict[symbol2])
                    
                    if len(returns1) > 1 and len(returns2) > 1:
                        correlation = float(np.corrcoef(returns1, returns2)[0, 1])
                        correlation_matrix[symbol1][symbol2] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0
        
        return correlation_matrix

# ===================== RISK ENGINE =====================

class RiskEngine:
    """Main risk management engine"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'risk_engine')
        self.metrics = get_metrics_collector()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
        # Risk tracking
        self.daily_pnl = Decimal('0')
        self.daily_trades = 0
        self.max_drawdown = Decimal('0')
        self.peak_portfolio_value = Decimal('0')
        self.daily_returns: List[float] = []
        self.portfolio_values: List[Decimal] = []
        
        # Risk limits
        self.risk_limits: Dict[str, RiskLimit] = {}
        self._initialize_risk_limits()
        
        # Threading
        self._lock = threading.Lock()
        
        # Start time for daily calculations
        self.start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _initialize_risk_limits(self):
        """Initialize risk limits from configuration"""
        trading_config = self.settings.trading
        
        # Position size limits
        self.risk_limits['max_position_size'] = RiskLimit(
            name='Maximum Position Size',
            limit_type='absolute',
            limit_value=Decimal(str(trading_config.max_position_size_dollars)),
            current_value=Decimal('0'),
            utilization_percent=0.0,
            is_breached=False
        )
        
        # Daily trade limit
        self.risk_limits['daily_trades'] = RiskLimit(
            name='Daily Trade Limit',
            limit_type='count',
            limit_value=Decimal(str(trading_config.max_daily_trades)),
            current_value=Decimal('0'),
            utilization_percent=0.0,
            is_breached=False
        )
        
        # Daily loss limit
        self.risk_limits['daily_loss'] = RiskLimit(
            name='Daily Loss Limit',
            limit_type='absolute',
            limit_value=Decimal(str(trading_config.max_daily_loss)),
            current_value=Decimal('0'),
            utilization_percent=0.0,
            is_breached=False
        )
        
        # Drawdown limit
        self.risk_limits['max_drawdown'] = RiskLimit(
            name='Maximum Drawdown',
            limit_type='percentage',
            limit_value=Decimal(str(trading_config.max_drawdown_percent * 100)),
            current_value=Decimal('0'),
            utilization_percent=0.0,
            is_breached=False
        )
    
    def update_position(self, symbol: str, quantity: int, price: Decimal, 
                       side: str, strategy: str = "default") -> bool:
        """Update position after trade execution"""
        with self._lock:
            current_time = datetime.now()
            
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                # Calculate new average price and quantity
                if side.lower() == 'buy':
                    if position.side == PositionSide.LONG:
                        # Adding to long position
                        total_cost = (position.avg_entry_price * position.quantity) + (price * quantity)
                        total_quantity = position.quantity + quantity
                        new_avg_price = total_cost / total_quantity if total_quantity > 0 else price
                        
                        position.quantity = total_quantity
                        position.avg_entry_price = new_avg_price
                    elif position.side == PositionSide.SHORT:
                        # Covering short position
                        position.quantity = max(0, position.quantity - quantity)
                        if position.quantity == 0:
                            position.side = PositionSide.FLAT
                    
                elif side.lower() == 'sell':
                    if position.side == PositionSide.LONG:
                        # Reducing long position
                        position.quantity = max(0, position.quantity - quantity)
                        if position.quantity == 0:
                            position.side = PositionSide.FLAT
                    elif position.side == PositionSide.SHORT:
                        # Adding to short position
                        total_cost = (position.avg_entry_price * position.quantity) + (price * quantity)
                        total_quantity = position.quantity + quantity
                        new_avg_price = total_cost / total_quantity if total_quantity > 0 else price
                        
                        position.quantity = total_quantity
                        position.avg_entry_price = new_avg_price
                
                position.current_price = price
                position.last_updated = current_time
                
            else:
                # Create new position
                position_side = PositionSide.LONG if side.lower() == 'buy' else PositionSide.SHORT
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    side=position_side,
                    avg_entry_price=price,
                    current_price=price,
                    market_value=price * quantity,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    entry_time=current_time,
                    last_updated=current_time,
                    strategy=strategy
                )
                
                self.positions[symbol] = position
            
            # Update position metrics
            self._update_position_metrics(position)
            
            # Update daily trade count
            self.daily_trades += 1
            
            # Log position update
            log_risk(
                f"Position updated: {side} {quantity} {symbol} @ ${price}",
                symbol=symbol,
                risk_type='position_update',
                risk_value=float(position.position_value)
            )
            
            return True
    
    def _update_position_metrics(self, position: Position):
        """Update position-related metrics"""
        # Calculate unrealized P&L
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (position.current_price - position.avg_entry_price) * position.quantity
        elif position.side == PositionSide.SHORT:
            position.unrealized_pnl = (position.avg_entry_price - position.current_price) * position.quantity
        else:
            position.unrealized_pnl = Decimal('0')
        
        position.market_value = position.position_value
    
    def check_pre_trade_risk(self, symbol: str, quantity: int, price: Decimal, 
                           side: str) -> Tuple[bool, str, RiskLevel]:
        """Comprehensive pre-trade risk check"""
        with self._lock:
            risk_issues = []
            max_risk_level = RiskLevel.LOW
            
            # 1. Position size check
            position_value = Decimal(str(quantity)) * price
            max_position_size = self.risk_limits['max_position_size'].limit_value
            
            if position_value > max_position_size:
                risk_issues.append(f"Position size ${position_value} exceeds limit ${max_position_size}")
                max_risk_level = RiskLevel.HIGH
            
            # 2. Daily trade limit check
            daily_trade_limit = int(self.risk_limits['daily_trades'].limit_value)
            if self.daily_trades >= daily_trade_limit:
                risk_issues.append(f"Daily trade limit reached: {self.daily_trades}/{daily_trade_limit}")
                max_risk_level = RiskLevel.CRITICAL
            
            # 3. Daily loss limit check
            daily_loss_limit = self.risk_limits['daily_loss'].limit_value
            if self.daily_pnl < -daily_loss_limit:
                risk_issues.append(f"Daily loss limit exceeded: ${self.daily_pnl}")
                max_risk_level = RiskLevel.CRITICAL
            
            # 4. Drawdown check
            current_drawdown_pct = self._calculate_current_drawdown_percent()
            max_drawdown_pct = float(self.risk_limits['max_drawdown'].limit_value)
            
            if current_drawdown_pct > max_drawdown_pct:
                risk_issues.append(f"Drawdown limit exceeded: {current_drawdown_pct:.2f}% > {max_drawdown_pct:.2f}%")
                max_risk_level = RiskLevel.CRITICAL
            
            # 5. Concentration risk check
            concentration_risk = self._calculate_concentration_risk(symbol, position_value)
            if concentration_risk > 0.3:  # 30% concentration threshold
                risk_issues.append(f"High concentration risk: {concentration_risk:.1%}")
                max_risk_level = max(max_risk_level, RiskLevel.MEDIUM)
            
            # 6. Correlation risk check
            correlation_risk = self._calculate_correlation_risk(symbol)
            if correlation_risk > 0.8:  # High correlation threshold
                risk_issues.append(f"High correlation risk: {correlation_risk:.2f}")
                max_risk_level = max(max_risk_level, RiskLevel.MEDIUM)
            
            # 7. Leverage check
            leverage = self._calculate_leverage()
            if leverage > 2.0:  # 2x leverage limit
                risk_issues.append(f"Leverage too high: {leverage:.2f}x")
                max_risk_level = max(max_risk_level, RiskLevel.HIGH)
            
            # Update risk limits utilization
            self._update_risk_limits_utilization()
            
            # Log risk check
            if risk_issues:
                log_risk(
                    f"Pre-trade risk check failed: {'; '.join(risk_issues)}",
                    symbol=symbol,
                    risk_type='pre_trade_check',
                    risk_value=float(position_value)
                )
                return False, '; '.join(risk_issues), max_risk_level
            else:
                return True, "All risk checks passed", RiskLevel.LOW
    
    def _calculate_current_drawdown_percent(self) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_portfolio_value <= 0:
            return 0.0
        
        current_value = self._get_current_portfolio_value()
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        return float(drawdown * 100)
    
    def _calculate_concentration_risk(self, symbol: str, new_position_value: Decimal) -> float:
        """Calculate concentration risk for symbol"""
        total_portfolio_value = self._get_current_portfolio_value()
        
        if total_portfolio_value <= 0:
            return 0.0
        
        # Current exposure to symbol
        current_exposure = Decimal('0')
        if symbol in self.positions:
            current_exposure = self.positions[symbol].position_value
        
        # New total exposure
        total_exposure = current_exposure + new_position_value
        
        return float(total_exposure / total_portfolio_value)
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        if not self.positions or len(self.positions) < 2:
            return 0.0
        
        # Simplified correlation calculation
        # In production, use actual price correlation
        sector_correlations = {
            'AAPL': ['MSFT', 'GOOGL', 'META'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN'],
            'GOOGL': ['AAPL', 'MSFT', 'META'],
            # Add more correlations as needed
        }
        
        if symbol not in sector_correlations:
            return 0.0
        
        correlated_symbols = sector_correlations[symbol]
        correlated_exposure = sum(
            float(pos.position_value) 
            for sym, pos in self.positions.items() 
            if sym in correlated_symbols
        )
        
        total_portfolio_value = float(self._get_current_portfolio_value())
        
        if total_portfolio_value <= 0:
            return 0.0
        
        return correlated_exposure / total_portfolio_value
    
    def _calculate_leverage(self) -> float:
        """Calculate current leverage"""
        gross_exposure = sum(float(pos.position_value) for pos in self.positions.values())
        portfolio_value = float(self._get_current_portfolio_value())
        
        if portfolio_value <= 0:
            return 0.0
        
        return gross_exposure / portfolio_value
    
    def _get_current_portfolio_value(self) -> Decimal:
        """Get current portfolio value"""
        # Calculate from positions + cash
        position_value = sum(pos.market_value for pos in self.positions.values())
        cash_value = Decimal('100000')  # TODO: Get actual cash from broker
        
        return position_value + cash_value
    
    def _update_risk_limits_utilization(self):
        """Update risk limits utilization"""
        # Position size utilization
        max_position_value = max(
            (float(pos.position_value) for pos in self.positions.values()), 
            default=0.0
        )
        position_limit = self.risk_limits['max_position_size']
        position_limit.current_value = Decimal(str(max_position_value))
        position_limit.utilization_percent = float(position_limit.current_value / position_limit.limit_value * 100)
        position_limit.is_breached = position_limit.utilization_percent > 100
        
        # Daily trades utilization
        trade_limit = self.risk_limits['daily_trades']
        trade_limit.current_value = Decimal(str(self.daily_trades))
        trade_limit.utilization_percent = float(trade_limit.current_value / trade_limit.limit_value * 100)
        trade_limit.is_breached = trade_limit.utilization_percent > 100
        
        # Daily loss utilization
        loss_limit = self.risk_limits['daily_loss']
        loss_limit.current_value = abs(min(self.daily_pnl, Decimal('0')))
        loss_limit.utilization_percent = float(loss_limit.current_value / loss_limit.limit_value * 100)
        loss_limit.is_breached = loss_limit.utilization_percent > 100
        
        # Drawdown utilization
        drawdown_limit = self.risk_limits['max_drawdown']
        current_drawdown = Decimal(str(self._calculate_current_drawdown_percent()))
        drawdown_limit.current_value = current_drawdown
        drawdown_limit.utilization_percent = float(current_drawdown / drawdown_limit.limit_value * 100)
        drawdown_limit.is_breached = drawdown_limit.utilization_percent > 100
    
    def update_market_prices(self, price_updates: Dict[str, Decimal]):
        """Update current market prices for positions"""
        with self._lock:
            for symbol, price in price_updates.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.current_price = price
                    position.last_updated = datetime.now()
                    
                    # Update position metrics
                    self._update_position_metrics(position)
            
            # Update portfolio-level metrics
            self._update_portfolio_metrics()
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level risk metrics"""
        current_value = self._get_current_portfolio_value()
        
        # Update peak value and drawdown
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        current_drawdown = self._calculate_current_drawdown_percent()
        self.max_drawdown = max(self.max_drawdown, Decimal(str(current_drawdown)))
        
        # Update daily P&L
        if self.portfolio_values:
            start_value = self.portfolio_values[0]
            self.daily_pnl = current_value - start_value
        
        # Store portfolio value for returns calculation
        self.portfolio_values.append(current_value)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                daily_return = float((current_value - prev_value) / prev_value)
                self.daily_returns.append(daily_return)
        
        # Update metrics
        self.metrics.update_portfolio_value(float(current_value))
        self.metrics.update_daily_pnl(float(self.daily_pnl))
        self.metrics.update_positions_count(len(self.positions))
        self.metrics.update_drawdown(current_drawdown)
        
        # Calculate and update risk score
        risk_score = self._calculate_overall_risk_score()
        self.metrics.update_risk_score(risk_score)
    
    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        risk_factors = []
        
        # Drawdown risk
        drawdown_pct = self._calculate_current_drawdown_percent()
        max_drawdown_pct = float(self.settings.trading.max_drawdown_percent * 100)
        drawdown_risk = min(1.0, drawdown_pct / max_drawdown_pct)
        risk_factors.append(drawdown_risk)
        
        # Concentration risk
        if self.positions:
            portfolio_value = float(self._get_current_portfolio_value())
            max_position_pct = max(
                float(pos.position_value) / portfolio_value 
                for pos in self.positions.values()
            ) if portfolio_value > 0 else 0
            concentration_risk = min(1.0, max_position_pct / 0.2)  # 20% threshold
            risk_factors.append(concentration_risk)
        
        # Leverage risk
        leverage = self._calculate_leverage()
        leverage_risk = min(1.0, leverage / 2.0)  # 2x threshold
        risk_factors.append(leverage_risk)
        
        # Daily loss risk
        daily_loss_pct = abs(float(min(self.daily_pnl, Decimal('0')))) / float(self.settings.trading.max_daily_loss)
        loss_risk = min(1.0, daily_loss_pct)
        risk_factors.append(loss_risk)
        
        # Return weighted average
        weights = [0.3, 0.25, 0.25, 0.2]  # Drawdown, concentration, leverage, daily loss
        return sum(risk * weight for risk, weight in zip(risk_factors, weights))
    
    def calculate_position_risk_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate risk metrics for specific position"""
        if symbol not in self.positions:
            return {}
        
        position = self.positions[symbol]
        portfolio_value = self._get_current_portfolio_value()
        
        return {
            'symbol': symbol,
            'position_value': float(position.position_value),
            'unrealized_pnl': float(position.unrealized_pnl),
            'unrealized_pnl_percent': position.unrealized_pnl_percent,
            'portfolio_weight': float(position.position_value / portfolio_value) if portfolio_value > 0 else 0,
            'days_held': (datetime.now() - position.entry_time).days,
            'risk_level': self._assess_position_risk_level(position).value
        }
    
    def _assess_position_risk_level(self, position: Position) -> RiskLevel:
        """Assess risk level for position"""
        # Check unrealized P&L percentage
        pnl_pct = abs(position.unrealized_pnl_percent)
        
        if pnl_pct > 10:
            return RiskLevel.CRITICAL
        elif pnl_pct > 5:
            return RiskLevel.HIGH
        elif pnl_pct > 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        with self._lock:
            portfolio_value = self._get_current_portfolio_value()
            
            # Calculate metrics
            total_exposure = sum(float(pos.position_value) for pos in self.positions.values())
            net_exposure = sum(
                float(pos.position_value) * (1 if pos.side == PositionSide.LONG else -1)
                for pos in self.positions.values()
            )
            
            # Risk metrics
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=Decimal(str(total_exposure)),
                net_exposure=Decimal(str(net_exposure)),
                gross_exposure=Decimal(str(abs(total_exposure))),
                leverage=self._calculate_leverage(),
                var_95=Decimal(str(RiskCalculator.calculate_var(self.daily_returns, 0.95))),
                var_99=Decimal(str(RiskCalculator.calculate_var(self.daily_returns, 0.99))),
                max_drawdown=self.max_drawdown,
                current_drawdown=Decimal(str(self._calculate_current_drawdown_percent())),
                sharpe_ratio=RiskCalculator.calculate_sharpe_ratio(self.daily_returns),
                beta=1.0,  # TODO: Calculate vs market
                correlation_risk=0.0,  # TODO: Calculate
                concentration_risk=max(
                    float(pos.position_value / portfolio_value) if portfolio_value > 0 else 0
                    for pos in self.positions.values()
                ) if self.positions else 0.0,
                timestamp=datetime.now()
            )
            
            return {
                'risk_metrics': {
                    'portfolio_value': float(risk_metrics.portfolio_value),
                    'total_exposure': float(risk_metrics.total_exposure),
                    'net_exposure': float(risk_metrics.net_exposure),
                    'leverage': risk_metrics.leverage,
                    'var_95': float(risk_metrics.var_95),
                    'var_99': float(risk_metrics.var_99),
                    'max_drawdown': float(risk_metrics.max_drawdown),
                    'current_drawdown': float(risk_metrics.current_drawdown),
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'concentration_risk': risk_metrics.concentration_risk
                },
                'risk_limits': {
                    name: {
                        'limit_value': float(limit.limit_value),
                        'current_value': float(limit.current_value),
                        'utilization_percent': limit.utilization_percent,
                        'is_breached': limit.is_breached,
                        'warning_threshold': limit.warning_threshold
                    }
                    for name, limit in self.risk_limits.items()
                },
                'positions': {
                    symbol: self.calculate_position_risk_metrics(symbol)
                    for symbol in self.positions.keys()
                },
                'daily_stats': {
                    'daily_pnl': float(self.daily_pnl),
                    'daily_trades': self.daily_trades,
                    'daily_return_pct': self.daily_returns[-1] * 100 if self.daily_returns else 0.0
                },
                'overall_risk_score': self._calculate_overall_risk_score()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Risk engine health check"""
        try:
            with self._lock:
                # Check for any breached limits
                breached_limits = [
                    name for name, limit in self.risk_limits.items() 
                    if limit.is_breached
                ]
                
                # Calculate health score
                if breached_limits:
                    health_score = 0.0
                    status = 'critical'
                    message = f"Risk limits breached: {', '.join(breached_limits)}"
                else:
                    risk_score = self._calculate_overall_risk_score()
                    health_score = 1.0 - risk_score
                    
                    if health_score >= 0.8:
                        status = 'healthy'
                        message = "All risk metrics within limits"
                    elif health_score >= 0.6:
                        status = 'degraded'
                        message = "Elevated risk levels detected"
                    else:
                        status = 'critical'
                        message = "High risk levels detected"
                
                return {
                    'status': status,
                    'message': message,
                    'metrics': {
                        'health_score': health_score,
                        'risk_score': self._calculate_overall_risk_score(),
                        'positions_count': len(self.positions),
                        'daily_pnl': float(self.daily_pnl),
                        'daily_trades': self.daily_trades,
                        'breached_limits': breached_limits,
                        'drawdown_percent': self._calculate_current_drawdown_percent()
                    }
                }
                
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Risk engine health check failed: {str(e)}',
                'metrics': {'error': str(e)}
            }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (called at start of new trading day)"""
        with self._lock:
            self.daily_pnl = Decimal('0')
            self.daily_trades = 0
            self.daily_returns.clear()
            self.start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            self.logger.info("Daily risk metrics reset")

# ===================== GLOBAL INSTANCE =====================

_risk_engine = None

def get_risk_engine() -> RiskEngine:
    """Get global risk engine instance"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine()
    return _risk_engine

async def initialize_risk_engine():
    """Initialize risk engine"""
    risk_engine = get_risk_engine()
    
    # Register health check
    from infrastructure.monitoring import get_health_monitor
    health_monitor = get_health_monitor()
    health_monitor.register_health_check('risk_engine', risk_engine.health_check)
    
    return risk_engine

"""
Strategy Models - Data Models for Strategy Engine
Step 8: World's #1 Strategy Engine
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

class StrategyType(Enum):
    """Strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    ML_BASED = "ml_based"
    MICROSTRUCTURE = "microstructure"
    SENTIMENT = "sentiment"
    HYBRID = "hybrid"
    QUANTUM = "quantum"

class StrategyStatus(Enum):
    """Strategy status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISABLED = "disabled"

class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalSource(Enum):
    """Signal sources"""
    TECHNICAL = "technical"
    ML = "ml"
    ALTERNATIVE_DATA = "alternative_data"
    SENTIMENT = "sentiment"
    QUANTUM = "quantum"
    AGGREGATED = "aggregated"

@dataclass
class Signal:
    """Signal model"""
    id: str
    symbol: str
    signal_type: SignalType
    source: SignalSource
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingSignal:
    """Trading signal model"""
    id: str
    symbol: str
    signal_type: SignalType
    strength: float
    confidence: float
    timestamp: datetime
    source: SignalSource
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Strategy:
    """Strategy model"""
    id: str
    name: str
    description: str
    strategy_type: StrategyType
    status: StrategyStatus
    symbols: List[str]
    technical_indicators: List[str] = field(default_factory=list)
    ml_models: List[str] = field(default_factory=list)
    alternative_data_sources: List[str] = field(default_factory=list)
    sentiment_analysis: bool = False
    quantum_computing: bool = False
    risk_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_frequency: int = 60  # seconds
    initial_capital: float = 10000.0
    volatility_threshold: Optional[float] = None
    liquidity_threshold: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPerformance:
    """Strategy performance model"""
    strategy_id: str
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class StrategyDiscovery:
    """Strategy discovery model"""
    id: str
    name: str
    description: str
    strategy_type: StrategyType
    symbols: List[str]
    parameters: Dict[str, Any]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class StrategyEvolution:
    """Strategy evolution model"""
    id: str
    parent_strategy_id: str
    name: str
    description: str
    strategy_type: StrategyType
    symbols: List[str]
    parameters: Dict[str, Any]
    fitness_score: float
    generation: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BacktestResult:
    """Backtest result model"""
    strategy_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    volatility: float
    beta: float
    alpha: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

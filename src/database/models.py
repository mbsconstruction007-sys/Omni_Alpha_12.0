"""
World-class database models for algorithmic trading
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
from pydantic import BaseModel, Field, ConfigDict, field_validator
import orjson

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    
class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    
class AssetType(str, Enum):
    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"
    
class ExchangeType(str, Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    AMEX = "amex"
    CBOE = "cboe"
    CME = "cme"
    BINANCE = "binance"
    COINBASE = "coinbase"

# Base model with common fields
class BaseTradeModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v),
        },
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
    )
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def model_dump_json(self, **kwargs) -> str:
        """Fast JSON serialization with orjson"""
        data = self.model_dump(**kwargs)
        # Convert Decimal to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        data = convert_decimals(data)
        return orjson.dumps(data).decode()

# Order model
class Order(BaseTradeModel):
    """High-performance order model"""
    
    # Order identification
    order_id: str = Field(..., description="Unique order identifier")
    client_order_id: Optional[str] = Field(None, description="Client-side order ID")
    parent_order_id: Optional[str] = Field(None, description="Parent order for multi-leg")
    account_id: str = Field(..., description="Trading account ID")
    
    # Asset information
    symbol: str = Field(..., description="Trading symbol")
    asset_type: AssetType
    exchange: ExchangeType
    
    # Order details
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(..., gt=0)
    filled_quantity: Decimal = Field(default=Decimal("0"))
    remaining_quantity: Decimal = Field(default=Decimal("0"))
    
    # Pricing
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    
    # Timing
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    status_message: Optional[str] = None
    
    # Risk management
    max_slippage: Optional[Decimal] = Field(None, description="Maximum allowed slippage")
    position_size_pct: Optional[Decimal] = Field(None, description="% of portfolio")
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    
    # Execution details
    commission: Decimal = Field(default=Decimal("0"))
    fees: Decimal = Field(default=Decimal("0"))
    slippage: Decimal = Field(default=Decimal("0"))
    
    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('remaining_quantity', mode='before')
    def calculate_remaining(cls, v, values):
        if 'quantity' in values and 'filled_quantity' in values:
            return values['quantity'] - values['filled_quantity']
        return v

# Trade/Execution model
class Trade(BaseTradeModel):
    """Individual trade execution"""
    
    trade_id: str = Field(..., description="Unique trade identifier")
    order_id: str = Field(..., description="Associated order ID")
    account_id: str = Field(..., description="Trading account ID")
    
    # Execution details
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    
    # Costs
    commission: Decimal = Field(default=Decimal("0"))
    fees: Decimal = Field(default=Decimal("0"))
    
    # Timing
    executed_at: datetime
    settled_at: Optional[datetime] = None
    
    # Venue
    exchange: ExchangeType
    liquidity_indicator: Optional[str] = None  # Add/Remove liquidity
    
    # Metadata
    execution_id: Optional[str] = None
    clearing_firm: Optional[str] = None

# Position model
class Position(BaseTradeModel):
    """Portfolio position tracking"""
    
    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account_id: str
    
    # Asset
    symbol: str
    asset_type: AssetType
    
    # Quantities
    quantity: Decimal
    available_quantity: Decimal  # Available for trading
    locked_quantity: Decimal = Field(default=Decimal("0"))  # In open orders
    
    # Pricing
    average_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    
    # P&L
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Field(default=Decimal("0"))
    total_pnl: Decimal
    pnl_percentage: Decimal
    
    # Risk metrics
    position_risk_score: Optional[float] = None
    var_95: Optional[Decimal] = None  # Value at Risk
    expected_shortfall: Optional[Decimal] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    opened_at: datetime
    last_modified: datetime

# Market data models
class MarketTick(BaseTradeModel):
    """High-frequency market tick data"""
    
    symbol: str
    timestamp: datetime
    
    # Pricing
    bid: Decimal
    ask: Decimal
    last: Decimal
    mid: Optional[Decimal] = None
    
    # Volume
    bid_size: int
    ask_size: int
    last_size: Optional[int] = None
    
    # Additional
    exchange: Optional[ExchangeType] = None
    conditions: List[str] = Field(default_factory=list)
    
    class Config:
        # Optimize for high-frequency data
        arbitrary_types_allowed = True

class OHLCV(BaseTradeModel):
    """OHLCV candlestick data"""
    
    symbol: str
    timeframe: str  # 1m, 5m, 15m, 1h, 1d, etc.
    
    # OHLCV
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    
    # Additional metrics
    vwap: Optional[Decimal] = None
    trade_count: Optional[int] = None
    
    # Timing
    period_start: datetime
    period_end: datetime

# Strategy models
class Strategy(BaseTradeModel):
    """Trading strategy configuration"""
    
    strategy_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    
    # Configuration
    enabled: bool = True
    parameters: Dict[str, Any]
    
    # Risk limits
    max_position_size: Decimal
    max_daily_loss: Decimal
    max_drawdown: Decimal
    position_limit: int
    
    # Performance
    total_pnl: Decimal = Field(default=Decimal("0"))
    win_rate: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)
    max_drawdown_pct: float = Field(default=0.0)
    
    # Execution
    symbols: List[str]
    asset_types: List[AssetType]
    
    # Status
    status: str = "active"  # active, paused, stopped
    last_signal_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None

# Signal model
class TradingSignal(BaseTradeModel):
    """Trading signal from strategy"""
    
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    
    # Signal details
    symbol: str
    action: OrderSide
    signal_strength: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    confidence: float = Field(ge=0.0, le=1.0)  # 0 to 1
    
    # Recommended order
    suggested_quantity: Decimal
    suggested_price: Optional[Decimal] = None
    suggested_stop_loss: Optional[Decimal] = None
    suggested_take_profit: Optional[Decimal] = None
    
    # Timing
    valid_until: Optional[datetime] = None
    
    # Analysis
    indicators: Dict[str, float] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    
    # Execution
    executed: bool = False
    order_id: Optional[str] = None
    execution_time: Optional[datetime] = None

# Account model
class Account(BaseTradeModel):
    """Trading account information"""
    
    account_id: str
    account_type: str  # cash, margin, portfolio
    
    # Balances
    cash_balance: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    
    # Margin (if applicable)
    margin_used: Decimal = Field(default=Decimal("0"))
    margin_available: Decimal = Field(default=Decimal("0"))
    maintenance_margin: Decimal = Field(default=Decimal("0"))
    
    # Risk metrics
    daily_pnl: Decimal = Field(default=Decimal("0"))
    total_pnl: Decimal = Field(default=Decimal("0"))
    risk_score: float = Field(default=0.0)
    
    # Limits
    day_trade_count: int = Field(default=0)
    pattern_day_trader: bool = Field(default=False)
    
    # Status
    active: bool = True
    restricted: bool = False
    restriction_reason: Optional[str] = None

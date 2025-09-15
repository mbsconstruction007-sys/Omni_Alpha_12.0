"""
Order Management System - Data Models
Handles all order types, states, and execution details
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator
import uuid

class OrderType(str, Enum):
    """Order types supported by the system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"

class OrderSide(str, Enum):
    """Order side (direction)"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"

class OrderStatus(str, Enum):
    """Order lifecycle states"""
    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING_REPLACE = "pending_replace"
    REPLACED = "replaced"

class TimeInForce(str, Enum):
    """Order time in force"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTX = "gtx"  # Good Till Extended
    OPG = "opg"  # At the Opening
    CLS = "cls"  # At the Close

class ExecutionVenue(str, Enum):
    """Execution venues"""
    ALPACA = "alpaca"
    UPSTOX = "upstox"
    SMART = "smart"  # Smart routing
    PRIMARY = "primary"
    DARK_POOL = "dark_pool"

class Order(BaseModel):
    """Complete order model"""
    # Identification
    order_id: str = Field(default_factory=lambda: f"OMNI5_{uuid.uuid4().hex[:8]}")
    client_order_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    
    # Basic Details
    symbol: str
    asset_class: str = "equity"
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    filled_quantity: Decimal = Field(default=Decimal("0"))
    remaining_quantity: Decimal = Field(default=Decimal("0"))
    
    # Order Type & Pricing
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trail_amount: Optional[Decimal] = None
    trail_percent: Optional[Decimal] = None
    
    # Execution Details
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False
    venue: ExecutionVenue = ExecutionVenue.SMART
    
    # Status & Timestamps
    status: OrderStatus = OrderStatus.PENDING_NEW
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Execution Results
    average_fill_price: Optional[Decimal] = None
    last_fill_price: Optional[Decimal] = None
    last_fill_quantity: Optional[Decimal] = None
    commission: Decimal = Field(default=Decimal("0"))
    slippage: Optional[Decimal] = None
    
    # Risk & Compliance
    risk_check_passed: bool = False
    risk_check_details: Dict[str, Any] = Field(default_factory=dict)
    compliance_check_passed: bool = True
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('remaining_quantity', always=True)
    def calculate_remaining(cls, v, values):
        """Calculate remaining quantity"""
        if 'quantity' in values and 'filled_quantity' in values:
            return values['quantity'] - values['filled_quantity']
        return v
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        """Validate limit price for limit orders"""
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Limit price required for limit orders")
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Validate stop price for stop orders"""
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Stop price required for stop orders")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }

class OrderRequest(BaseModel):
    """API request for creating an order"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OrderUpdate(BaseModel):
    """Order update/modification request"""
    quantity: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: Optional[TimeInForce] = None

class Fill(BaseModel):
    """Trade fill information"""
    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    venue: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    commission: Decimal = Field(default=Decimal("0"))
    liquidity: str = "taker"  # maker/taker
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Position(BaseModel):
    """Position tracking model"""
    symbol: str
    quantity: Decimal
    side: str  # long/short
    average_entry_price: Decimal
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Field(default=Decimal("0"))
    cost_basis: Decimal
    updated_at: datetime = Field(default_factory=datetime.utcnow)

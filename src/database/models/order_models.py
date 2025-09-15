"""
Database models for Order Management System
SQLAlchemy models for order persistence
"""

from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, 
    DateTime, JSON, Enum, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class OrderStatusDB(enum.Enum):
    """Order status enum for database"""
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

class OrderTypeDB(enum.Enum):
    """Order type enum for database"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"
    ICEBERG = "iceberg"

class OrderSideDB(enum.Enum):
    """Order side enum for database"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"

class TimeInForceDB(enum.Enum):
    """Time in force enum for database"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTX = "gtx"
    OPG = "opg"
    CLS = "cls"

class OrderModel(Base):
    """Order database model"""
    __tablename__ = "orders"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Order identifiers
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    client_order_id = Column(String(100), unique=True, nullable=False)
    parent_order_id = Column(String(50), index=True)
    strategy_id = Column(String(50), index=True)
    user_id = Column(String(50), index=True)
    
    # Basic details
    symbol = Column(String(20), nullable=False, index=True)
    asset_class = Column(String(20), default="equity")
    side = Column(Enum(OrderSideDB), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    filled_quantity = Column(Numeric(20, 8), default=0)
    remaining_quantity = Column(Numeric(20, 8), default=0)
    
    # Order type and pricing
    order_type = Column(Enum(OrderTypeDB), nullable=False)
    limit_price = Column(Numeric(20, 8))
    stop_price = Column(Numeric(20, 8))
    trail_amount = Column(Numeric(20, 8))
    trail_percent = Column(Numeric(10, 4))
    
    # Execution details
    time_in_force = Column(Enum(TimeInForceDB), default=TimeInForceDB.DAY)
    extended_hours = Column(Boolean, default=False)
    venue = Column(String(20), default="smart")
    
    # Status and timestamps
    status = Column(Enum(OrderStatusDB), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    expired_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Execution results
    average_fill_price = Column(Numeric(20, 8))
    last_fill_price = Column(Numeric(20, 8))
    last_fill_quantity = Column(Numeric(20, 8))
    commission = Column(Numeric(20, 8), default=0)
    slippage = Column(Numeric(20, 8))
    
    # Risk and compliance
    risk_check_passed = Column(Boolean, default=False)
    risk_check_details = Column(JSON)
    compliance_check_passed = Column(Boolean, default=True)
    
    # Metadata
    tags = Column(JSON)
    notes = Column(String(500))
    order_metadata = Column(JSON)
    
    # Relationships
    fills = relationship("FillModel", back_populates="order", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_order_user_status', 'user_id', 'status'),
        Index('idx_order_symbol_status', 'symbol', 'status'),
        Index('idx_order_created_at', 'created_at'),
        Index('idx_order_strategy', 'strategy_id', 'status'),
    )

class FillModel(Base):
    """Fill/execution database model"""
    __tablename__ = "fills"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Fill identifiers
    fill_id = Column(String(50), unique=True, nullable=False, index=True)
    order_id = Column(String(50), ForeignKey('orders.order_id'), nullable=False)
    
    # Fill details
    symbol = Column(String(20), nullable=False)
    side = Column(Enum(OrderSideDB), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    venue = Column(String(20))
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Costs
    commission = Column(Numeric(20, 8), default=0)
    fees = Column(Numeric(20, 8), default=0)
    
    # Additional info
    liquidity = Column(String(10), default="taker")  # maker/taker
    fill_metadata = Column(JSON)
    
    # Relationships
    order = relationship("OrderModel", back_populates="fills")
    
    # Indexes
    __table_args__ = (
        Index('idx_fill_order', 'order_id'),
        Index('idx_fill_timestamp', 'timestamp'),
        Index('idx_fill_symbol', 'symbol', 'timestamp'),
    )

class PositionModel(Base):
    """Position tracking database model"""
    __tablename__ = "positions"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Position identifiers
    user_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    quantity = Column(Numeric(20, 8), nullable=False)
    side = Column(String(10), nullable=False)  # long/short
    average_entry_price = Column(Numeric(20, 8), nullable=False)
    
    # Market values
    current_price = Column(Numeric(20, 8))
    market_value = Column(Numeric(20, 8))
    
    # P&L
    unrealized_pnl = Column(Numeric(20, 8))
    realized_pnl = Column(Numeric(20, 8), default=0)
    
    # Cost basis
    cost_basis = Column(Numeric(20, 8), nullable=False)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Metadata
    position_metadata = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_position_user_symbol', 'user_id', 'symbol', unique=True),
        Index('idx_position_updated', 'updated_at'),
    )

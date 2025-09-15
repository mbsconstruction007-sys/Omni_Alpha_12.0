"""
Base Strategy and Signal classes
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field
from enum import Enum


class SignalAction(str, Enum):
    """Signal action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class Signal(BaseModel):
    """Trading signal model"""
    
    # Basic signal info
    symbol: str
    action: SignalAction
    strength: float = Field(ge=0, le=100, description="Signal strength 0-100")
    confidence: float = Field(ge=0, le=1, description="Signal confidence 0-1")
    
    # Price levels
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expiry: Optional[datetime] = None
    
    # Market context
    regime_alignment: float = Field(ge=0, le=1, default=0.5, description="Alignment with market regime")
    psychology_score: float = Field(ge=0, le=1, default=0.5, description="Psychology-based score")
    
    # Technical indicators
    indicators: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    strategy_name: str = "unknown"
    source: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }


class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = True
        self.signals_generated = 0
        self.last_signal_time = None
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Generate a trading signal based on market data"""
        raise NotImplementedError("Subclasses must implement generate_signal")
    
    async def update(self, market_data: Dict[str, Any]):
        """Update strategy with new market data"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'signals_generated': self.signals_generated,
            'last_signal_time': self.last_signal_time
        }
    
    def activate(self):
        """Activate the strategy"""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate the strategy"""
        self.is_active = False

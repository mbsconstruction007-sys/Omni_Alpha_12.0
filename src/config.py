"""
Configuration Management
Loads and validates all configuration from environment
"""

from pydantic import BaseSettings, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Trading Engine Core
    TRADING_ENGINE_ENABLED: bool = Field(default=True)
    TRADING_ENGINE_MODE: str = Field(default="paper")
    ENGINE_START_TIME: str = Field(default="09:30:00")
    ENGINE_STOP_TIME: str = Field(default="16:00:00")
    ENGINE_TIMEZONE: str = Field(default="America/New_York")
    ENGINE_LOG_LEVEL: str = Field(default="INFO")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")
    
    # Database
    DATABASE_URL: str = Field(default="postgresql://user:pass@localhost:5432/trading")
    REDIS_URL: str = Field(default="redis://localhost:6379")
    
    # Strategies
    ACTIVE_STRATEGIES: List[str] = Field(
        default=["momentum_breakout", "mean_reversion", "smart_money_flow"]
    )
    POSITION_SIZING_METHOD: str = Field(default="kelly_criterion")
    
    # Risk Management
    MAX_DAILY_LOSS_PERCENT: float = Field(default=2.0)
    MAX_DRAWDOWN_PERCENT: float = Field(default=15.0)
    RISK_PER_TRADE_PERCENT: float = Field(default=1.0)
    MAX_OPEN_POSITIONS: int = Field(default=20)
    
    # Market Data
    MARKET_DATA_PROVIDER: str = Field(default="alpaca")
    REAL_TIME_DATA_ENABLED: bool = Field(default=True)
    
    # Broker Configuration
    ALPACA_API_KEY: Optional[str] = Field(default=None)
    ALPACA_SECRET_KEY: Optional[str] = Field(default=None)
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets")
    
    # Machine Learning
    ML_MODELS_ENABLED: bool = Field(default=True)
    ML_CONFIDENCE_THRESHOLD: float = Field(default=0.65)
    
    # Crisis Management
    CRISIS_DETECTION_ENABLED: bool = Field(default=True)
    CRISIS_VIX_THRESHOLD: float = Field(default=40)
    
    # Notifications
    ALERTS_ENABLED: bool = Field(default=True)
    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None)
    
    # Development
    DEV_MODE: bool = Field(default=False)
    
    # CORS
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000"])
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator("ACTIVE_STRATEGIES", pre=True)
    def parse_strategies(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    def dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key) 
            for key in self.__fields__.keys()
        }

# Create global settings instance
settings = Settings()

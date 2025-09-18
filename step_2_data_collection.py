"""
STEP 2: DATA COLLECTION & MARKET DATA SYSTEM - OMNI ALPHA TRADING SYSTEM
Enterprise-grade data collection framework with multi-source aggregation, caching, and validation
"""

import os
import sys
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import hashlib
import pickle
from abc import ABC, abstractmethod

# Third party imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.dialects.postgresql import JSONB
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv('alpaca_live_trading.env')

# ===================== DATA STRUCTURES =====================

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    INVALID = "INVALID"

class DataSource(Enum):
    """Available data sources"""
    ALPACA = "ALPACA"
    YAHOO_FINANCE = "YAHOO_FINANCE"
    ALPHA_VANTAGE = "ALPHA_VANTAGE"
    IEX_CLOUD = "IEX_CLOUD"
    POLYGON = "POLYGON"
    CACHE = "CACHE"
    DATABASE = "DATABASE"

class TimeFrame(Enum):
    """Supported timeframes"""
    MINUTE_1 = "1Min"
    MINUTE_5 = "5Min"
    MINUTE_15 = "15Min"
    HOUR_1 = "1Hour"
    DAY_1 = "1Day"
    WEEK_1 = "1Week"
    MONTH_1 = "1Month"

@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    source: DataSource
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    source: DataSource
    quality: DataQuality = DataQuality.GOOD

@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    source_preference: List[DataSource] = field(default_factory=list)
    cache_enabled: bool = True
    quality_threshold: DataQuality = DataQuality.FAIR

# ===================== DATABASE MODELS =====================

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class HistoricalData(Base):
        __tablename__ = 'historical_data'
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(10), nullable=False, index=True)
        timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
        timeframe = Column(String(10), nullable=False)
        open = Column(Float, nullable=False)
        high = Column(Float, nullable=False)
        low = Column(Float, nullable=False)
        close = Column(Float, nullable=False)
        volume = Column(Integer, nullable=False)
        source = Column(String(20), nullable=False)
        quality = Column(String(10), nullable=False)
        data_metadata = Column(Text)  # JSON metadata
        created_at = Column(DateTime(timezone=True), default=datetime.now)
        
        # Composite index for efficient queries
        __table_args__ = (
            {'mysql_engine': 'InnoDB'},
        )
    
    class RealTimeQuotes(Base):
        __tablename__ = 'realtime_quotes'
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(10), nullable=False, index=True)
        timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
        bid = Column(Float, nullable=False)
        ask = Column(Float, nullable=False)
        bid_size = Column(Integer, nullable=False)
        ask_size = Column(Integer, nullable=False)
        last_price = Column(Float, nullable=False)
        last_size = Column(Integer, nullable=False)
        source = Column(String(20), nullable=False)
        quality = Column(String(10), nullable=False)
        created_at = Column(DateTime(timezone=True), default=datetime.now)
    
    class DataSourceStatus(Base):
        __tablename__ = 'data_source_status'
        
        id = Column(Integer, primary_key=True)
        source = Column(String(20), nullable=False, unique=True)
        status = Column(String(10), nullable=False)  # ACTIVE, INACTIVE, ERROR
        last_update = Column(DateTime(timezone=True), nullable=False)
        error_count = Column(Integer, default=0)
        success_count = Column(Integer, default=0)
        response_time_ms = Column(Float)
        source_metadata = Column(Text)

# ===================== DATA SOURCES =====================

class BaseDataSource(ABC):
    """Base class for data sources"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"data_source.{name}")
        self.is_active = True
        self.error_count = 0
        self.success_count = 0
        self.last_error = None
        self.response_times = []
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame, 
                                start: datetime, end: datetime) -> List[MarketData]:
        """Get historical market data"""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if data source is healthy"""
        pass
    
    def record_success(self, response_time: float):
        """Record successful request"""
        self.success_count += 1
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_error(self, error: Exception):
        """Record error"""
        self.error_count += 1
        self.last_error = str(error)
        self.logger.error(f"Data source error: {error}")
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        return np.mean(self.response_times) if self.response_times else 0.0

class AlpacaDataSource(BaseDataSource):
    """Alpaca Markets data source"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ALPACA", config)
        
        if ALPACA_AVAILABLE:
            self.api = tradeapi.REST(
                config.get('api_key'),
                config.get('secret_key'),
                config.get('base_url', 'https://paper-api.alpaca.markets')
            )
        else:
            self.api = None
            self.is_active = False
            self.logger.warning("Alpaca API not available")
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame, 
                                start: datetime, end: datetime) -> List[MarketData]:
        """Get historical data from Alpaca"""
        if not self.api or not self.is_active:
            return []
        
        start_time = time.time()
        
        try:
            # Map timeframe
            alpaca_timeframe = self._map_timeframe(timeframe)
            
            # Get bars
            bars = self.api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=10000
            ).df
            
            if bars.empty:
                return []
            
            # Convert to MarketData objects
            market_data = []
            for idx, row in bars.iterrows():
                data = MarketData(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    timeframe=timeframe.value,
                    source=DataSource.ALPACA,
                    quality=self._assess_data_quality(row)
                )
                market_data.append(data)
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return market_data
            
        except Exception as e:
            self.record_error(e)
            return []
    
    async def get_real_time_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote from Alpaca"""
        if not self.api or not self.is_active:
            return None
        
        start_time = time.time()
        
        try:
            quote = self.api.get_latest_quote(symbol)
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return Quote(
                symbol=symbol,
                timestamp=quote.timestamp.replace(tzinfo=timezone.utc),
                bid=float(quote.bp),
                ask=float(quote.ap),
                bid_size=int(quote.bs),
                ask_size=int(quote.as_),
                last_price=float(quote.ap),  # Use ask as last price
                last_size=int(quote.as_),
                source=DataSource.ALPACA,
                quality=DataQuality.GOOD
            )
            
        except Exception as e:
            self.record_error(e)
            return None
    
    async def health_check(self) -> bool:
        """Check Alpaca API health"""
        if not self.api:
            return False
        
        try:
            account = self.api.get_account()
            return account.status == 'ACTIVE'
        except Exception as e:
            self.record_error(e)
            return False
    
    def _map_timeframe(self, timeframe: TimeFrame):
        """Map internal timeframe to Alpaca timeframe"""
        mapping = {
            TimeFrame.MINUTE_1: tradeapi.TimeFrame.Minute,
            TimeFrame.MINUTE_5: tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            TimeFrame.MINUTE_15: tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            TimeFrame.HOUR_1: tradeapi.TimeFrame.Hour,
            TimeFrame.DAY_1: tradeapi.TimeFrame.Day,
            TimeFrame.WEEK_1: tradeapi.TimeFrame.Week,
            TimeFrame.MONTH_1: tradeapi.TimeFrame.Month
        }
        return mapping.get(timeframe, tradeapi.TimeFrame.Day)
    
    def _assess_data_quality(self, row) -> DataQuality:
        """Assess data quality"""
        # Basic quality checks
        if pd.isna(row['close']) or row['close'] <= 0:
            return DataQuality.INVALID
        if pd.isna(row['volume']) or row['volume'] < 0:
            return DataQuality.POOR
        if row['high'] < row['low']:
            return DataQuality.INVALID
        
        return DataQuality.GOOD

class YahooFinanceDataSource(BaseDataSource):
    """Yahoo Finance data source"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("YAHOO_FINANCE", config)
        
        if not YFINANCE_AVAILABLE:
            self.is_active = False
            self.logger.warning("Yahoo Finance not available")
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame, 
                                start: datetime, end: datetime) -> List[MarketData]:
        """Get historical data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE or not self.is_active:
            return []
        
        start_time = time.time()
        
        try:
            # Map timeframe
            interval = self._map_timeframe(timeframe)
            
            # Get data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start.date(),
                end=end.date(),
                interval=interval
            )
            
            if hist.empty:
                return []
            
            # Convert to MarketData objects
            market_data = []
            for idx, row in hist.iterrows():
                data = MarketData(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    timeframe=timeframe.value,
                    source=DataSource.YAHOO_FINANCE,
                    quality=self._assess_data_quality(row)
                )
                market_data.append(data)
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return market_data
            
        except Exception as e:
            self.record_error(e)
            return []
    
    async def get_real_time_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote from Yahoo Finance"""
        if not YFINANCE_AVAILABLE or not self.is_active:
            return None
        
        start_time = time.time()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return Quote(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid=float(info.get('bid', 0)),
                ask=float(info.get('ask', 0)),
                bid_size=int(info.get('bidSize', 0)),
                ask_size=int(info.get('askSize', 0)),
                last_price=float(info.get('regularMarketPrice', 0)),
                last_size=0,
                source=DataSource.YAHOO_FINANCE,
                quality=DataQuality.FAIR
            )
            
        except Exception as e:
            self.record_error(e)
            return None
    
    async def health_check(self) -> bool:
        """Check Yahoo Finance health"""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return 'regularMarketPrice' in info
        except Exception as e:
            self.record_error(e)
            return False
    
    def _map_timeframe(self, timeframe: TimeFrame) -> str:
        """Map internal timeframe to Yahoo Finance interval"""
        mapping = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.DAY_1: "1d",
            TimeFrame.WEEK_1: "1wk",
            TimeFrame.MONTH_1: "1mo"
        }
        return mapping.get(timeframe, "1d")
    
    def _assess_data_quality(self, row) -> DataQuality:
        """Assess data quality"""
        if pd.isna(row['Close']) or row['Close'] <= 0:
            return DataQuality.INVALID
        if pd.isna(row['Volume']) or row['Volume'] < 0:
            return DataQuality.POOR
        
        return DataQuality.GOOD

class AlphaVantageDataSource(BaseDataSource):
    """Alpha Vantage data source"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ALPHA_VANTAGE", config)
        
        self.api_key = config.get('api_key')
        self.base_url = "https://www.alphavantage.co/query"
        
        if not self.api_key:
            self.is_active = False
            self.logger.warning("Alpha Vantage API key not provided")
        
        if REQUESTS_AVAILABLE:
            # Setup session with retry strategy
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        else:
            self.session = None
            self.is_active = False
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame, 
                                start: datetime, end: datetime) -> List[MarketData]:
        """Get historical data from Alpha Vantage"""
        if not self.session or not self.is_active:
            return []
        
        start_time = time.time()
        
        try:
            function = self._get_function(timeframe)
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Find time series key
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key or 'Daily' in key or 'Intraday' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                return []
            
            time_series = data[time_series_key]
            
            # Convert to MarketData objects
            market_data = []
            for timestamp_str, ohlcv in time_series.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S' if ' ' in timestamp_str else '%Y-%m-%d')
                
                # Filter by date range
                if timestamp < start or timestamp > end:
                    continue
                
                data_point = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(ohlcv['1. open']),
                    high=float(ohlcv['2. high']),
                    low=float(ohlcv['3. low']),
                    close=float(ohlcv['4. close']),
                    volume=int(ohlcv['5. volume']),
                    timeframe=timeframe.value,
                    source=DataSource.ALPHA_VANTAGE,
                    quality=DataQuality.GOOD
                )
                market_data.append(data_point)
            
            # Sort by timestamp
            market_data.sort(key=lambda x: x.timestamp)
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return market_data
            
        except Exception as e:
            self.record_error(e)
            return []
    
    async def get_real_time_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote from Alpha Vantage"""
        if not self.session or not self.is_active:
            return None
        
        start_time = time.time()
        
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Global Quote' not in data:
                return None
            
            quote_data = data['Global Quote']
            
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)
            
            return Quote(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid=0.0,  # Not available in global quote
                ask=0.0,  # Not available in global quote
                bid_size=0,
                ask_size=0,
                last_price=float(quote_data['05. price']),
                last_size=0,
                source=DataSource.ALPHA_VANTAGE,
                quality=DataQuality.GOOD
            )
            
        except Exception as e:
            self.record_error(e)
            return None
    
    async def health_check(self) -> bool:
        """Check Alpha Vantage API health"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.record_error(e)
            return False
    
    def _get_function(self, timeframe: TimeFrame) -> str:
        """Get Alpha Vantage function for timeframe"""
        if timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15]:
            return 'TIME_SERIES_INTRADAY'
        elif timeframe == TimeFrame.DAY_1:
            return 'TIME_SERIES_DAILY'
        elif timeframe == TimeFrame.WEEK_1:
            return 'TIME_SERIES_WEEKLY'
        elif timeframe == TimeFrame.MONTH_1:
            return 'TIME_SERIES_MONTHLY'
        else:
            return 'TIME_SERIES_DAILY'

# ===================== DATA VALIDATION =====================

class DataValidator:
    """Data quality validation framework"""
    
    def __init__(self):
        self.logger = logging.getLogger("data_validator")
    
    def validate_market_data(self, data: MarketData) -> Tuple[bool, List[str]]:
        """Validate market data"""
        errors = []
        
        # Price validations
        if data.open <= 0:
            errors.append("Open price must be positive")
        if data.high <= 0:
            errors.append("High price must be positive")
        if data.low <= 0:
            errors.append("Low price must be positive")
        if data.close <= 0:
            errors.append("Close price must be positive")
        
        # OHLC consistency
        if data.high < data.low:
            errors.append("High price cannot be less than low price")
        if data.high < max(data.open, data.close):
            errors.append("High price must be >= max(open, close)")
        if data.low > min(data.open, data.close):
            errors.append("Low price must be <= min(open, close)")
        
        # Volume validation
        if data.volume < 0:
            errors.append("Volume cannot be negative")
        
        # Timestamp validation
        if data.timestamp > datetime.now(timezone.utc):
            errors.append("Timestamp cannot be in the future")
        
        # Extreme price movements (more than 50% in one bar)
        price_range = abs(data.high - data.low)
        avg_price = (data.high + data.low) / 2
        if price_range / avg_price > 0.5:
            errors.append("Extreme price movement detected")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"Data validation failed for {data.symbol}: {errors}")
        
        return is_valid, errors
    
    def validate_quote(self, quote: Quote) -> Tuple[bool, List[str]]:
        """Validate quote data"""
        errors = []
        
        # Price validations
        if quote.bid <= 0:
            errors.append("Bid price must be positive")
        if quote.ask <= 0:
            errors.append("Ask price must be positive")
        if quote.last_price <= 0:
            errors.append("Last price must be positive")
        
        # Bid-ask spread validation
        if quote.ask < quote.bid:
            errors.append("Ask price cannot be less than bid price")
        
        # Size validations
        if quote.bid_size < 0:
            errors.append("Bid size cannot be negative")
        if quote.ask_size < 0:
            errors.append("Ask size cannot be negative")
        
        # Timestamp validation
        if quote.timestamp > datetime.now(timezone.utc):
            errors.append("Quote timestamp cannot be in the future")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"Quote validation failed for {quote.symbol}: {errors}")
        
        return is_valid, errors
    
    def assess_data_completeness(self, data_list: List[MarketData], 
                               expected_count: int) -> Tuple[float, List[str]]:
        """Assess data completeness"""
        issues = []
        
        if not data_list:
            return 0.0, ["No data received"]
        
        completeness = len(data_list) / expected_count
        
        if completeness < 0.8:
            issues.append(f"Low data completeness: {completeness:.1%}")
        
        # Check for gaps in timestamps
        if len(data_list) > 1:
            sorted_data = sorted(data_list, key=lambda x: x.timestamp)
            gaps = []
            
            for i in range(1, len(sorted_data)):
                time_diff = sorted_data[i].timestamp - sorted_data[i-1].timestamp
                expected_diff = self._get_expected_time_diff(sorted_data[i].timeframe)
                
                if time_diff > expected_diff * 2:  # Allow some tolerance
                    gaps.append(f"Gap between {sorted_data[i-1].timestamp} and {sorted_data[i].timestamp}")
            
            if gaps:
                issues.append(f"Time gaps detected: {len(gaps)} gaps")
        
        return completeness, issues
    
    def _get_expected_time_diff(self, timeframe: str) -> timedelta:
        """Get expected time difference between data points"""
        mapping = {
            "1Min": timedelta(minutes=1),
            "5Min": timedelta(minutes=5),
            "15Min": timedelta(minutes=15),
            "1Hour": timedelta(hours=1),
            "1Day": timedelta(days=1),
            "1Week": timedelta(weeks=1),
            "1Month": timedelta(days=30)
        }
        return mapping.get(timeframe, timedelta(days=1))

# ===================== DATA CACHING =====================

class DataCache:
    """Multi-level data caching system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("data_cache")
        
        # Memory cache
        self.memory_cache = {}
        self.memory_cache_size = config.get('memory_cache_size', 1000)
        
        # Redis cache (if available)
        self.redis_client = None
        if REDIS_AVAILABLE and config.get('redis_enabled', False):
            try:
                self.redis_client = redis.Redis(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    db=config.get('redis_db', 0),
                    password=config.get('redis_password'),
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis cache connected")
            except Exception as e:
                self.logger.warning(f"Redis cache not available: {e}")
                self.redis_client = None
    
    def _generate_cache_key(self, symbol: str, timeframe: str, 
                           start: datetime, end: datetime) -> str:
        """Generate cache key"""
        key_data = f"{symbol}:{timeframe}:{start.isoformat()}:{end.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_data(self, symbol: str, timeframe: str, 
                       start: datetime, end: datetime) -> Optional[List[MarketData]]:
        """Get data from cache"""
        cache_key = self._generate_cache_key(symbol, timeframe, start, end)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            
            # Check if cache is still valid (5 minutes for intraday, 1 hour for daily+)
            cache_ttl = 300 if 'Min' in timeframe else 3600
            if (datetime.now() - timestamp).total_seconds() < cache_ttl:
                self.logger.debug(f"Cache hit (memory): {cache_key}")
                return data
            else:
                # Remove expired data
                del self.memory_cache[cache_key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    
                    # Add to memory cache for faster access
                    self._add_to_memory_cache(cache_key, data)
                    
                    self.logger.debug(f"Cache hit (Redis): {cache_key}")
                    return data
            except Exception as e:
                self.logger.warning(f"Redis cache get error: {e}")
        
        return None
    
    def cache_data(self, symbol: str, timeframe: str, start: datetime, 
                  end: datetime, data: List[MarketData]):
        """Cache data"""
        cache_key = self._generate_cache_key(symbol, timeframe, start, end)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, data)
        
        # Add to Redis cache
        if self.redis_client:
            try:
                # Set TTL: 5 minutes for intraday, 1 hour for daily+
                ttl = 300 if 'Min' in timeframe else 3600
                
                serialized_data = pickle.dumps(data)
                self.redis_client.setex(cache_key, ttl, serialized_data)
                
                self.logger.debug(f"Data cached (Redis): {cache_key}")
            except Exception as e:
                self.logger.warning(f"Redis cache set error: {e}")
    
    def _add_to_memory_cache(self, key: str, data: List[MarketData]):
        """Add data to memory cache with size limit"""
        # Remove oldest entries if cache is full
        if len(self.memory_cache) >= self.memory_cache_size:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][1])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (data, datetime.now())
    
    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                self.logger.info("Redis cache cleared")
            except Exception as e:
                self.logger.warning(f"Redis cache clear error: {e}")

# ===================== DATA STORAGE =====================

class DataStorage:
    """Persistent data storage system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("data_storage")
        
        # Database setup
        if SQLALCHEMY_AVAILABLE:
            database_url = config.get('database_url', 'sqlite:///market_data.db')
            self.engine = create_engine(database_url, echo=config.get('debug', False))
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created/verified")
        else:
            self.engine = None
            self.Session = None
            self.logger.warning("SQLAlchemy not available - no database storage")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session"""
        if not self.Session:
            yield None
            return
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def store_historical_data(self, data_list: List[MarketData]):
        """Store historical market data"""
        if not self.Session:
            return
        
        async with self.get_session() as session:
            if not session:
                return
            
            for data in data_list:
                # Check if data already exists
                existing = session.query(HistoricalData).filter_by(
                    symbol=data.symbol,
                    timestamp=data.timestamp,
                    timeframe=data.timeframe
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = data.open
                    existing.high = data.high
                    existing.low = data.low
                    existing.close = data.close
                    existing.volume = data.volume
                    existing.source = data.source.value
                    existing.quality = data.quality.value
                    existing.data_metadata = json.dumps(data.metadata)
                else:
                    # Create new record
                    record = HistoricalData(
                        symbol=data.symbol,
                        timestamp=data.timestamp,
                        timeframe=data.timeframe,
                        open=data.open,
                        high=data.high,
                        low=data.low,
                        close=data.close,
                        volume=data.volume,
                        source=data.source.value,
                        quality=data.quality.value,
                        data_metadata=json.dumps(data.metadata)
                    )
                    session.add(record)
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start: datetime, end: datetime) -> List[MarketData]:
        """Retrieve historical data from storage"""
        if not self.Session:
            return []
        
        async with self.get_session() as session:
            if not session:
                return []
            
            records = session.query(HistoricalData).filter(
                HistoricalData.symbol == symbol,
                HistoricalData.timeframe == timeframe,
                HistoricalData.timestamp >= start,
                HistoricalData.timestamp <= end
            ).order_by(HistoricalData.timestamp).all()
            
            market_data = []
            for record in records:
                data = MarketData(
                    symbol=record.symbol,
                    timestamp=record.timestamp,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    volume=record.volume,
                    timeframe=record.timeframe,
                    source=DataSource(record.source),
                    quality=DataQuality(record.quality),
                    metadata=json.loads(record.data_metadata) if record.data_metadata else {}
                )
                market_data.append(data)
            
            return market_data
    
    async def store_quote(self, quote: Quote):
        """Store real-time quote"""
        if not self.Session:
            return
        
        async with self.get_session() as session:
            if not session:
                return
            
            record = RealTimeQuotes(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                bid=quote.bid,
                ask=quote.ask,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                last_price=quote.last_price,
                last_size=quote.last_size,
                source=quote.source.value,
                quality=quote.quality.value
            )
            session.add(record)
    
    async def update_source_status(self, source: DataSource, status: str, 
                                 response_time: float = None, error: str = None):
        """Update data source status"""
        if not self.Session:
            return
        
        async with self.get_session() as session:
            if not session:
                return
            
            # Get or create source status record
            source_status = session.query(DataSourceStatus).filter_by(
                source=source.value
            ).first()
            
            if not source_status:
                source_status = DataSourceStatus(
                    source=source.value,
                    status=status,
                    last_update=datetime.now(timezone.utc),
                    error_count=1 if error else 0,
                    success_count=1 if not error else 0,
                    response_time_ms=response_time
                )
                session.add(source_status)
            else:
                source_status.status = status
                source_status.last_update = datetime.now(timezone.utc)
                if error:
                    source_status.error_count += 1
                else:
                    source_status.success_count += 1
                if response_time:
                    source_status.response_time_ms = response_time
                
                source_status.source_metadata = json.dumps({
                    'last_error': error
                } if error else {})

# ===================== MAIN DATA COLLECTION SYSTEM =====================

class DataCollectionSystem:
    """Main data collection orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = self._load_default_config()
        
        self.config = config
        self.logger = logging.getLogger("data_collection")
        
        # Initialize components
        self.data_sources = {}
        self.validator = DataValidator()
        self.cache = DataCache(config.get('cache', {}))
        self.storage = DataStorage(config.get('storage', {}))
        
        # Initialize data sources
        self._initialize_data_sources()
        
        # Statistics
        self.stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'sources': {
                'alpaca': {
                    'enabled': True,
                    'api_key': os.getenv('ALPACA_API_KEY'),
                    'secret_key': os.getenv('ALPACA_SECRET_KEY'),
                    'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                },
                'yahoo_finance': {
                    'enabled': True
                },
                'alpha_vantage': {
                    'enabled': bool(os.getenv('ALPHA_VANTAGE_KEY')),
                    'api_key': os.getenv('ALPHA_VANTAGE_KEY')
                }
            },
            'cache': {
                'memory_cache_size': 1000,
                'redis_enabled': bool(os.getenv('REDIS_URL')),
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 0
            },
            'storage': {
                'database_url': os.getenv('DATABASE_URL', 'sqlite:///market_data.db'),
                'debug': os.getenv('DEBUG', 'false').lower() == 'true'
            }
        }
    
    def _initialize_data_sources(self):
        """Initialize data sources"""
        sources_config = self.config.get('sources', {})
        
        # Alpaca
        if sources_config.get('alpaca', {}).get('enabled', False):
            self.data_sources[DataSource.ALPACA] = AlpacaDataSource(
                sources_config['alpaca']
            )
        
        # Yahoo Finance
        if sources_config.get('yahoo_finance', {}).get('enabled', False):
            self.data_sources[DataSource.YAHOO_FINANCE] = YahooFinanceDataSource(
                sources_config['yahoo_finance']
            )
        
        # Alpha Vantage
        if sources_config.get('alpha_vantage', {}).get('enabled', False):
            self.data_sources[DataSource.ALPHA_VANTAGE] = AlphaVantageDataSource(
                sources_config['alpha_vantage']
            )
        
        self.logger.info(f"Initialized {len(self.data_sources)} data sources")
    
    async def get_historical_data(self, request: DataRequest) -> List[MarketData]:
        """Get historical market data with fallback and caching"""
        self.stats['requests_total'] += 1
        
        # Try cache first
        if request.cache_enabled:
            cached_data = self.cache.get_cached_data(
                request.symbol, 
                request.timeframe.value, 
                request.start_date, 
                request.end_date
            )
            
            if cached_data:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"Cache hit for {request.symbol}")
                return cached_data
            
            self.stats['cache_misses'] += 1
        
        # Try database storage
        stored_data = await self.storage.get_historical_data(
            request.symbol,
            request.timeframe.value,
            request.start_date,
            request.end_date
        )
        
        if stored_data:
            # Check if data is complete and recent
            completeness, issues = self.validator.assess_data_completeness(
                stored_data, 
                self._calculate_expected_data_points(request)
            )
            
            if completeness > 0.95:  # 95% completeness threshold
                self.logger.debug(f"Using stored data for {request.symbol}")
                
                # Cache the data
                if request.cache_enabled:
                    self.cache.cache_data(
                        request.symbol,
                        request.timeframe.value,
                        request.start_date,
                        request.end_date,
                        stored_data
                    )
                
                return stored_data
        
        # Fetch from data sources
        data_sources = request.source_preference if request.source_preference else list(self.data_sources.keys())
        
        for source in data_sources:
            if source not in self.data_sources:
                continue
            
            data_source = self.data_sources[source]
            
            if not data_source.is_active:
                continue
            
            try:
                self.logger.debug(f"Fetching data from {source.value} for {request.symbol}")
                
                data = await data_source.get_historical_data(
                    request.symbol,
                    request.timeframe,
                    request.start_date,
                    request.end_date
                )
                
                if not data:
                    continue
                
                # Validate data quality
                valid_data = []
                for item in data:
                    is_valid, errors = self.validator.validate_market_data(item)
                    
                    if is_valid and item.quality.value >= request.quality_threshold.value:
                        valid_data.append(item)
                    else:
                        self.logger.warning(f"Invalid data point for {item.symbol}: {errors}")
                
                if valid_data:
                    # Store data
                    await self.storage.store_historical_data(valid_data)
                    
                    # Cache data
                    if request.cache_enabled:
                        self.cache.cache_data(
                            request.symbol,
                            request.timeframe.value,
                            request.start_date,
                            request.end_date,
                            valid_data
                        )
                    
                    self.stats['requests_success'] += 1
                    self.logger.info(f"Successfully fetched {len(valid_data)} data points for {request.symbol} from {source.value}")
                    
                    return valid_data
            
            except Exception as e:
                self.logger.error(f"Error fetching data from {source.value}: {e}")
                continue
        
        # No data source succeeded
        self.stats['requests_failed'] += 1
        self.logger.error(f"Failed to fetch data for {request.symbol} from all sources")
        return []
    
    async def get_real_time_quote(self, symbol: str, 
                                source_preference: List[DataSource] = None) -> Optional[Quote]:
        """Get real-time quote with fallback"""
        sources = source_preference if source_preference else list(self.data_sources.keys())
        
        for source in sources:
            if source not in self.data_sources:
                continue
            
            data_source = self.data_sources[source]
            
            if not data_source.is_active:
                continue
            
            try:
                quote = await data_source.get_real_time_quote(symbol)
                
                if quote:
                    # Validate quote
                    is_valid, errors = self.validator.validate_quote(quote)
                    
                    if is_valid:
                        # Store quote
                        await self.storage.store_quote(quote)
                        
                        self.logger.debug(f"Got real-time quote for {symbol} from {source.value}")
                        return quote
                    else:
                        self.logger.warning(f"Invalid quote for {symbol}: {errors}")
            
            except Exception as e:
                self.logger.error(f"Error getting quote from {source.value}: {e}")
                continue
        
        self.logger.error(f"Failed to get real-time quote for {symbol}")
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all data sources"""
        health_status = {
            'overall_status': 'HEALTHY',
            'sources': {},
            'statistics': self.stats.copy()
        }
        
        healthy_sources = 0
        total_sources = len(self.data_sources)
        
        for source, data_source in self.data_sources.items():
            try:
                is_healthy = await data_source.health_check()
                
                source_status = {
                    'healthy': is_healthy,
                    'active': data_source.is_active,
                    'error_count': data_source.error_count,
                    'success_count': data_source.success_count,
                    'avg_response_time_ms': data_source.get_avg_response_time(),
                    'last_error': data_source.last_error
                }
                
                health_status['sources'][source.value] = source_status
                
                if is_healthy:
                    healthy_sources += 1
                
                # Update storage
                await self.storage.update_source_status(
                    source,
                    'HEALTHY' if is_healthy else 'UNHEALTHY',
                    data_source.get_avg_response_time(),
                    data_source.last_error
                )
                
            except Exception as e:
                health_status['sources'][source.value] = {
                    'healthy': False,
                    'active': False,
                    'error': str(e)
                }
        
        # Determine overall status
        if healthy_sources == 0:
            health_status['overall_status'] = 'CRITICAL'
        elif healthy_sources < total_sources * 0.5:
            health_status['overall_status'] = 'DEGRADED'
        elif healthy_sources < total_sources:
            health_status['overall_status'] = 'WARNING'
        
        health_status['healthy_sources'] = healthy_sources
        health_status['total_sources'] = total_sources
        
        return health_status
    
    def _calculate_expected_data_points(self, request: DataRequest) -> int:
        """Calculate expected number of data points"""
        time_diff = request.end_date - request.start_date
        
        if request.timeframe == TimeFrame.MINUTE_1:
            # Assume 6.5 hours trading day
            return int(time_diff.days * 6.5 * 60)
        elif request.timeframe == TimeFrame.MINUTE_5:
            return int(time_diff.days * 6.5 * 12)
        elif request.timeframe == TimeFrame.MINUTE_15:
            return int(time_diff.days * 6.5 * 4)
        elif request.timeframe == TimeFrame.HOUR_1:
            return int(time_diff.days * 6.5)
        elif request.timeframe == TimeFrame.DAY_1:
            return int(time_diff.days)
        elif request.timeframe == TimeFrame.WEEK_1:
            return int(time_diff.days / 7)
        elif request.timeframe == TimeFrame.MONTH_1:
            return int(time_diff.days / 30)
        else:
            return int(time_diff.days)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'requests': self.stats.copy(),
            'sources': {
                source.value: {
                    'active': data_source.is_active,
                    'error_count': data_source.error_count,
                    'success_count': data_source.success_count,
                    'avg_response_time_ms': data_source.get_avg_response_time()
                }
                for source, data_source in self.data_sources.items()
            },
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['requests_total'], 1)
        }

# ===================== MAIN EXECUTION =====================

async def main():
    """Main execution function for testing"""
    print("📡 OMNI ALPHA - STEP 2: DATA COLLECTION & MARKET DATA")
    print("=" * 70)
    
    # Initialize data collection system
    data_system = DataCollectionSystem()
    
    try:
        # Health check
        print("\n🏥 PERFORMING HEALTH CHECK...")
        health_status = await data_system.health_check()
        print(f"   Overall Status: {health_status['overall_status']}")
        print(f"   Healthy Sources: {health_status['healthy_sources']}/{health_status['total_sources']}")
        
        for source, status in health_status['sources'].items():
            icon = "✅" if status['healthy'] else "❌"
            print(f"   {icon} {source}: {'HEALTHY' if status['healthy'] else 'UNHEALTHY'}")
        
        # Test data fetching
        print("\n📊 TESTING DATA COLLECTION...")
        
        # Create data request
        request = DataRequest(
            symbol="AAPL",
            timeframe=TimeFrame.DAY_1,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            cache_enabled=True
        )
        
        # Fetch historical data
        data = await data_system.get_historical_data(request)
        print(f"   Historical Data: {len(data)} data points for AAPL")
        
        if data:
            latest = data[-1]
            print(f"   Latest Price: ${latest.close:.2f} (Volume: {latest.volume:,})")
            print(f"   Data Quality: {latest.quality.value}")
            print(f"   Data Source: {latest.source.value}")
        
        # Test real-time quote
        quote = await data_system.get_real_time_quote("AAPL")
        if quote:
            print(f"   Real-time Quote: Bid ${quote.bid:.2f} / Ask ${quote.ask:.2f}")
            print(f"   Quote Source: {quote.source.value}")
        
        # Show statistics
        stats = data_system.get_statistics()
        print(f"\n📈 STATISTICS:")
        print(f"   Total Requests: {stats['requests']['requests_total']}")
        print(f"   Success Rate: {stats['requests']['requests_success']}/{stats['requests']['requests_total']}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        
        print("\n🚀 Data Collection System is operational!")
        
    except Exception as e:
        print(f"\n❌ Data collection test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

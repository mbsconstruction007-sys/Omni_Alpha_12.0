"""
OMNI ALPHA 5.0 - MERGED ULTIMATE STEP 1 & 2
===========================================
Best of both worlds: Original simplicity + Enhanced enterprise features
"""

import os
import sys
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import sqlite3

# Third party imports
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    from alpaca.trading.client import TradingClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Database imports
try:
    import asyncpg
    import redis
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    DATABASES_AVAILABLE = True
except ImportError:
    DATABASES_AVAILABLE = False

# Monitoring imports
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Security imports
try:
    from cryptography.fernet import Fernet
    import jwt
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Load environment
load_dotenv()

# ===================== CONFIGURATION (MERGED BEST PRACTICES) =====================

class TradingMode(Enum):
    """Trading modes"""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class MergedConfig:
    """Merged configuration combining simplicity with enterprise features"""
    
    # Original simple config (from previous)
    telegram_token: str = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
    alpaca_key: str = 'PK02D3BXIPSW11F0Q9OW'
    alpaca_secret: str = os.getenv('ALPACA_SECRET_KEY', '')
    google_api_key: str = 'AIzaSyDpKZV5XTysC2T9lJax29v2kIAR2q6LXnU'
    
    # Trading parameters (from original)
    max_positions: int = 5
    position_size_pct: float = 0.10
    stop_loss: float = 0.02
    take_profit: float = 0.05
    confidence_threshold: int = 65
    scan_symbols: List[str] = None
    
    # Enhanced enterprise config (from new)
    environment: str = os.getenv('ENVIRONMENT', 'production')
    trading_mode: TradingMode = TradingMode.PAPER
    max_position_size_dollars: float = float(os.getenv('MAX_POSITION_SIZE_DOLLARS', '10000'))
    max_daily_trades: int = int(os.getenv('MAX_DAILY_TRADES', '100'))
    max_daily_loss: float = float(os.getenv('MAX_DAILY_LOSS', '1000'))
    
    # Database config (enhanced)
    db_host: str = os.getenv('DB_HOST', 'localhost')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', 'omni_alpha')
    db_user: str = os.getenv('DB_USER', 'postgres')
    db_password: str = os.getenv('DB_PASSWORD', 'postgres')
    
    # Redis config
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    
    # Monitoring config
    monitoring_enabled: bool = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
    prometheus_port: int = int(os.getenv('PROMETHEUS_PORT', '8001'))
    
    def __post_init__(self):
        if self.scan_symbols is None:
            self.scan_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

# Global config instance
config = MergedConfig()

# Logging setup (enhanced from new, simplified from original)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('omni_alpha_merged.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== STEP 1: MERGED CORE INFRASTRUCTURE =====================

class MergedCoreInfrastructure:
    """
    Merged core infrastructure combining:
    - Original simplicity and direct API access
    - Enhanced enterprise features and monitoring
    """
    
    def __init__(self):
        # Original simple connection (from previous) with fallback
        if ALPACA_AVAILABLE and config.alpaca_secret:
            try:
                self.api = tradeapi.REST(config.alpaca_key, config.alpaca_secret, 'https://paper-api.alpaca.markets')
            except Exception as e:
                logger.warning(f"Alpaca API initialization failed: {e}")
                self.api = None
        else:
            logger.warning("Alpaca credentials not available, running in demo mode")
            self.api = None
            
        # Enhanced database support (from new)
        self.databases = {
            'postgres': None,
            'redis': None,
            'influxdb': None,
            'sqlite': None
        }
        
        # Enhanced monitoring (from new)
        self.metrics = self._setup_monitoring()
        
        # Enhanced security (from new)
        self.security = self._setup_security()
        
        # State management
        self.connected = False
        self.system_status = 'initializing'
        self.health_score = 0.0
        
    def _setup_monitoring(self):
        """Setup monitoring (enhanced feature)"""
        if not PROMETHEUS_AVAILABLE or not config.monitoring_enabled:
            return None
            
        try:
            # Start Prometheus server
            start_http_server(config.prometheus_port)
            
            # Define metrics
            metrics = {
                'trades_total': Counter('trades_total', 'Total trades executed'),
                'errors_total': Counter('errors_total', 'Total errors', ['component']),
                'system_health': Gauge('system_health', 'System health score'),
                'latency_seconds': Histogram('request_latency_seconds', 'Request latency'),
                'active_connections': Gauge('active_connections', 'Active connections', ['type'])
            }
            
            logger.info(f"Monitoring started on port {config.prometheus_port}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Monitoring setup failed: {e}")
            return None
    
    def _setup_security(self):
        """Setup security (enhanced feature)"""
        if not SECURITY_AVAILABLE:
            return None
            
        try:
            # Generate encryption key if not exists
            key_file = 'security_key.key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
            
            cipher = Fernet(key)
            logger.info("Security encryption initialized")
            return cipher
            
        except Exception as e:
            logger.warning(f"Security setup failed: {e}")
            return None
    
    async def initialize(self):
        """Initialize all infrastructure components"""
        logger.info("🚀 Initializing Merged Core Infrastructure...")
        
        # Test Alpaca connection (original simplicity)
        connection_result = self.test_connection()
        
        # Initialize databases (enhanced feature)
        await self._initialize_databases()
        
        # Calculate health score
        self._calculate_health_score()
        
        # Display status (enhanced)
        self._display_status()
        
        return connection_result['status'] == 'connected'
    
    def test_connection(self):
        """Test Alpaca connection (original method enhanced)"""
        try:
            if not self.api:
                # Demo mode - simulate connection
                self.connected = False
                self.system_status = 'demo_mode'
                return {
                    'status': 'demo_mode',
                    'message': 'Running in demo mode - Alpaca API not available',
                    'cash': 100000.0,
                    'buying_power': 100000.0
                }
                
            account = self.api.get_account()
            self.connected = True
            self.system_status = 'connected'
            
            # Record metrics (enhanced feature)
            if self.metrics:
                self.metrics['active_connections'].labels(type='alpaca').set(1)
            
            return {
                'status': 'connected',
                'account_id': account.account_number,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'trading_mode': config.trading_mode.value
            }
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            
            # Record error (enhanced feature)
            if self.metrics:
                self.metrics['errors_total'].labels(component='alpaca').inc()
            
            return {'status': 'error', 'message': str(e)}
    
    async def _initialize_databases(self):
        """Initialize databases with fallbacks (enhanced feature)"""
        if not DATABASES_AVAILABLE:
            # Fallback to SQLite (original simplicity)
            self._setup_sqlite_fallback()
            return
        
        # Try PostgreSQL
        try:
            self.databases['postgres'] = await asyncpg.create_pool(
                host=config.db_host,
                port=config.db_port,
                user=config.db_user,
                password=config.db_password,
                database=config.db_name,
                min_size=2,
                max_size=10
            )
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.warning(f"PostgreSQL failed: {e}, using SQLite fallback")
            self._setup_sqlite_fallback()
        
        # Try Redis
        try:
            self.databases['redis'] = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
            self.databases['redis'].ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis failed: {e}, using memory cache")
            self.databases['redis'] = {}  # Use dict as fallback
        
        # Try InfluxDB
        try:
            self.databases['influxdb'] = InfluxDBClient(
                url=os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
                token=os.getenv('INFLUXDB_TOKEN', 'my-token'),
                org=os.getenv('INFLUXDB_ORG', 'omni-alpha')
            )
            logger.info("InfluxDB connected")
        except Exception as e:
            logger.warning(f"InfluxDB failed: {e}, continuing without metrics DB")
    
    def _setup_sqlite_fallback(self):
        """Setup SQLite fallback (original simplicity)"""
        self.databases['sqlite'] = sqlite3.connect('omni_alpha_merged.db')
        self.databases['sqlite'].execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.databases['sqlite'].commit()
        logger.info("SQLite fallback initialized")
    
    def _calculate_health_score(self):
        """Calculate system health score (enhanced feature)"""
        score = 0.0
        total_components = 5
        
        # Alpaca connection
        if self.connected:
            score += 0.3
        
        # Database connections
        if self.databases['postgres']:
            score += 0.2
        elif self.databases['sqlite']:
            score += 0.1
        
        if self.databases['redis']:
            score += 0.2
        elif isinstance(self.databases['redis'], dict):
            score += 0.1
        
        if self.databases['influxdb']:
            score += 0.1
        
        # Monitoring
        if self.metrics:
            score += 0.1
        
        # Security
        if self.security:
            score += 0.1
        
        self.health_score = score
        
        # Update metrics
        if self.metrics:
            self.metrics['system_health'].set(score)
    
    def _display_status(self):
        """Display system status (enhanced from both)"""
        print("\n" + "=" * 60)
        print("🎯 OMNI ALPHA 5.0 - MERGED SYSTEM STATUS")
        print("=" * 60)
        
        # Connection status (original style)
        if self.connected:
            print("✅ Alpaca Connection: CONNECTED")
        else:
            print("❌ Alpaca Connection: FAILED")
        
        # Database status (enhanced feature)
        print("\n📊 Database Status:")
        if self.databases['postgres']:
            print("   ✅ PostgreSQL: Connected (Primary)")
        elif self.databases['sqlite']:
            print("   ⚠️ SQLite: Connected (Fallback)")
        else:
            print("   ❌ Database: Not Connected")
        
        if self.databases['redis']:
            print("   ✅ Redis: Connected")
        elif isinstance(self.databases['redis'], dict):
            print("   ⚠️ Memory Cache: Active (Redis Fallback)")
        
        if self.databases['influxdb']:
            print("   ✅ InfluxDB: Connected")
        else:
            print("   ⚪ InfluxDB: Not Available")
        
        # System health (enhanced feature)
        print(f"\n🏥 System Health: {self.health_score:.1%}")
        if self.health_score >= 0.8:
            print("   🚀 Status: EXCELLENT - Ready for production")
        elif self.health_score >= 0.6:
            print("   ⚠️ Status: GOOD - Ready for trading")
        elif self.health_score >= 0.4:
            print("   ⚠️ Status: DEGRADED - Limited functionality")
        else:
            print("   ❌ Status: CRITICAL - Needs attention")
        
        # Configuration (original + enhanced)
        print(f"\n⚙️ Configuration:")
        print(f"   Trading Mode: {config.trading_mode.value}")
        print(f"   Max Position: ${config.max_position_size_dollars:,.2f}")
        print(f"   Max Daily Loss: ${config.max_daily_loss:,.2f}")
        print(f"   Scan Symbols: {len(config.scan_symbols)} symbols")
        
        # Monitoring endpoints (enhanced)
        if self.metrics:
            print(f"\n🌐 Endpoints:")
            print(f"   Metrics: http://localhost:{config.prometheus_port}/metrics")
        
        print("=" * 60 + "\n")

# ===================== STEP 2: MERGED DATA COLLECTION =====================

class MergedDataCollection:
    """
    Merged data collection combining:
    - Original simple API calls and caching
    - Enhanced real-time streaming and validation
    """
    
    def __init__(self, api, infrastructure):
        # Original simple setup
        self.api = api
        self.data_cache = {}
        
        # Enhanced features
        self.infrastructure = infrastructure
        self.stream_client = None
        self.is_streaming = False
        self.data_handlers = []
        self.data_quality_stats = {
            'total_ticks': 0,
            'valid_ticks': 0,
            'invalid_ticks': 0,
            'last_update': None
        }
        
        # Setup streaming (enhanced feature)
        if ALPACA_AVAILABLE and config.alpaca_key and config.alpaca_secret:
            self._setup_streaming()
    
    def _setup_streaming(self):
        """Setup real-time streaming (enhanced feature)"""
        try:
            self.stream_client = StockDataStream(config.alpaca_key, config.alpaca_secret)
            self.stream_client.subscribe_bars(self._handle_bar, *config.scan_symbols)
            self.stream_client.subscribe_quotes(self._handle_quote, *config.scan_symbols)
            logger.info("Real-time streaming setup complete")
        except Exception as e:
            logger.error(f"Streaming setup failed: {e}")
    
    async def start_streaming(self):
        """Start real-time data streaming (enhanced feature)"""
        if not self.stream_client:
            logger.warning("Streaming not available")
            return False
        
        try:
            self.is_streaming = True
            asyncio.create_task(self._run_stream())
            logger.info(f"Started streaming for {len(config.scan_symbols)} symbols")
            return True
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    async def _run_stream(self):
        """Run streaming with error recovery (enhanced feature)"""
        while self.is_streaming:
            try:
                self.stream_client.run()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)
                # Auto-reconnect
                if self.is_streaming:
                    self._setup_streaming()
    
    async def _handle_bar(self, bar):
        """Handle bar data (enhanced feature)"""
        try:
            # Validate data quality
            if self._validate_bar_data(bar):
                self.data_quality_stats['valid_ticks'] += 1
                
                # Store in cache (original method)
                symbol = bar.symbol
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = []
                
                self.data_cache[symbol].append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
                
                # Call handlers
                for handler in self.data_handlers:
                    await handler('bar', bar)
                
                # Update metrics
                if self.infrastructure.metrics:
                    self.infrastructure.metrics['trades_total'].inc()
            else:
                self.data_quality_stats['invalid_ticks'] += 1
                
            self.data_quality_stats['total_ticks'] += 1
            self.data_quality_stats['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Bar handling error: {e}")
    
    async def _handle_quote(self, quote):
        """Handle quote data (enhanced feature)"""
        try:
            # Store latest quote (original method enhanced)
            symbol = quote.symbol
            self.data_cache[f"{symbol}_quote"] = {
                'symbol': symbol,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp
            }
            
            # Call handlers
            for handler in self.data_handlers:
                await handler('quote', quote)
                
        except Exception as e:
            logger.error(f"Quote handling error: {e}")
    
    def _validate_bar_data(self, bar) -> bool:
        """Validate bar data quality (enhanced feature)"""
        try:
            # Basic validation checks
            if bar.close <= 0 or bar.volume < 0:
                return False
            
            if bar.high < bar.low:
                return False
            
            if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
                return False
            
            # Price movement check (max 20% change)
            if hasattr(bar, 'previous_close') and bar.previous_close > 0:
                change_pct = abs(bar.close - bar.previous_close) / bar.previous_close
                if change_pct > 0.20:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_market_data(self, symbol, timeframe='1Day', days=30):
        """Get historical market data (original method enhanced)"""
        try:
            # Try API call (original method)
            bars = self.api.get_bars(
                symbol, timeframe,
                start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                limit=days
            ).df
            
            # Cache data (original method)
            self.data_cache[symbol] = bars
            
            # Record metrics (enhanced feature)
            if self.infrastructure.metrics:
                self.infrastructure.metrics['trades_total'].inc()
            
            logger.info(f"Retrieved {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            
            # Record error (enhanced feature)
            if self.infrastructure.metrics:
                self.infrastructure.metrics['errors_total'].labels(component='data').inc()
            
            return None
    
    def get_latest_quote(self, symbol):
        """Get latest quote (original method enhanced)"""
        try:
            # Try real-time cache first (enhanced feature)
            cached_quote = self.data_cache.get(f"{symbol}_quote")
            if cached_quote and (datetime.now() - cached_quote['timestamp']).seconds < 5:
                return cached_quote
            
            # Fallback to API call (original method)
            quote = self.api.get_latest_quote(symbol)
            
            result = {
                'symbol': symbol,
                'bid': quote.bp,
                'ask': quote.ap,
                'timestamp': datetime.now(),
                'spread': quote.ap - quote.bp,
                'mid_price': (quote.ap + quote.bp) / 2
            }
            
            # Cache result (enhanced)
            self.data_cache[f"{symbol}_quote"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return None
    
    def add_data_handler(self, handler: Callable):
        """Add data handler (enhanced feature)"""
        self.data_handlers.append(handler)
    
    def get_data_quality_report(self):
        """Get data quality report (enhanced feature)"""
        total = self.data_quality_stats['total_ticks']
        valid = self.data_quality_stats['valid_ticks']
        
        return {
            'total_ticks': total,
            'valid_ticks': valid,
            'invalid_ticks': self.data_quality_stats['invalid_ticks'],
            'quality_rate': (valid / total * 100) if total > 0 else 0,
            'last_update': self.data_quality_stats['last_update'],
            'cached_symbols': len([k for k in self.data_cache.keys() if not k.endswith('_quote')]),
            'cached_quotes': len([k for k in self.data_cache.keys() if k.endswith('_quote')])
        }
    
    async def close(self):
        """Close data collection (enhanced feature)"""
        self.is_streaming = False
        if self.stream_client:
            self.stream_client.stop()
        logger.info("Data collection stopped")

# ===================== MERGED SYSTEM ORCHESTRATOR =====================

class MergedSystemOrchestrator:
    """Orchestrator combining original simplicity with enhanced features"""
    
    def __init__(self):
        self.infrastructure = None
        self.data_collection = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize merged system"""
        print("🚀 OMNI ALPHA 5.0 - MERGED SYSTEM INITIALIZATION")
        print("=" * 70)
        
        # Initialize infrastructure (Step 1)
        print("📋 Step 1: Initializing Core Infrastructure...")
        self.infrastructure = MergedCoreInfrastructure()
        infra_success = await self.infrastructure.initialize()
        
        # Initialize data collection (Step 2)
        print("📋 Step 2: Initializing Data Collection...")
        self.data_collection = MergedDataCollection(self.infrastructure.api, self.infrastructure)
        
        # Start streaming if available
        if self.infrastructure.connected:
            await self.data_collection.start_streaming()
        
        # System ready
        if infra_success:
            print("✅ Merged system initialization complete!")
            self.is_running = True
        else:
            print("⚠️ System initialized with limitations")
            self.is_running = True
        
        return True
    
    async def run(self):
        """Run the merged system"""
        if not self.is_running:
            await self.initialize()
        
        print("\n🎯 OMNI ALPHA 5.0 - MERGED SYSTEM RUNNING")
        print("=" * 50)
        print("System operational with merged features!")
        print("Original simplicity + Enhanced enterprise features")
        print("Press Ctrl+C to shutdown...")
        print()
        
        try:
            # Operational loop
            loop_count = 0
            while self.is_running:
                await asyncio.sleep(1)
                loop_count += 1
                
                # Periodic status (every 30 seconds)
                if loop_count % 30 == 0:
                    quality_report = self.data_collection.get_data_quality_report()
                    print(f"⏰ Status: Health {self.infrastructure.health_score:.1%}, "
                          f"Data Quality {quality_report['quality_rate']:.1f}%, "
                          f"Cached: {quality_report['cached_symbols']} symbols")
                
        except KeyboardInterrupt:
            print("\n🛑 Graceful shutdown...")
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown merged system"""
        self.is_running = False
        
        if self.data_collection:
            await self.data_collection.close()
        
        print("✅ Merged system shutdown complete")
    
    def get_system_info(self):
        """Get complete system information"""
        return {
            'system_name': 'Omni Alpha 5.0 Merged',
            'version': '5.0.0-merged',
            'features': {
                'original_simplicity': True,
                'enhanced_enterprise': True,
                'real_time_streaming': self.data_collection.is_streaming if self.data_collection else False,
                'monitoring': self.infrastructure.metrics is not None if self.infrastructure else False,
                'security': self.infrastructure.security is not None if self.infrastructure else False,
                'multi_database': bool(self.infrastructure.databases['postgres']) if self.infrastructure else False
            },
            'health_score': self.infrastructure.health_score if self.infrastructure else 0,
            'data_quality': self.data_collection.get_data_quality_report() if self.data_collection else {},
            'is_running': self.is_running
        }

# ===================== MAIN EXECUTION =====================

async def main():
    """Main execution for merged system"""
    print("🎊 OMNI ALPHA 5.0 - MERGED ULTIMATE SYSTEM")
    print("=" * 60)
    print("Combining original simplicity with enhanced enterprise features")
    print(f"Started: {datetime.now()}")
    print()
    
    # Run merged system
    orchestrator = MergedSystemOrchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())

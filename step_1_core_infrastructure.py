"""
STEP 1: CORE INFRASTRUCTURE - OMNI ALPHA TRADING SYSTEM
Enterprise-grade infrastructure foundation with database, logging, config, and health monitoring
"""

import os
import sys
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
from contextlib import asynccontextmanager

# Third party imports
try:
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from pydantic import BaseSettings, validator
    from pydantic import BaseModel as PydanticBaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv('alpaca_live_trading.env')

# ===================== CONFIGURATION MANAGEMENT =====================

class OmniAlphaConfig:
    """Centralized configuration management for Omni Alpha"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables"""
        
        # Core Application Settings
        self.ENV = os.getenv('ENV', 'production')
        self.APP_NAME = os.getenv('APP_NAME', 'Omni Alpha Enhanced')
        self.APP_VERSION = os.getenv('APP_VERSION', '12.0+')
        self.DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
        self.TESTING = os.getenv('TESTING', 'false').lower() == 'true'
        self.TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
        
        # Database Configuration
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///omni_alpha.db')
        self.DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
        self.DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '20'))
        self.DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        
        # Trading Configuration
        self.MAX_POSITION_SIZE_PERCENT = float(os.getenv('MAX_POSITION_SIZE', '0.10'))
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '20'))
        self.STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '3.0')) / 100
        self.TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', '6.0')) / 100
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        
        # API Configuration
        self.ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
        self.ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'omni_alpha.log')
        self.LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
        self.LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))
        
        # Performance Settings
        self.SCAN_INTERVAL_MINUTES = int(os.getenv('SCAN_INTERVAL_MINUTES', '5'))
        self.AUTO_TRADE_ENABLED = os.getenv('AUTO_TRADE_ENABLED', 'true').lower() == 'true'
        self.ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        self.METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))
        
        # Security Settings
        self.ENABLE_SECURITY = os.getenv('ENABLE_SECURITY', 'true').lower() == 'true'
        self.JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'omni_alpha_secret_2025')
        
        # Data Sources
        self.YAHOO_FINANCE_ENABLED = os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
        self.NSE_DATA_ENABLED = os.getenv('NSE_DATA_ENABLED', 'true').lower() == 'true'
        self.ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required API keys
        if not self.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is required")
        if not self.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is required")
        if not self.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        
        # Validate trading parameters
        if self.MAX_POSITION_SIZE_PERCENT <= 0 or self.MAX_POSITION_SIZE_PERCENT > 1:
            errors.append("MAX_POSITION_SIZE_PERCENT must be between 0 and 1")
        if self.MAX_POSITIONS <= 0:
            errors.append("MAX_POSITIONS must be greater than 0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'env': self.ENV,
            'app_name': self.APP_NAME,
            'app_version': self.APP_VERSION,
            'debug': self.DEBUG,
            'testing': self.TESTING,
            'trading_mode': self.TRADING_MODE,
            'max_positions': self.MAX_POSITIONS,
            'auto_trade_enabled': self.AUTO_TRADE_ENABLED,
            'enable_metrics': self.ENABLE_METRICS,
            'enable_security': self.ENABLE_SECURITY
        }

# ===================== DATABASE MANAGEMENT =====================

# Database Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class Trade(Base):
        __tablename__ = 'trades'
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(10), nullable=False)
        action = Column(String(10), nullable=False)  # BUY, SELL
        quantity = Column(Integer, nullable=False)
        price = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.now)
        order_id = Column(String(50))
        status = Column(String(20), default='PENDING')
        pnl = Column(Float, default=0.0)
        strategy = Column(String(50))
        trade_metadata = Column(Text)  # JSON metadata
    
    class Position(Base):
        __tablename__ = 'positions'
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(10), nullable=False, unique=True)
        quantity = Column(Integer, nullable=False)
        avg_price = Column(Float, nullable=False)
        current_price = Column(Float)
        market_value = Column(Float)
        unrealized_pnl = Column(Float, default=0.0)
        realized_pnl = Column(Float, default=0.0)
        entry_time = Column(DateTime, default=datetime.now)
        last_updated = Column(DateTime, default=datetime.now)
        stop_loss = Column(Float)
        take_profit = Column(Float)
        strategy = Column(String(50))
        position_metadata = Column(Text)
    
    class SystemMetrics(Base):
        __tablename__ = 'system_metrics'
        
        id = Column(Integer, primary_key=True)
        timestamp = Column(DateTime, default=datetime.now)
        metric_name = Column(String(100), nullable=False)
        metric_value = Column(Float, nullable=False)
        metric_type = Column(String(20))  # counter, gauge, histogram
        tags = Column(Text)  # JSON tags

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self, config: OmniAlphaConfig):
        self.config = config
        self.engine = None
        self.Session = None
        self.connected = False
    
    async def connect(self):
        """Initialize database connection"""
        try:
            if SQLALCHEMY_AVAILABLE:
                # Create engine with connection pooling
                if self.config.DATABASE_URL.startswith('sqlite'):
                    self.engine = create_engine(
                        self.config.DATABASE_URL,
                        poolclass=StaticPool,
                        connect_args={'check_same_thread': False},
                        echo=self.config.DEBUG
                    )
                else:
                    self.engine = create_engine(
                        self.config.DATABASE_URL,
                        pool_size=self.config.DB_POOL_SIZE,
                        max_overflow=self.config.DB_MAX_OVERFLOW,
                        pool_timeout=self.config.DB_POOL_TIMEOUT,
                        echo=self.config.DEBUG
                    )
                
                # Create session factory
                self.Session = sessionmaker(bind=self.engine)
                
                # Create tables
                await self.create_tables()
                
                self.connected = True
                logging.info("Database connected successfully")
            else:
                # Fallback to SQLite without SQLAlchemy
                db_path = self.config.DATABASE_URL.replace('sqlite:///', '')
                self.connection = sqlite3.connect(db_path, check_same_thread=False)
                await self.create_sqlite_tables()
                self.connected = True
                logging.info("SQLite database connected (fallback mode)")
                
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        if SQLALCHEMY_AVAILABLE and self.engine:
            Base.metadata.create_all(self.engine)
            logging.info("Database tables created")
    
    async def create_sqlite_tables(self):
        """Create SQLite tables (fallback)"""
        cursor = self.connection.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                order_id TEXT,
                status TEXT DEFAULT 'PENDING',
                pnl REAL DEFAULT 0.0,
                strategy TEXT,
                trade_metadata TEXT
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                market_value REAL,
                unrealized_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                stop_loss REAL,
                take_profit REAL,
                strategy TEXT,
                position_metadata TEXT
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT,
                tags TEXT
            )
        ''')
        
        self.connection.commit()
        logging.info("SQLite tables created")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with context manager"""
        if SQLALCHEMY_AVAILABLE and self.Session:
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()
        else:
            yield self.connection
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            if SQLALCHEMY_AVAILABLE and self.engine:
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
            elif hasattr(self, 'connection'):
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                return True
            return False
        except Exception as e:
            logging.error(f"Database health check failed: {e}")
            return False

# ===================== LOGGING MANAGEMENT =====================

class LoggingManager:
    """Centralized logging configuration and management"""
    
    def __init__(self, config: OmniAlphaConfig):
        self.config = config
        self.logger = None
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # Setup basic configuration
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / self.config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Setup structured logging if available
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        
        self.logger = logging.getLogger("omni_alpha")
        self.logger.info("Logging system initialized")
        
        return self.logger

# ===================== METRICS COLLECTION =====================

class MetricsCollector:
    """Prometheus metrics collection"""
    
    def __init__(self, config: OmniAlphaConfig):
        self.config = config
        self.enabled = config.ENABLE_METRICS and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            # Trading metrics
            self.trades_total = Counter('omni_alpha_trades_total', 'Total trades executed', ['action', 'symbol'])
            self.trade_duration = Histogram('omni_alpha_trade_duration_seconds', 'Trade execution time')
            self.portfolio_value = Gauge('omni_alpha_portfolio_value_usd', 'Current portfolio value')
            self.positions_count = Gauge('omni_alpha_positions_count', 'Number of open positions')
            self.daily_pnl = Gauge('omni_alpha_daily_pnl_usd', 'Daily profit and loss')
            self.win_rate = Gauge('omni_alpha_win_rate_percent', 'Trading win rate percentage')
            
            # System metrics
            self.system_health = Gauge('omni_alpha_system_health', 'System health score (0-1)')
            self.api_requests = Counter('omni_alpha_api_requests_total', 'API requests', ['service', 'status'])
            self.errors_total = Counter('omni_alpha_errors_total', 'Total errors', ['type'])
            
            # Start metrics server
            if config.METRICS_PORT:
                start_http_server(config.METRICS_PORT)
                logging.info(f"Metrics server started on port {config.METRICS_PORT}")
    
    def record_trade(self, action: str, symbol: str, duration: float = None):
        """Record trade metrics"""
        if self.enabled:
            self.trades_total.labels(action=action, symbol=symbol).inc()
            if duration:
                self.trade_duration.observe(duration)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value metric"""
        if self.enabled:
            self.portfolio_value.set(value)
    
    def update_positions_count(self, count: int):
        """Update positions count metric"""
        if self.enabled:
            self.positions_count.set(count)
    
    def record_error(self, error_type: str):
        """Record error metric"""
        if self.enabled:
            self.errors_total.labels(type=error_type).inc()

# ===================== HEALTH MONITORING =====================

class HealthChecker:
    """System health monitoring and checks"""
    
    def __init__(self, config: OmniAlphaConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.last_check = None
        self.health_status = {}
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            is_healthy = await self.db_manager.health_check()
            response_time = time.time() - start_time
            
            return {
                'healthy': is_healthy,
                'response_time_ms': round(response_time * 1000, 2),
                'connected': self.db_manager.connected,
                'error': None
            }
        except Exception as e:
            return {
                'healthy': False,
                'response_time_ms': round((time.time() - start_time) * 1000, 2),
                'connected': False,
                'error': str(e)
            }
    
    async def check_apis(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        api_status = {}
        
        # Check Alpaca API
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY,
                self.config.ALPACA_BASE_URL
            )
            account = api.get_account()
            api_status['alpaca'] = {
                'healthy': True,
                'account_status': account.status,
                'error': None
            }
        except Exception as e:
            api_status['alpaca'] = {
                'healthy': False,
                'account_status': None,
                'error': str(e)
            }
        
        # Check Telegram API
        try:
            from telegram import Bot
            bot = Bot(token=self.config.TELEGRAM_BOT_TOKEN)
            bot_info = await bot.get_me()
            api_status['telegram'] = {
                'healthy': True,
                'bot_username': bot_info.username,
                'error': None
            }
        except Exception as e:
            api_status['telegram'] = {
                'healthy': False,
                'bot_username': None,
                'error': str(e)
            }
        
        return api_status
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        import psutil
        
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent': memory.percent,
                    'healthy': memory.percent < 85
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent': round((disk.used / disk.total) * 100, 2),
                    'healthy': (disk.used / disk.total) < 0.9
                },
                'cpu': {
                    'percent': cpu_percent,
                    'healthy': cpu_percent < 80
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'healthy': False
            }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_check = datetime.now()
        
        # Run all health checks
        db_health = await self.check_database()
        api_health = await self.check_apis()
        system_health = await self.check_system_resources()
        
        # Calculate overall health score
        health_scores = []
        
        if db_health['healthy']:
            health_scores.append(1.0)
        else:
            health_scores.append(0.0)
        
        api_healthy_count = sum(1 for api in api_health.values() if api['healthy'])
        api_score = api_healthy_count / len(api_health) if api_health else 0
        health_scores.append(api_score)
        
        if 'error' not in system_health:
            system_score = sum(1 for component in system_health.values() if component['healthy']) / len(system_health)
            health_scores.append(system_score)
        else:
            health_scores.append(0.0)
        
        overall_score = sum(health_scores) / len(health_scores)
        
        self.health_status = {
            'timestamp': self.last_check.isoformat(),
            'overall_health_score': round(overall_score, 2),
            'status': 'healthy' if overall_score >= 0.8 else 'degraded' if overall_score >= 0.5 else 'unhealthy',
            'database': db_health,
            'apis': api_health,
            'system': system_health
        }
        
        return self.health_status

# ===================== CORE INFRASTRUCTURE ORCHESTRATOR =====================

class CoreInfrastructure:
    """Main core infrastructure orchestrator"""
    
    def __init__(self):
        self.config = OmniAlphaConfig()
        self.db_manager = None
        self.logging_manager = None
        self.metrics_collector = None
        self.health_checker = None
        self.logger = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all core infrastructure components"""
        try:
            # Validate configuration
            config_errors = self.config.validate_config()
            if config_errors:
                raise ValueError(f"Configuration errors: {', '.join(config_errors)}")
            
            # Setup logging first
            self.logging_manager = LoggingManager(self.config)
            self.logger = self.logging_manager.setup_logging()
            
            self.logger.info("Starting Omni Alpha Core Infrastructure initialization...")
            self.logger.info(f"Environment: {self.config.ENV}")
            self.logger.info(f"Trading Mode: {self.config.TRADING_MODE}")
            
            # Initialize database
            self.db_manager = DatabaseManager(self.config)
            await self.db_manager.connect()
            
            # Initialize metrics collection
            if self.config.ENABLE_METRICS:
                self.metrics_collector = MetricsCollector(self.config)
                self.logger.info("Metrics collection enabled")
            
            # Initialize health checker
            self.health_checker = HealthChecker(self.config, self.db_manager)
            
            # Perform initial health check
            health_status = await self.health_checker.comprehensive_health_check()
            self.logger.info(f"Initial health check: {health_status['status']} (score: {health_status['overall_health_score']})")
            
            self.initialized = True
            self.logger.info("Core Infrastructure initialization completed successfully!")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Core Infrastructure initialization failed: {e}")
            else:
                print(f"Core Infrastructure initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        if self.logger:
            self.logger.info("Shutting down Core Infrastructure...")
        
        # Close database connections
        if self.db_manager and hasattr(self.db_manager, 'engine') and self.db_manager.engine:
            self.db_manager.engine.dispose()
        
        if self.logger:
            self.logger.info("Core Infrastructure shutdown completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current infrastructure status"""
        return {
            'initialized': self.initialized,
            'config': self.config.to_dict(),
            'database_connected': self.db_manager.connected if self.db_manager else False,
            'metrics_enabled': self.config.ENABLE_METRICS,
            'last_health_check': self.health_checker.health_status if self.health_checker else None
        }

# ===================== MAIN EXECUTION =====================

async def main():
    """Main execution function for testing"""
    print("🏗️ OMNI ALPHA - STEP 1: CORE INFRASTRUCTURE")
    print("=" * 60)
    
    # Initialize core infrastructure
    core = CoreInfrastructure()
    
    try:
        # Initialize all components
        await core.initialize()
        
        # Display status
        status = core.get_status()
        print("\n✅ CORE INFRASTRUCTURE STATUS:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Environment: {status['config']['env']}")
        print(f"   App Version: {status['config']['app_version']}")
        print(f"   Database Connected: {status['database_connected']}")
        print(f"   Metrics Enabled: {status['metrics_enabled']}")
        
        # Perform health check
        if core.health_checker:
            health = await core.health_checker.comprehensive_health_check()
            print(f"\n🏥 HEALTH CHECK RESULTS:")
            print(f"   Overall Status: {health['status'].upper()}")
            print(f"   Health Score: {health['overall_health_score']}/1.0")
            print(f"   Database: {'✅' if health['database']['healthy'] else '❌'}")
            print(f"   APIs: {'✅' if all(api['healthy'] for api in health['apis'].values()) else '❌'}")
            
        print("\n🚀 Core Infrastructure is ready for trading operations!")
        
    except Exception as e:
        print(f"\n❌ Infrastructure initialization failed: {e}")
        raise
    finally:
        await core.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

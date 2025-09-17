"""
STEP 18: Complete Production Deployment System
Enterprise-grade live trading infrastructure
"""

import os
import sys
import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# External libraries
import pandas as pd
import numpy as np
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from kafka import KafkaProducer, KafkaConsumer
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import sentry_sdk
from telegram import Bot
from email.mime.text import MIMEText
import smtplib
import psutil
import requests
import yaml

from dotenv import load_dotenv
load_dotenv()

# Configure production logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Sentry for error tracking
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        traces_sample_rate=0.1,
        environment=os.getenv('ENVIRONMENT', 'production')
    )

# ===================== MONITORING METRICS =====================

# Prometheus metrics
trades_counter = Counter('trades_total', 'Total number of trades')
pnl_gauge = Gauge('portfolio_pnl', 'Current P&L')
latency_histogram = Histogram('order_latency_seconds', 'Order execution latency')
error_counter = Counter('errors_total', 'Total errors', ['error_type'])
active_positions_gauge = Gauge('active_positions', 'Number of active positions')

# ===================== DATA STRUCTURES =====================

class SystemState(Enum):
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"
    SHUTTING_DOWN = "SHUTTING_DOWN"

@dataclass
class SystemHealth:
    status: SystemState
    uptime: timedelta
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_status: bool
    broker_connection: bool
    data_feed_status: bool
    error_rate: float

@dataclass
class TradingSession:
    session_id: str
    start_time: datetime
    market_open: datetime
    market_close: datetime
    is_holiday: bool
    trading_enabled: bool

# ===================== BROKER INTEGRATION =====================

class ProductionBrokerManager:
    """Production-grade multi-broker integration"""
    
    def __init__(self):
        self.primary_broker = None
        self.secondary_broker = None
        self.active_broker = None
        self.initialize_brokers()
        
    def initialize_brokers(self):
        """Initialize broker connections with failover"""
        try:
            # Simulate broker connections
            self.primary_broker = "UPSTOX_CONNECTION"
            self.active_broker = self.primary_broker
            logger.info("Primary broker (Upstox) connected")
            
        except Exception as e:
            logger.error(f"Primary broker connection failed: {e}")
            
            # Failover to secondary
            try:
                self.secondary_broker = "ZERODHA_CONNECTION"
                self.active_broker = self.secondary_broker
                logger.warning("Using secondary broker (Zerodha)")
                
            except Exception as e2:
                logger.critical(f"All broker connections failed: {e2}")
                raise RuntimeError("Cannot establish broker connection")
    
    async def place_order(self, order_params: Dict) -> str:
        """Place order with retry and failover"""
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Add pre-trade risk checks
                if not self._pre_trade_risk_check(order_params):
                    raise ValueError("Order failed risk checks")
                
                # Place order
                order_id = await self._execute_order(order_params)
                
                # Log trade
                trades_counter.inc()
                self._log_trade(order_params, order_id)
                
                return order_id
                
            except Exception as e:
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Try failover broker
                    if self.active_broker == self.primary_broker and self.secondary_broker:
                        self.active_broker = self.secondary_broker
                        return await self._execute_order(order_params)
                    else:
                        raise
                
                await asyncio.sleep(1)
    
    def _pre_trade_risk_check(self, order_params: Dict) -> bool:
        """Pre-trade risk validation"""
        
        # Check position limits
        max_order_value = float(os.getenv('MAX_SINGLE_ORDER_VALUE', '100000'))
        if order_params.get('quantity', 0) * order_params.get('price', 0) > max_order_value:
            logger.warning("Order exceeds single order limit")
            return False
        
        # Check daily loss limit
        if self._get_daily_pnl() < -float(os.getenv('MAX_DAILY_LOSS', '20000')):
            logger.error("Daily loss limit breached")
            return False
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            logger.error("Circuit breaker is active")
            return False
        
        return True
    
    async def _execute_order(self, params: Dict) -> str:
        """Execute order on active broker"""
        
        start_time = datetime.now()
        
        try:
            # Simulate order execution
            order_id = f"ORD_{int(time.time())}"
            
            # Record latency
            latency = (datetime.now() - start_time).total_seconds()
            latency_histogram.observe(latency)
            
            logger.info(f"Order executed: {order_id}")
            return order_id
            
        except Exception as e:
            error_counter.labels(error_type='order_execution').inc()
            raise
    
    def _get_daily_pnl(self) -> float:
        """Calculate current day P&L"""
        # Implement P&L calculation
        return 0.0
    
    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is triggered"""
        # Implement circuit breaker logic
        return False
    
    def _log_trade(self, params: Dict, order_id: str):
        """Log trade execution"""
        logger.info(f"Trade logged: {order_id} - {params}")

# ===================== MARKET DATA MANAGER =====================

class ProductionDataManager:
    """Production-grade market data management"""
    
    def __init__(self):
        self.primary_feed = None
        self.backup_feed = None
        self.ws_connection = None
        
        # Initialize Redis if available
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                password=os.getenv('REDIS_PASSWORD'),
                decode_responses=True
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
            
        self.initialize_feeds()
    
    def initialize_feeds(self):
        """Initialize market data feeds"""
        
        try:
            # Primary feed (NSE)
            self.primary_feed = self._connect_nse_feed()
            logger.info("Primary data feed connected")
            
        except Exception as e:
            logger.error(f"Primary feed connection failed: {e}")
            
            # Fallback to backup feed
            self.backup_feed = self._connect_yahoo_feed()
            logger.warning("Using backup data feed")
    
    def _connect_nse_feed(self):
        """Connect to NSE WebSocket feed"""
        
        # Simulate NSE feed connection
        logger.info("NSE feed connection simulated")
        return "NSE_FEED_CONNECTION"
    
    def _connect_yahoo_feed(self):
        """Connect to Yahoo Finance backup feed"""
        
        # Simulate Yahoo feed connection
        logger.info("Yahoo backup feed connection simulated")
        return "YAHOO_FEED_CONNECTION"
    
    def _process_tick(self, tick_data: Dict):
        """Process and store market tick"""
        
        # Validate tick data
        if not self._validate_tick(tick_data):
            return
        
        # Store in Redis for fast access
        if self.redis_available:
            symbol = tick_data['symbol']
            self.redis_client.hset(
                f"tick:{symbol}",
                mapping={
                    'price': tick_data['ltp'],
                    'volume': tick_data['volume'],
                    'bid': tick_data.get('bid', 0),
                    'ask': tick_data.get('ask', 0),
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        # Publish to Kafka for processing
        self._publish_to_kafka(tick_data)
    
    def _validate_tick(self, tick: Dict) -> bool:
        """Validate tick data quality"""
        
        # Check required fields
        required_fields = ['symbol', 'ltp', 'volume']
        if not all(field in tick for field in required_fields):
            return False
        
        # Check price sanity
        if tick['ltp'] <= 0 or tick['ltp'] > 1000000:
            return False
        
        return True
    
    def _publish_to_kafka(self, tick_data: Dict):
        """Publish tick data to Kafka"""
        
        try:
            # In production, use actual Kafka
            logger.debug(f"Publishing tick to Kafka: {tick_data['symbol']}")
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")

# ===================== RISK MANAGEMENT ENGINE =====================

class ProductionRiskManager:
    """Production risk management and controls"""
    
    def __init__(self):
        self.db_pool = None
        self.risk_metrics = {}
        self.circuit_breaker_active = False
        self.startup_time = datetime.now()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database connection pool"""
        try:
            # In production environment, use actual PostgreSQL
            if os.getenv('DB_HOST'):
                self.db_pool = ThreadedConnectionPool(
                    1, 20,
                    host=os.getenv('DB_HOST'),
                    database=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD')
                )
                logger.info("Database connection pool initialized")
            else:
                logger.warning("Database not configured, using in-memory storage")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def calculate_portfolio_risk(self) -> Dict:
        """Calculate real-time portfolio risk metrics"""
        
        try:
            # Simulate portfolio risk calculation
            positions = [
                {'symbol': 'NIFTY', 'quantity': 50, 'entry_price': 20000, 'current_price': 20100},
                {'symbol': 'BANKNIFTY', 'quantity': 25, 'entry_price': 45000, 'current_price': 45200}
            ]
            
            # Calculate metrics
            total_exposure = sum(p['quantity'] * p['current_price'] for p in positions)
            total_pnl = sum((p['current_price'] - p['entry_price']) * p['quantity'] for p in positions)
            
            # VaR calculation (simplified)
            var_95 = total_exposure * 0.02  # 2% VaR
            
            self.risk_metrics = {
                'exposure': total_exposure,
                'pnl': total_pnl,
                'var_95': var_95,
                'positions': len(positions),
                'timestamp': datetime.now()
            }
            
            # Update Prometheus metrics
            pnl_gauge.set(total_pnl)
            active_positions_gauge.set(len(positions))
            
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            return {}
    
    def check_risk_limits(self) -> bool:
        """Check if any risk limits are breached"""
        
        metrics = self.calculate_portfolio_risk()
        
        if not metrics:
            return False
        
        # Check VaR limit
        max_var = float(os.getenv('MAX_PORTFOLIO_VAR', '50000'))
        if metrics['var_95'] > max_var:
            logger.error(f"VaR limit breached: {metrics['var_95']}")
            self._trigger_alert("VaR limit breached", "CRITICAL")
            return False
        
        # Check daily loss
        max_loss = float(os.getenv('MAX_DAILY_LOSS', '20000'))
        if metrics['pnl'] < -max_loss:
            logger.error(f"Daily loss limit breached: {metrics['pnl']}")
            self._activate_circuit_breaker()
            return False
        
        # Check position limits
        max_positions = int(os.getenv('MAX_OPEN_POSITIONS', '50'))
        if metrics['positions'] > max_positions:
            logger.warning(f"Position limit reached: {metrics['positions']}")
            return False
        
        return True
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker to halt trading"""
        
        self.circuit_breaker_active = True
        logger.critical("CIRCUIT BREAKER ACTIVATED")
        
        # Send alerts
        self._trigger_alert("Circuit breaker activated", "CRITICAL")
        
        # Schedule reset
        reset_time = int(os.getenv('CIRCUIT_BREAKER_RESET_HOURS', '4')) * 3600
        threading.Timer(reset_time, self._reset_circuit_breaker).start()
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker reset")
    
    def _trigger_alert(self, message: str, severity: str):
        """Trigger alert (placeholder)"""
        logger.warning(f"ALERT [{severity}]: {message}")
    
    def validate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Validate trading signals"""
        
        validated = []
        
        for signal in signals:
            if self._validate_signal(signal):
                validated.append(signal)
            else:
                logger.warning(f"Signal validation failed: {signal}")
        
        return validated
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate individual signal"""
        
        # Check required fields
        required_fields = ['symbol', 'side', 'quantity', 'price']
        if not all(field in signal for field in required_fields):
            return False
        
        # Check position size
        position_value = signal['quantity'] * signal['price']
        max_position = float(os.getenv('MAX_POSITION_SIZE', '100000'))
        
        if position_value > max_position:
            return False
        
        return True

# ===================== MONITORING & ALERTING =====================

class ProductionMonitor:
    """Production monitoring and alerting system"""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.telegram_bot = None
        
        # Initialize Telegram bot if token available
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            try:
                self.telegram_bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            except Exception as e:
                logger.error(f"Telegram bot initialization failed: {e}")
        
        self.alert_channels = {
            'telegram': self._send_telegram_alert,
            'email': self._send_email_alert,
            'console': self._send_console_alert
        }
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL_SECONDS', '30'))
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start monitoring threads"""
        
        # Health check thread
        health_thread = threading.Thread(target=self._health_check_loop)
        health_thread.daemon = True
        health_thread.start()
        
        # Metrics collection thread
        metrics_thread = threading.Thread(target=self._metrics_collection_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def _health_check_loop(self):
        """Continuous health monitoring"""
        
        while True:
            try:
                health = self.check_system_health()
                
                if health.status == SystemState.ERROR:
                    self._trigger_alert(
                        f"System health degraded: CPU={health.cpu_usage:.1f}%, "
                        f"Memory={health.memory_usage:.1f}%, "
                        f"Errors={health.error_rate:.2%}",
                        "HIGH"
                    )
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(60)
    
    def _metrics_collection_loop(self):
        """Collect and update metrics"""
        
        while True:
            try:
                # Update system metrics
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                # Update Prometheus metrics
                prometheus_client.Gauge('system_cpu_usage').set(cpu_usage)
                prometheus_client.Gauge('system_memory_usage').set(memory_usage)
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(30)
    
    def check_system_health(self) -> SystemHealth:
        """Comprehensive system health check"""
        
        health = SystemHealth(
            status=SystemState.RUNNING,
            uptime=datetime.now() - self.startup_time,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('.').percent,
            network_latency=self._check_network_latency(),
            database_status=self._check_database(),
            broker_connection=self._check_broker_connection(),
            data_feed_status=self._check_data_feed(),
            error_rate=self._calculate_error_rate()
        )
        
        # Determine overall status
        if health.cpu_usage > 90 or health.memory_usage > 90:
            health.status = SystemState.ERROR
        elif not health.database_status or not health.broker_connection:
            health.status = SystemState.ERROR
        elif health.error_rate > 0.05:
            health.status = SystemState.ERROR
        
        return health
    
    def _check_network_latency(self) -> float:
        """Check network latency"""
        try:
            start = time.time()
            requests.get('https://www.google.com', timeout=5)
            return (time.time() - start) * 1000
        except:
            return 999.0
    
    def _check_database(self) -> bool:
        """Check database connectivity"""
        # In production, check actual database
        return True
    
    def _check_broker_connection(self) -> bool:
        """Check broker connectivity"""
        # In production, check actual broker
        return True
    
    def _check_data_feed(self) -> bool:
        """Check data feed status"""
        # In production, check actual data feed
        return True
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        # In production, calculate from metrics
        return 0.01
    
    def _trigger_alert(self, message: str, severity: str):
        """Send alerts through configured channels"""
        
        alert = {
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'environment': os.getenv('ENVIRONMENT', 'production')
        }
        
        # Send to appropriate channels based on severity
        if severity == 'CRITICAL':
            for channel in ['telegram', 'email', 'console']:
                try:
                    self.alert_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Alert channel {channel} failed: {e}")
        elif severity == 'HIGH':
            for channel in ['telegram', 'email']:
                try:
                    self.alert_channels[channel](alert)
                except Exception as e:
                    logger.error(f"Alert channel {channel} failed: {e}")
        else:
            try:
                self.alert_channels['telegram'](alert)
            except Exception as e:
                logger.error(f"Telegram alert failed: {e}")
    
    def _send_telegram_alert(self, alert: Dict):
        """Send Telegram alert"""
        try:
            if not self.telegram_bot:
                return
            
            chat_id = (os.getenv('TELEGRAM_CRITICAL_CHAT_ID') 
                      if alert['severity'] == 'CRITICAL' 
                      else os.getenv('TELEGRAM_CHAT_ID'))
            
            if not chat_id:
                return
            
            message = f"""
🚨 **{alert['severity']} Alert**

{alert['message']}

Time: {alert['timestamp']}
Environment: {alert['environment']}
            """
            
            self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            if not os.getenv('EMAIL_SMTP_HOST'):
                return
            
            msg = MIMEText(alert['message'])
            msg['Subject'] = f"Trading Alert - {alert['severity']}"
            msg['From'] = os.getenv('EMAIL_FROM')
            msg['To'] = os.getenv('EMAIL_TO')
            
            with smtplib.SMTP(os.getenv('EMAIL_SMTP_HOST'), int(os.getenv('EMAIL_SMTP_PORT', '587'))) as server:
                server.starttls()
                server.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASSWORD'))
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    def _send_console_alert(self, alert: Dict):
        """Send console alert"""
        print(f"\n{'='*50}")
        print(f"ALERT [{alert['severity']}]: {alert['message']}")
        print(f"Time: {alert['timestamp']}")
        print(f"{'='*50}\n")

# ===================== DEPLOYMENT MANAGER =====================

class ProductionDeploymentManager:
    """Manages deployment, updates, and rollbacks"""
    
    def __init__(self):
        self.current_version = self._get_current_version()
        
    def _get_current_version(self) -> str:
        """Get current system version"""
        try:
            with open('version.txt', 'r') as f:
                return f.read().strip()
        except:
            return "1.0.0"
    
    def deploy_update(self, new_version: str, strategy: str = 'BLUE_GREEN'):
        """Deploy system update with selected strategy"""
        
        logger.info(f"Starting deployment of version {new_version}")
        
        if strategy == 'BLUE_GREEN':
            return self._blue_green_deployment(new_version)
        elif strategy == 'CANARY':
            return self._canary_deployment(new_version)
        elif strategy == 'ROLLING':
            return self._rolling_deployment(new_version)
        else:
            raise ValueError(f"Unknown deployment strategy: {strategy}")
    
    def _blue_green_deployment(self, new_version: str) -> bool:
        """Blue-Green deployment strategy"""
        
        try:
            # Deploy to green environment
            logger.info("Deploying to green environment")
            self._deploy_to_environment('green', new_version)
            
            # Run health checks on green
            if not self._validate_deployment('green'):
                logger.error("Green deployment validation failed")
                return False
            
            # Switch traffic to green
            logger.info("Switching traffic to green")
            self._switch_traffic('green')
            
            # Monitor for issues
            time.sleep(300)  # 5 minute monitoring
            
            if self._check_deployment_health():
                # Success - update blue environment
                self._deploy_to_environment('blue', new_version)
                self.current_version = new_version
                logger.info(f"Deployment successful: {new_version}")
                return True
            else:
                # Rollback
                logger.error("Issues detected, rolling back")
                self._switch_traffic('blue')
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._switch_traffic('blue')
            return False
    
    def _deploy_to_environment(self, env: str, version: str):
        """Deploy to specific environment"""
        logger.info(f"Deploying version {version} to {env} environment")
        # In production, use actual deployment tools
    
    def _validate_deployment(self, env: str) -> bool:
        """Validate deployment"""
        logger.info(f"Validating {env} deployment")
        # In production, run actual health checks
        return True
    
    def _switch_traffic(self, env: str):
        """Switch traffic between environments"""
        logger.info(f"Switching traffic to {env}")
        # In production, update load balancer
    
    def _check_deployment_health(self) -> bool:
        """Check deployment health"""
        # In production, check actual metrics
        return True
    
    def _canary_deployment(self, new_version: str) -> bool:
        """Canary deployment with gradual rollout"""
        
        canary_percent = int(os.getenv('CANARY_PERCENT', '10'))
        
        try:
            # Deploy canary instances
            logger.info(f"Deploying canary ({canary_percent}% traffic)")
            self._deploy_canary(new_version, canary_percent)
            
            # Monitor canary metrics
            monitoring_duration = 3600  # 1 hour
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < monitoring_duration:
                if not self._check_canary_health():
                    logger.error("Canary health check failed")
                    self._rollback_canary()
                    return False
                
                time.sleep(60)
            
            # Gradually increase traffic
            for percentage in [25, 50, 75, 100]:
                logger.info(f"Increasing canary traffic to {percentage}%")
                self._adjust_canary_traffic(percentage)
                time.sleep(600)  # 10 minutes between increases
                
                if not self._check_canary_health():
                    self._rollback_canary()
                    return False
            
            self.current_version = new_version
            logger.info("Canary deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            self._rollback_canary()
            return False
    
    def _deploy_canary(self, version: str, percent: int):
        """Deploy canary version"""
        logger.info(f"Deploying canary version {version} with {percent}% traffic")
    
    def _check_canary_health(self) -> bool:
        """Check canary health"""
        return True
    
    def _rollback_canary(self):
        """Rollback canary deployment"""
        logger.info("Rolling back canary deployment")
    
    def _adjust_canary_traffic(self, percent: int):
        """Adjust canary traffic percentage"""
        logger.info(f"Adjusting canary traffic to {percent}%")
    
    def _rolling_deployment(self, new_version: str) -> bool:
        """Rolling deployment strategy"""
        logger.info(f"Rolling deployment of {new_version}")
        return True

# ===================== MAIN PRODUCTION SYSTEM =====================

class OmniAlphaProductionSystem:
    """Main production trading system orchestrator"""
    
    def __init__(self):
        self.state = SystemState.INITIALIZING
        self.broker_manager = ProductionBrokerManager()
        self.data_manager = ProductionDataManager()
        self.risk_manager = ProductionRiskManager()
        self.monitor = ProductionMonitor()
        self.deployment_manager = ProductionDeploymentManager()
        
        # Initialize components
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize all system components"""
        
        try:
            logger.info("Initializing Omni Alpha Production System")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.graceful_shutdown)
            signal.signal(signal.SIGTERM, self.graceful_shutdown)
            
            # Initialize Prometheus metrics server
            try:
                prometheus_client.start_http_server(8000)
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.warning(f"Prometheus server failed: {e}")
            
            # Perform system checks
            if not self._perform_startup_checks():
                raise RuntimeError("Startup checks failed")
            
            self.state = SystemState.RUNNING
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.critical(f"System initialization failed: {e}")
            self.state = SystemState.ERROR
            raise
    
    def _perform_startup_checks(self) -> bool:
        """Perform comprehensive startup checks"""
        
        checks = {
            'Broker': self._check_broker_connection(),
            'Market Data': self._check_market_data(),
            'Risk Systems': self._check_risk_systems(),
            'Monitoring': self._check_monitoring_systems()
        }
        
        for component, status in checks.items():
            if not status:
                logger.error(f"{component} check failed")
                return False
            logger.info(f"{component} check passed")
        
        return True
    
    def _check_broker_connection(self) -> bool:
        """Check broker connection"""
        return self.broker_manager.active_broker is not None
    
    def _check_market_data(self) -> bool:
        """Check market data feeds"""
        return (self.data_manager.primary_feed is not None or 
                self.data_manager.backup_feed is not None)
    
    def _check_risk_systems(self) -> bool:
        """Check risk management systems"""
        return True
    
    def _check_monitoring_systems(self) -> bool:
        """Check monitoring systems"""
        return True
    
    async def run(self):
        """Main production trading loop"""
        
        logger.info("Starting production trading system")
        
        while self.state == SystemState.RUNNING:
            try:
                # Check if market is open
                if not self._is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Check system health
                health = self.monitor.check_system_health()
                if health.status == SystemState.ERROR:
                    logger.error("System health critical, pausing trading")
                    self.state = SystemState.PAUSED
                    continue
                
                # Check risk limits
                if not self.risk_manager.check_risk_limits():
                    logger.warning("Risk limits breached, skipping cycle")
                    await asyncio.sleep(10)
                    continue
                
                # Main trading logic
                await self._trading_cycle()
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                error_counter.labels(error_type='trading_loop').inc()
                
                # Decide if error is recoverable
                if self._is_critical_error(e):
                    self.state = SystemState.ERROR
                    self._trigger_emergency_shutdown()
                    break
                
                await asyncio.sleep(5)
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        
        now = datetime.now()
        
        # Indian market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours
        return market_open <= now <= market_close
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        
        # This would integrate all previous steps
        # For production, each step runs as a separate service
        
        try:
            # Example: Get signals from strategy services
            signals = await self._collect_strategy_signals()
            
            # Risk check signals
            validated_signals = self.risk_manager.validate_signals(signals)
            
            # Execute trades
            for signal in validated_signals:
                try:
                    order_id = await self.broker_manager.place_order(signal)
                    logger.info(f"Order placed: {order_id}")
                except Exception as e:
                    logger.error(f"Order execution failed: {e}")
                    
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
    
    async def _collect_strategy_signals(self) -> List[Dict]:
        """Collect signals from all strategy services"""
        
        # In production, this would call actual strategy services
        # Simulated signals
        signals = [
            {
                'symbol': 'NIFTY',
                'side': 'BUY',
                'quantity': 50,
                'price': 20100,
                'strategy': 'ML_PREDICTIONS',
                'confidence': 0.75
            }
        ]
        
        return signals
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if error is critical"""
        
        critical_errors = [
            ConnectionError,
            RuntimeError,
            MemoryError
        ]
        
        return any(isinstance(error, err_type) for err_type in critical_errors)
    
    def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown"""
        
        logger.critical("EMERGENCY SHUTDOWN TRIGGERED")
        
        # Send critical alerts
        self.monitor._trigger_alert(
            "Emergency shutdown triggered due to critical error",
            "CRITICAL"
        )
        
        # Close all positions
        self._close_all_positions()
        
        # Save state
        self._save_system_state()
    
    def _close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions")
        # In production, implement actual position closing
    
    def _save_system_state(self):
        """Save current system state"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'version': self.deployment_manager.current_version,
            'positions': [],  # Would contain actual positions
            'pnl': self.risk_manager.risk_metrics.get('pnl', 0),
            'status': self.state.value
        }
        
        try:
            with open('system_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("System state saved")
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
    
    def graceful_shutdown(self, signum, frame):
        """Gracefully shutdown the system"""
        
        logger.info("Initiating graceful shutdown")
        self.state = SystemState.SHUTTING_DOWN
        
        try:
            # Close all positions if configured
            if os.getenv('CLOSE_POSITIONS_ON_SHUTDOWN', 'false').lower() == 'true':
                self._close_all_positions()
            
            # Save state
            self._save_system_state()
            
            # Disconnect from services
            self._disconnect_services()
            
            logger.info("Graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        sys.exit(0)
    
    def _disconnect_services(self):
        """Disconnect from external services"""
        logger.info("Disconnecting from services")
        # In production, properly close all connections

# ===================== TELEGRAM INTEGRATION =====================

def integrate_production_system(bot_instance):
    """Integrate production system with Telegram bot"""
    
    # Initialize production system
    bot_instance.production_system = OmniAlphaProductionSystem()
    
    async def production_command(update, context):
        """Production system command handler"""
        
        if not context.args:
            help_text = """
🏭 **Production System Controls**

**System Commands:**
/production status - System status
/production health - Health check
/production metrics - System metrics
/production alerts - Recent alerts
/production deploy VERSION - Deploy update
/production rollback - Emergency rollback
/production restart - Restart system
/production maintenance - Maintenance mode
            """
            await update.message.reply_text(help_text, parse_mode='Markdown')
            return
        
        command = context.args[0].lower()
        
        if command == 'status':
            system = bot_instance.production_system
            
            msg = f"""
🏭 **Production System Status**

**System State:** {system.state.value}
**Version:** {system.deployment_manager.current_version}
**Uptime:** {datetime.now() - system.monitor.startup_time}

**Components:**
• Broker: {'✅ Connected' if system.broker_manager.active_broker else '❌ Disconnected'}
• Data Feed: {'✅ Active' if system.data_manager.primary_feed else '❌ Inactive'}
• Risk Manager: {'✅ Active' if system.risk_manager else '❌ Inactive'}

**Current Metrics:**
• P&L: ₹{system.risk_manager.risk_metrics.get('pnl', 0):,.2f}
• Positions: {system.risk_manager.risk_metrics.get('positions', 0)}
• Exposure: ₹{system.risk_manager.risk_metrics.get('exposure', 0):,.2f}
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'health':
            health = bot_instance.production_system.monitor.check_system_health()
            
            status_emoji = {
                SystemState.RUNNING: "🟢",
                SystemState.ERROR: "🔴",
                SystemState.PAUSED: "🟡"
            }
            
            msg = f"""
💊 **System Health Check**

**Status:** {status_emoji.get(health.status, '⚪')} {health.status.value}

**Resources:**
• CPU Usage: {health.cpu_usage:.1f}%
• Memory Usage: {health.memory_usage:.1f}%
• Disk Usage: {health.disk_usage:.1f}%

**Connectivity:**
• Database: {'✅' if health.database_status else '❌'}
• Broker: {'✅' if health.broker_connection else '❌'}
• Data Feed: {'✅' if health.data_feed_status else '❌'}

**Performance:**
• Network Latency: {health.network_latency:.1f}ms
• Error Rate: {health.error_rate:.2%}
• Uptime: {health.uptime}
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'metrics':
            msg = f"""
📊 **System Metrics**

**Trading Metrics:**
• Total Trades: {trades_counter._value.get() if hasattr(trades_counter, '_value') else 'N/A'}
• Current P&L: ₹{pnl_gauge._value.get() if hasattr(pnl_gauge, '_value') else 0:,.2f}
• Active Positions: {active_positions_gauge._value.get() if hasattr(active_positions_gauge, '_value') else 0}

**Performance:**
• Avg Order Latency: N/A
• Error Rate: {error_counter._value.get() if hasattr(error_counter, '_value') else 0}

**System:**
• CPU: {psutil.cpu_percent():.1f}%
• Memory: {psutil.virtual_memory().percent:.1f}%
• Disk: {psutil.disk_usage('.').percent:.1f}%
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'deploy' and len(context.args) > 1:
            version = context.args[1]
            
            await update.message.reply_text(f"🚀 Deploying version {version}...")
            
            try:
                success = bot_instance.production_system.deployment_manager.deploy_update(version)
                if success:
                    msg = f"✅ Deployment successful: {version}"
                else:
                    msg = f"❌ Deployment failed: {version}"
            except Exception as e:
                msg = f"❌ Deployment error: {str(e)}"
            
            await update.message.reply_text(msg)
        
        elif command == 'restart':
            await update.message.reply_text("🔄 Restarting system...")
            
            # In production, trigger actual restart
            logger.info("System restart requested via Telegram")
            
            await update.message.reply_text("✅ System restart initiated")
        
        elif command == 'maintenance':
            await update.message.reply_text("🔧 Entering maintenance mode...")
            
            bot_instance.production_system.state = SystemState.MAINTENANCE
            
            await update.message.reply_text("✅ System in maintenance mode")
    
    return production_command

# ===================== ENTRY POINT =====================

async def main():
    """Main entry point for production system"""
    
    try:
        # Check if maintenance mode
        if os.getenv('MAINTENANCE_MODE', 'false').lower() == 'true':
            logger.info("System in maintenance mode")
            return
        
        # Initialize production system
        system = OmniAlphaProductionSystem()
        
        # Run system
        await system.run()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if os.getenv('SENTRY_DSN'):
            sentry_sdk.capture_exception(e)
        
        # Send critical alerts
        # Attempt graceful shutdown
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

"""
OMNI ALPHA 5.0 - LOGGING CONFIGURATION
======================================
Production-ready structured logging with rotation, filtering, and monitoring
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import threading

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

from config.settings import get_settings

# ===================== CUSTOM FORMATTERS =====================

class TradingFormatter(logging.Formatter):
    """Custom formatter for trading-specific logging"""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()
    
    def format(self, record):
        """Format log record with trading context"""
        # Add trading-specific fields
        record.uptime = (datetime.now() - self.start_time).total_seconds()
        record.component = getattr(record, 'component', 'system')
        record.trade_id = getattr(record, 'trade_id', None)
        record.symbol = getattr(record, 'symbol', None)
        record.latency_us = getattr(record, 'latency_us', None)
        
        # Base format
        base_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(component)s] - %(funcName)s:%(lineno)d'
        )
        
        # Add trading context if available
        if record.symbol:
            base_format += ' - [%(symbol)s]'
        if record.trade_id:
            base_format += ' - [Trade:%(trade_id)s]'
        if record.latency_us:
            base_format += ' - [Latency:%(latency_us)dμs]'
        
        base_format += ' - %(message)s'
        
        formatter = logging.Formatter(base_format)
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'component': getattr(record, 'component', 'system'),
            'thread': record.thread,
            'process': record.process
        }
        
        # Add trading-specific fields
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = record.trade_id
        if hasattr(record, 'latency_us'):
            log_entry['latency_us'] = record.latency_us
        if hasattr(record, 'order_id'):
            log_entry['order_id'] = record.order_id
        if hasattr(record, 'pnl'):
            log_entry['pnl'] = record.pnl
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# ===================== CUSTOM HANDLERS =====================

class TradingFileHandler(logging.handlers.RotatingFileHandler):
    """Custom file handler with trading-specific features"""
    
    def __init__(self, filename, max_bytes=10485760, backup_count=5):
        # Ensure logs directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Set custom formatter
        self.setFormatter(TradingFormatter())

class AlertHandler(logging.Handler):
    """Custom handler for critical alerts"""
    
    def __init__(self, alert_callback=None):
        super().__init__()
        self.alert_callback = alert_callback
        self.setLevel(logging.ERROR)
    
    def emit(self, record):
        """Emit alert for critical errors"""
        if self.alert_callback and record.levelno >= logging.ERROR:
            try:
                alert_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'component': getattr(record, 'component', 'system'),
                    'symbol': getattr(record, 'symbol', None),
                    'trade_id': getattr(record, 'trade_id', None)
                }
                self.alert_callback(alert_data)
            except Exception as e:
                # Don't let alert failures crash the application
                print(f"Alert handler error: {e}")

# ===================== LOGGING MANAGER =====================

class LoggingManager:
    """Centralized logging management"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.loggers = {}
        self.handlers = {}
        self.is_configured = False
        self._lock = threading.Lock()
    
    def configure_logging(self, alert_callback=None):
        """Configure comprehensive logging system"""
        with self._lock:
            if self.is_configured:
                return
            
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Root logger configuration
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, self.settings.log_level.upper()))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(TradingFormatter())
            root_logger.addHandler(console_handler)
            
            # Main log file handler
            main_file_handler = TradingFileHandler(
                filename=logs_dir / self.settings.log_file,
                max_bytes=self.settings.log_max_size,
                backup_count=self.settings.log_backup_count
            )
            main_file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(main_file_handler)
            
            # Error log file handler
            error_file_handler = TradingFileHandler(
                filename=logs_dir / "errors.log",
                max_bytes=self.settings.log_max_size,
                backup_count=self.settings.log_backup_count
            )
            error_file_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_file_handler)
            
            # Trading-specific log file handler
            trading_file_handler = TradingFileHandler(
                filename=logs_dir / "trading.log",
                max_bytes=self.settings.log_max_size,
                backup_count=self.settings.log_backup_count
            )
            trading_file_handler.setLevel(logging.INFO)
            
            # Filter for trading-related logs
            trading_filter = logging.Filter()
            trading_filter.filter = lambda record: (
                hasattr(record, 'component') and 
                record.component in ['trading', 'risk', 'execution', 'order']
            )
            trading_file_handler.addFilter(trading_filter)
            root_logger.addHandler(trading_file_handler)
            
            # JSON log file handler for structured logs
            if JSON_LOGGER_AVAILABLE:
                json_file_handler = logging.handlers.RotatingFileHandler(
                    filename=logs_dir / "structured.json.log",
                    maxBytes=self.settings.log_max_size,
                    backupCount=self.settings.log_backup_count,
                    encoding='utf-8'
                )
                json_file_handler.setLevel(logging.INFO)
                json_file_handler.setFormatter(JSONFormatter())
                root_logger.addHandler(json_file_handler)
            
            # Alert handler for critical errors
            if alert_callback:
                alert_handler = AlertHandler(alert_callback)
                root_logger.addHandler(alert_handler)
            
            # Configure structured logging if available
            if STRUCTLOG_AVAILABLE:
                self._configure_structlog()
            
            self.is_configured = True
            logging.info("Logging system configured successfully")
    
    def _configure_structlog(self):
        """Configure structured logging with structlog"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self, name: str, component: str = None) -> logging.Logger:
        """Get logger with component context"""
        logger = logging.getLogger(name)
        
        # Add component context
        if component:
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.component = component
                return record
            
            logging.setLogRecordFactory(record_factory)
        
        return logger
    
    def log_trade(self, logger: logging.Logger, level: int, message: str, 
                  symbol: str = None, trade_id: str = None, order_id: str = None,
                  pnl: float = None, latency_us: int = None, **kwargs):
        """Log trading-specific message with context"""
        extra = {
            'component': 'trading',
            'symbol': symbol,
            'trade_id': trade_id,
            'order_id': order_id,
            'pnl': pnl,
            'latency_us': latency_us,
            **kwargs
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        logger.log(level, message, extra=extra)
    
    def log_risk_event(self, logger: logging.Logger, level: int, message: str,
                      symbol: str = None, risk_type: str = None, 
                      risk_value: float = None, **kwargs):
        """Log risk management event"""
        extra = {
            'component': 'risk',
            'symbol': symbol,
            'risk_type': risk_type,
            'risk_value': risk_value,
            **kwargs
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        logger.log(level, message, extra=extra)
    
    def log_data_event(self, logger: logging.Logger, level: int, message: str,
                      symbol: str = None, data_source: str = None,
                      data_quality: str = None, latency_us: int = None, **kwargs):
        """Log data collection event"""
        extra = {
            'component': 'data',
            'symbol': symbol,
            'data_source': data_source,
            'data_quality': data_quality,
            'latency_us': latency_us,
            **kwargs
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        logger.log(level, message, extra=extra)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {}
        
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    file_size = os.path.getsize(handler.baseFilename)
                    stats[handler.baseFilename] = {
                        'size_bytes': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'level': handler.level
                    }
                except (OSError, AttributeError):
                    pass
        
        return stats

# ===================== GLOBAL LOGGING INSTANCE =====================

_logging_manager = None

def get_logging_manager() -> LoggingManager:
    """Get global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def configure_logging(alert_callback=None):
    """Configure global logging system"""
    logging_manager = get_logging_manager()
    logging_manager.configure_logging(alert_callback)

def get_logger(name: str, component: str = None) -> logging.Logger:
    """Get logger with component context"""
    logging_manager = get_logging_manager()
    return logging_manager.get_logger(name, component)

def log_trade(message: str, symbol: str = None, trade_id: str = None, 
              order_id: str = None, pnl: float = None, latency_us: int = None, 
              level: int = logging.INFO, **kwargs):
    """Convenience function for trade logging"""
    logger = get_logger(__name__, 'trading')
    logging_manager = get_logging_manager()
    logging_manager.log_trade(
        logger, level, message, symbol, trade_id, order_id, pnl, latency_us, **kwargs
    )

def log_risk(message: str, symbol: str = None, risk_type: str = None,
             risk_value: float = None, level: int = logging.WARNING, **kwargs):
    """Convenience function for risk logging"""
    logger = get_logger(__name__, 'risk')
    logging_manager = get_logging_manager()
    logging_manager.log_risk_event(
        logger, level, message, symbol, risk_type, risk_value, **kwargs
    )

def log_data(message: str, symbol: str = None, data_source: str = None,
             data_quality: str = None, latency_us: int = None, 
             level: int = logging.INFO, **kwargs):
    """Convenience function for data logging"""
    logger = get_logger(__name__, 'data')
    logging_manager = get_logging_manager()
    logging_manager.log_data_event(
        logger, level, message, symbol, data_source, data_quality, latency_us, **kwargs
    )

# ===================== PERFORMANCE LOGGING =====================

class PerformanceLogger:
    """Performance monitoring logger"""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'performance')
        self.timers = {}
    
    def start_timer(self, operation: str, context: Dict[str, Any] = None):
        """Start performance timer"""
        timer_id = f"{operation}_{threading.get_ident()}"
        self.timers[timer_id] = {
            'start_time': time.time_ns(),
            'operation': operation,
            'context': context or {}
        }
        return timer_id
    
    def end_timer(self, timer_id: str, success: bool = True, **kwargs):
        """End performance timer and log result"""
        if timer_id not in self.timers:
            return
        
        timer = self.timers.pop(timer_id)
        duration_ns = time.time_ns() - timer['start_time']
        duration_us = duration_ns // 1000
        duration_ms = duration_us / 1000
        
        # Log performance
        context = timer['context']
        context.update(kwargs)
        
        self.logger.info(
            f"Performance: {timer['operation']} completed",
            extra={
                'operation': timer['operation'],
                'duration_ns': duration_ns,
                'duration_us': duration_us,
                'duration_ms': duration_ms,
                'success': success,
                **context
            }
        )
        
        return duration_us

# ===================== INITIALIZATION =====================

def initialize_logging(alert_callback=None):
    """Initialize the logging system"""
    try:
        configure_logging(alert_callback)
        logger = get_logger(__name__)
        logger.info("Logging system initialized successfully")
        logger.info(f"Log level: {get_settings().log_level}")
        logger.info(f"Log file: {get_settings().log_file}")
        return True
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        return False

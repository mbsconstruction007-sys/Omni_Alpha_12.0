"""
OMNI ALPHA 5.0 - MONITORING SYSTEM
==================================
Comprehensive monitoring with Prometheus metrics, health checks, and performance tracking
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        start_http_server, CollectorRegistry, CONTENT_TYPE_LATEST,
        generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

# ===================== METRIC TYPES =====================

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class HealthStatus:
    """Component health status"""
    component: str
    status: ComponentStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)

# ===================== METRICS COLLECTOR =====================

class MetricsCollector:
    """Centralized metrics collection and export"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'monitoring')
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.metrics = {}
        self.custom_metrics = {}
        self._lock = threading.Lock()
        
        if PROMETHEUS_AVAILABLE and settings.monitoring.metrics_enabled:
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # Trading Metrics
        self.metrics['trades_total'] = Counter(
            'omni_alpha_trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'strategy'],
            registry=self.registry
        )
        
        self.metrics['trade_duration'] = Histogram(
            'omni_alpha_trade_duration_seconds',
            'Time taken to execute trades',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        self.metrics['portfolio_value'] = Gauge(
            'omni_alpha_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.metrics['positions_count'] = Gauge(
            'omni_alpha_positions_count',
            'Number of open positions',
            registry=self.registry
        )
        
        self.metrics['daily_pnl'] = Gauge(
            'omni_alpha_daily_pnl_usd',
            'Daily profit and loss in USD',
            registry=self.registry
        )
        
        # Data Collection Metrics
        self.metrics['data_points_collected'] = Counter(
            'omni_alpha_data_points_total',
            'Total data points collected',
            ['source', 'symbol', 'data_type'],
            registry=self.registry
        )
        
        self.metrics['data_latency'] = Histogram(
            'omni_alpha_data_latency_microseconds',
            'Data collection latency in microseconds',
            ['source', 'operation'],
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000],
            registry=self.registry
        )
        
        self.metrics['data_quality_score'] = Gauge(
            'omni_alpha_data_quality_score',
            'Data quality score (0-1)',
            ['symbol', 'source'],
            registry=self.registry
        )
        
        # System Metrics
        self.metrics['system_health_score'] = Gauge(
            'omni_alpha_system_health_score',
            'Overall system health score (0-1)',
            registry=self.registry
        )
        
        self.metrics['api_requests_total'] = Counter(
            'omni_alpha_api_requests_total',
            'Total API requests made',
            ['service', 'method', 'status'],
            registry=self.registry
        )
        
        self.metrics['errors_total'] = Counter(
            'omni_alpha_errors_total',
            'Total errors encountered',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Risk Metrics
        self.metrics['risk_score'] = Gauge(
            'omni_alpha_risk_score',
            'Current portfolio risk score',
            registry=self.registry
        )
        
        self.metrics['drawdown_percent'] = Gauge(
            'omni_alpha_drawdown_percent',
            'Current drawdown percentage',
            registry=self.registry
        )
        
        # Latency Metrics
        self.metrics['order_latency'] = Histogram(
            'omni_alpha_order_latency_microseconds',
            'Order execution latency in microseconds',
            ['operation'],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
            registry=self.registry
        )
        
        self.logger.info(f"Initialized {len(self.metrics)} Prometheus metrics")
    
    def record_trade(self, symbol: str, side: str, strategy: str = 'default', 
                    duration: float = None, **labels):
        """Record trade execution"""
        if 'trades_total' in self.metrics:
            self.metrics['trades_total'].labels(
                symbol=symbol, 
                side=side, 
                strategy=strategy
            ).inc()
        
        if duration and 'trade_duration' in self.metrics:
            self.metrics['trade_duration'].labels(
                symbol=symbol, 
                side=side
            ).observe(duration)
        
        self.logger.info(
            f"Trade recorded: {side} {symbol}",
            extra={'symbol': symbol, 'side': side, 'strategy': strategy}
        )
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value metric"""
        if 'portfolio_value' in self.metrics:
            self.metrics['portfolio_value'].set(value)
    
    def update_positions_count(self, count: int):
        """Update positions count metric"""
        if 'positions_count' in self.metrics:
            self.metrics['positions_count'].set(count)
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L metric"""
        if 'daily_pnl' in self.metrics:
            self.metrics['daily_pnl'].set(pnl)
    
    def record_data_point(self, source: str, symbol: str, data_type: str):
        """Record data point collection"""
        if 'data_points_collected' in self.metrics:
            self.metrics['data_points_collected'].labels(
                source=source,
                symbol=symbol,
                data_type=data_type
            ).inc()
    
    def record_data_latency(self, source: str, operation: str, latency_us: float):
        """Record data collection latency"""
        if 'data_latency' in self.metrics:
            self.metrics['data_latency'].labels(
                source=source,
                operation=operation
            ).observe(latency_us)
    
    def update_data_quality(self, symbol: str, source: str, quality_score: float):
        """Update data quality score"""
        if 'data_quality_score' in self.metrics:
            self.metrics['data_quality_score'].labels(
                symbol=symbol,
                source=source
            ).set(quality_score)
    
    def update_system_health(self, health_score: float):
        """Update overall system health score"""
        if 'system_health_score' in self.metrics:
            self.metrics['system_health_score'].set(health_score)
    
    def record_api_request(self, service: str, method: str, status: str):
        """Record API request"""
        if 'api_requests_total' in self.metrics:
            self.metrics['api_requests_total'].labels(
                service=service,
                method=method,
                status=status
            ).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence"""
        if 'errors_total' in self.metrics:
            self.metrics['errors_total'].labels(
                component=component,
                error_type=error_type
            ).inc()
    
    def record_order_latency(self, operation: str, latency_us: float):
        """Record order execution latency"""
        if 'order_latency' in self.metrics:
            self.metrics['order_latency'].labels(operation=operation).observe(latency_us)
    
    def update_risk_score(self, risk_score: float):
        """Update risk score"""
        if 'risk_score' in self.metrics:
            self.metrics['risk_score'].set(risk_score)
    
    def update_drawdown(self, drawdown_percent: float):
        """Update drawdown percentage"""
        if 'drawdown_percent' in self.metrics:
            self.metrics['drawdown_percent'].set(drawdown_percent)
    
    def create_custom_metric(self, name: str, description: str, 
                           metric_type: MetricType, labels: List[str] = None):
        """Create custom metric"""
        if not PROMETHEUS_AVAILABLE:
            return None
        
        labels = labels or []
        
        try:
            if metric_type == MetricType.COUNTER:
                metric = Counter(name, description, labels, registry=self.registry)
            elif metric_type == MetricType.GAUGE:
                metric = Gauge(name, description, labels, registry=self.registry)
            elif metric_type == MetricType.HISTOGRAM:
                metric = Histogram(name, description, labels, registry=self.registry)
            elif metric_type == MetricType.SUMMARY:
                metric = Summary(name, description, labels, registry=self.registry)
            else:
                return None
            
            self.custom_metrics[name] = metric
            self.logger.info(f"Created custom metric: {name}")
            return metric
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metric {name}: {e}")
            return None
    
    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output"""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def start_metrics_server(self, port: int = None):
        """Start Prometheus metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus not available, metrics server not started")
            return False
        
        if port is None:
            port = self.settings.monitoring.metrics_port
        
        try:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Metrics server started on port {port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            return False

# ===================== HEALTH MONITOR =====================

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'health')
        self.component_status: Dict[str, HealthStatus] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.is_monitoring = False
        self._lock = threading.Lock()
    
    def register_health_check(self, component: str, check_func: Callable):
        """Register health check function for component"""
        self.health_checks[component] = check_func
        self.logger.info(f"Registered health check for {component}")
    
    async def check_component_health(self, component: str) -> HealthStatus:
        """Check health of specific component"""
        if component not in self.health_checks:
            return HealthStatus(
                component=component,
                status=ComponentStatus.UNKNOWN,
                message="No health check registered",
                timestamp=datetime.now()
            )
        
        try:
            check_func = self.health_checks[component]
            
            # Run health check with timeout
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=10)
            else:
                result = check_func()
            
            response_time = (time.time() - start_time) * 1000
            
            # Parse result
            if isinstance(result, bool):
                status = ComponentStatus.HEALTHY if result else ComponentStatus.CRITICAL
                message = "OK" if result else "Failed"
                metrics = {'response_time_ms': response_time}
            elif isinstance(result, dict):
                status = ComponentStatus(result.get('status', 'unknown'))
                message = result.get('message', 'No message')
                metrics = result.get('metrics', {})
                metrics['response_time_ms'] = response_time
            else:
                status = ComponentStatus.UNKNOWN
                message = str(result)
                metrics = {'response_time_ms': response_time}
            
            health_status = HealthStatus(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
            
            with self._lock:
                self.component_status[component] = health_status
            
            return health_status
            
        except asyncio.TimeoutError:
            health_status = HealthStatus(
                component=component,
                status=ComponentStatus.CRITICAL,
                message="Health check timeout",
                timestamp=datetime.now(),
                metrics={'timeout': True}
            )
            
            with self._lock:
                self.component_status[component] = health_status
            
            return health_status
            
        except Exception as e:
            health_status = HealthStatus(
                component=component,
                status=ComponentStatus.CRITICAL,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now(),
                metrics={'error': str(e)}
            )
            
            with self._lock:
                self.component_status[component] = health_status
            
            self.logger.error(f"Health check failed for {component}: {e}")
            return health_status
    
    async def check_all_components(self) -> Dict[str, HealthStatus]:
        """Check health of all registered components"""
        tasks = []
        
        for component in self.health_checks.keys():
            tasks.append(self.check_component_health(component))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            health_results = {}
            for i, (component, result) in enumerate(zip(self.health_checks.keys(), results)):
                if isinstance(result, Exception):
                    health_results[component] = HealthStatus(
                        component=component,
                        status=ComponentStatus.CRITICAL,
                        message=f"Exception: {str(result)}",
                        timestamp=datetime.now()
                    )
                else:
                    health_results[component] = result
            
            return health_results
        
        return {}
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        with self._lock:
            if not self.component_status:
                return {
                    'status': 'UNKNOWN',
                    'score': 0.0,
                    'components': {},
                    'summary': 'No health checks available'
                }
            
            # Calculate health score
            status_scores = {
                ComponentStatus.HEALTHY: 1.0,
                ComponentStatus.DEGRADED: 0.5,
                ComponentStatus.CRITICAL: 0.0,
                ComponentStatus.UNKNOWN: 0.25
            }
            
            total_score = 0.0
            component_count = len(self.component_status)
            
            for status in self.component_status.values():
                total_score += status_scores.get(status.status, 0.0)
            
            overall_score = total_score / component_count if component_count > 0 else 0.0
            
            # Determine overall status
            if overall_score >= 0.8:
                overall_status = "HEALTHY"
            elif overall_score >= 0.5:
                overall_status = "DEGRADED"
            else:
                overall_status = "CRITICAL"
            
            return {
                'status': overall_status,
                'score': overall_score,
                'components': {
                    comp: {
                        'status': status.status.value,
                        'message': status.message,
                        'timestamp': status.timestamp.isoformat(),
                        'metrics': status.metrics
                    }
                    for comp, status in self.component_status.items()
                },
                'summary': f"{component_count} components monitored, {overall_score:.1%} healthy"
            }
    
    async def start_monitoring(self, interval: int = None):
        """Start continuous health monitoring"""
        if interval is None:
            interval = self.settings.monitoring.health_check_interval
        
        self.is_monitoring = True
        self.logger.info(f"Starting health monitoring (interval: {interval}s)")
        
        while self.is_monitoring:
            try:
                await self.check_all_components()
                
                # Update system health metric
                overall_health = self.get_overall_health()
                
                # Log health status
                if overall_health['status'] != 'HEALTHY':
                    self.logger.warning(f"System health: {overall_health['status']} ({overall_health['score']:.1%})")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        self.logger.info("Health monitoring stopped")

# ===================== PERFORMANCE TRACKER =====================

class PerformanceTracker:
    """Track and analyze system performance"""
    
    def __init__(self):
        self.logger = get_logger(__name__, 'performance')
        self.operation_times = defaultdict(lambda: deque(maxlen=1000))
        self.active_operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation: str, context: Dict[str, Any] = None) -> str:
        """Start tracking operation performance"""
        operation_id = f"{operation}_{threading.get_ident()}_{time.time_ns()}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                'operation': operation,
                'start_time': time.time_ns(),
                'context': context or {}
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, **kwargs) -> Optional[float]:
        """End operation tracking and return duration in microseconds"""
        with self._lock:
            if operation_id not in self.active_operations:
                return None
            
            op_data = self.active_operations.pop(operation_id)
            
        duration_ns = time.time_ns() - op_data['start_time']
        duration_us = duration_ns / 1000
        
        # Store performance data
        self.operation_times[op_data['operation']].append(duration_us)
        
        # Log performance
        context = op_data['context']
        context.update(kwargs)
        
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(
            log_level,
            f"Operation {op_data['operation']} completed in {duration_us:.1f}μs",
            extra={
                'operation': op_data['operation'],
                'duration_us': duration_us,
                'success': success,
                **context
            }
        )
        
        return duration_us
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for operation"""
        with self._lock:
            times = list(self.operation_times[operation])
        
        if not times:
            return {}
        
        import numpy as np
        
        return {
            'count': len(times),
            'mean_us': np.mean(times),
            'median_us': np.median(times),
            'p95_us': np.percentile(times, 95),
            'p99_us': np.percentile(times, 99),
            'min_us': min(times),
            'max_us': max(times),
            'std_us': np.std(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self.operation_times.keys()
        }

# ===================== MONITORING MANAGER =====================

class MonitoringManager:
    """Main monitoring system orchestrator"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'monitoring')
        
        # Initialize components
        self.metrics_collector = MetricsCollector(settings)
        self.health_monitor = HealthMonitor(settings)
        self.performance_tracker = PerformanceTracker()
        
        self.is_started = False
    
    async def start(self):
        """Start monitoring system"""
        try:
            self.logger.info("Starting monitoring system...")
            
            # Start metrics server
            if self.settings.monitoring.metrics_enabled:
                success = self.metrics_collector.start_metrics_server()
                if not success:
                    self.logger.warning("Metrics server failed to start")
            
            # Start health monitoring
            if self.settings.monitoring.health_check_enabled:
                asyncio.create_task(self.health_monitor.start_monitoring())
            
            self.is_started = True
            self.logger.info("Monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            raise
    
    def register_component_health_check(self, component: str, check_func: Callable):
        """Register health check for component"""
        self.health_monitor.register_health_check(component, check_func)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health = self.health_monitor.get_overall_health()
        performance = self.performance_tracker.get_all_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_enabled': self.is_started,
            'health': health,
            'performance': performance,
            'metrics_available': PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled
        }
    
    async def stop(self):
        """Stop monitoring system"""
        self.logger.info("Stopping monitoring system...")
        
        self.health_monitor.stop_monitoring()
        self.is_started = False
        
        self.logger.info("Monitoring system stopped")

# ===================== GLOBAL MONITORING INSTANCE =====================

_monitoring_manager = None

def get_monitoring_manager() -> MonitoringManager:
    """Get global monitoring manager instance"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager

async def start_monitoring():
    """Start global monitoring system"""
    manager = get_monitoring_manager()
    await manager.start()

async def stop_monitoring():
    """Stop global monitoring system"""
    global _monitoring_manager
    if _monitoring_manager:
        await _monitoring_manager.stop()
        _monitoring_manager = None

def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return get_monitoring_manager().metrics_collector

def get_health_monitor() -> HealthMonitor:
    """Get health monitor instance"""
    return get_monitoring_manager().health_monitor

def get_performance_tracker() -> PerformanceTracker:
    """Get performance tracker instance"""
    return get_monitoring_manager().performance_tracker

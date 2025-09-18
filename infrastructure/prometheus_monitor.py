from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from typing import Dict, Any
import threading

logger = logging.getLogger(__name__)

class PrometheusMonitor:
    """Simple Prometheus monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = config.get('PROMETHEUS_PORT', 8001)
        self.server_started = False
        
        # Define metrics
        self.trade_counter = Counter('trades_total', 'Total number of trades')
        self.error_counter = Counter('errors_total', 'Total number of errors', ['component'])
        self.latency_histogram = Histogram('request_latency_seconds', 'Request latency')
        self.system_health = Gauge('system_health', 'System health score (0-1)')
        self.active_connections = Gauge('active_connections', 'Active connections', ['type'])
        
    def start_server(self):
        """Start Prometheus metrics server"""
        if not self.server_started:
            try:
                start_http_server(self.port)
                self.server_started = True
                logger.info(f"Prometheus server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
                
    def record_trade(self):
        """Record a trade"""
        self.trade_counter.inc()
        
    def record_error(self, component: str):
        """Record an error"""
        self.error_counter.labels(component=component).inc()
        
    def record_latency(self, latency: float):
        """Record request latency"""
        self.latency_histogram.observe(latency)
        
    def update_health(self, health_score: float):
        """Update system health score"""
        self.system_health.set(health_score)
        
    def update_connections(self, connection_type: str, count: int):
        """Update connection count"""
        self.active_connections.labels(type=connection_type).set(count)

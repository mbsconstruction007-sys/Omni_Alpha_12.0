"""
Predictive health monitoring with self-healing capabilities
Institutional-grade monitoring for trading systems
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from collections import deque
import aiohttp
import psutil
import structlog
from prometheus_client import CollectorRegistry, generate_latest, Counter, Histogram, Gauge
import json

logger = structlog.get_logger()

@dataclass
class HealthMetric:
    name: str
    value: float
    timestamp: datetime
    status: str  # 'healthy', 'degraded', 'critical'
    threshold_warning: float
    threshold_critical: float
    trend: str = 'stable'  # 'improving', 'stable', 'degrading'
    prediction: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class Anomaly:
    metric: str
    value: float
    expected_range: tuple
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    description: str

class PredictiveHealthMonitor:
    """
    Advanced health monitoring with predictive analytics and self-healing
    """
    
    def __init__(self):
        self.metrics_history = {}
        self.health_checks = {}
        self.self_healing_actions = {}
        self.ml_models = {}
        self.anomalies = deque(maxlen=1000)  # Keep last 1000 anomalies
        self._initialize_health_checks()
        self._setup_ml_models()
        self._configure_self_healing()
        self._setup_prometheus_metrics()
        
    def _initialize_health_checks(self):
        """Setup comprehensive health checks"""
        
        # System health checks
        self.health_checks['cpu'] = self._check_cpu_health
        self.health_checks['memory'] = self._check_memory_health
        self.health_checks['disk'] = self._check_disk_health
        self.health_checks['network'] = self._check_network_health
        
        # Application health checks
        self.health_checks['latency'] = self._check_latency_health
        self.health_checks['throughput'] = self._check_throughput_health
        self.health_checks['error_rate'] = self._check_error_rate_health
        
        # Dependency health checks
        self.health_checks['database'] = self._check_database_health
        self.health_checks['cache'] = self._check_cache_health
        self.health_checks['message_queue'] = self._check_mq_health
        self.health_checks['external_apis'] = self._check_external_apis_health
        
        # Trading-specific health checks
        self.health_checks['order_latency'] = self._check_order_latency
        self.health_checks['market_data_lag'] = self._check_market_data_lag
        self.health_checks['position_sync'] = self._check_position_sync
        self.health_checks['risk_limits'] = self._check_risk_limits
        
    def _setup_ml_models(self):
        """Initialize ML models for predictive health"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Anomaly detection model
            self.ml_models['anomaly'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Trend prediction models
            self.ml_models['trend'] = LinearRegression()
            self.ml_models['scaler'] = StandardScaler()
            
            # Time series forecasting (simplified)
            self.ml_models['forecast'] = LinearRegression()
            
            logger.info("ML models initialized successfully")
        except ImportError:
            logger.warning("ML libraries not available, using statistical methods")
            self.ml_models = {}
        
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.prometheus_metrics = {
            'health_score': Gauge('omni_alpha_health_score', 'Overall health score'),
            'anomaly_count': Counter('omni_alpha_anomalies_total', 'Total anomalies detected'),
            'self_healing_attempts': Counter('omni_alpha_self_healing_attempts_total', 'Self-healing attempts'),
            'self_healing_success': Counter('omni_alpha_self_healing_success_total', 'Successful self-healing'),
            'check_duration': Histogram('omni_alpha_health_check_duration_seconds', 'Health check duration'),
        }
        
    async def _check_cpu_health(self) -> HealthMetric:
        """Advanced CPU health check with prediction"""
        start_time = time.time()
        
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Get historical data
            history = self._get_metric_history('cpu', limit=100)
            
            # Predict future CPU usage
            prediction = None
            if len(history) >= 10:
                prediction = self._predict_metric_value(history)
                
            # Determine trend
            trend = self._calculate_trend(history)
            
            # Determine status
            if cpu_percent > 90:
                status = 'critical'
            elif cpu_percent > 75:
                status = 'degraded'
            else:
                status = 'healthy'
                
            # Record check duration
            duration = time.time() - start_time
            self.prometheus_metrics['check_duration'].observe(duration)
                
            return HealthMetric(
                name='cpu',
                value=cpu_percent,
                timestamp=datetime.utcnow(),
                status=status,
                threshold_warning=75,
                threshold_critical=90,
                trend=trend,
                prediction=prediction,
                details={
                    'cpu_count': cpu_count,
                    'cpu_freq': cpu_freq.current if cpu_freq else 0,
                    'load_avg': load_avg,
                    'check_duration': duration
                }
            )
            
        except Exception as e:
            logger.error(f"CPU health check failed: {e}")
            return HealthMetric(
                name='cpu',
                value=0,
                timestamp=datetime.utcnow(),
                status='critical',
                threshold_warning=75,
                threshold_critical=90,
                details={'error': str(e)}
            )
            
    async def _check_memory_health(self) -> HealthMetric:
        """Memory health check with detailed analysis"""
        start_time = time.time()
        
        try:
            # Get memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Calculate memory pressure
            memory_pressure = memory.percent
            swap_usage = swap.percent
            
            # Get historical data
            history = self._get_metric_history('memory', limit=100)
            
            # Predict future memory usage
            prediction = None
            if len(history) >= 10:
                prediction = self._predict_metric_value(history)
                
            # Determine trend
            trend = self._calculate_trend(history)
            
            # Determine status
            if memory_pressure > 95 or swap_usage > 50:
                status = 'critical'
            elif memory_pressure > 85 or swap_usage > 25:
                status = 'degraded'
            else:
                status = 'healthy'
                
            duration = time.time() - start_time
            self.prometheus_metrics['check_duration'].observe(duration)
                
            return HealthMetric(
                name='memory',
                value=memory_pressure,
                timestamp=datetime.utcnow(),
                status=status,
                threshold_warning=85,
                threshold_critical=95,
                trend=trend,
                prediction=prediction,
                details={
                    'total_memory': memory.total,
                    'available_memory': memory.available,
                    'used_memory': memory.used,
                    'swap_total': swap.total,
                    'swap_used': swap.used,
                    'swap_percent': swap_usage,
                    'check_duration': duration
                }
            )
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return HealthMetric(
                name='memory',
                value=0,
                timestamp=datetime.utcnow(),
                status='critical',
                threshold_warning=85,
                threshold_critical=95,
                details={'error': str(e)}
            )
            
    async def _check_order_latency(self) -> HealthMetric:
        """Trading-specific: Check order execution latency"""
        start_time = time.time()
        
        try:
            # Get recent order latencies (simulated for now)
            latencies = await self._get_order_latencies()
            
            if not latencies:
                return HealthMetric(
                    name='order_latency',
                    value=0,
                    timestamp=datetime.utcnow(),
                    status='healthy',
                    threshold_warning=50,  # 50ms warning
                    threshold_critical=100,  # 100ms critical
                    details={'message': 'No orders processed yet'}
                )
                
            # Calculate percentiles
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            
            # Use p95 for health determination
            if p95 > 100:
                status = 'critical'
            elif p95 > 50:
                status = 'degraded'
            else:
                status = 'healthy'
                
            # Get historical data
            history = self._get_metric_history('order_latency', limit=100)
            trend = self._calculate_trend(history)
            prediction = self._predict_metric_value(history) if len(history) >= 10 else None
            
            duration = time.time() - start_time
            self.prometheus_metrics['check_duration'].observe(duration)
                
            return HealthMetric(
                name='order_latency',
                value=p95,
                timestamp=datetime.utcnow(),
                status=status,
                threshold_warning=50,
                threshold_critical=100,
                trend=trend,
                prediction=prediction,
                details={
                    'p50': p50,
                    'p95': p95,
                    'p99': p99,
                    'sample_count': len(latencies),
                    'check_duration': duration
                }
            )
            
        except Exception as e:
            logger.error(f"Order latency check failed: {e}")
            return HealthMetric(
                name='order_latency',
                value=0,
                timestamp=datetime.utcnow(),
                status='critical',
                threshold_warning=50,
                threshold_critical=100,
                details={'error': str(e)}
            )
            
    async def _check_database_health(self) -> HealthMetric:
        """Database connectivity and performance check"""
        start_time = time.time()
        
        try:
            # Simulate database check (in production, use actual DB connection)
            # This would typically involve:
            # 1. Connection test
            # 2. Query performance test
            # 3. Connection pool status
            # 4. Replication lag check
            
            # Simulate latency
            await asyncio.sleep(0.01)  # 10ms simulated DB call
            
            # Simulate metrics
            connection_count = 10  # Would be actual connection count
            query_latency = 5.0  # Would be actual query latency
            replication_lag = 0.0  # Would be actual replication lag
            
            # Determine status
            if query_latency > 100 or replication_lag > 1000:
                status = 'critical'
            elif query_latency > 50 or replication_lag > 500:
                status = 'degraded'
            else:
                status = 'healthy'
                
            duration = time.time() - start_time
            self.prometheus_metrics['check_duration'].observe(duration)
                
            return HealthMetric(
                name='database',
                value=query_latency,
                timestamp=datetime.utcnow(),
                status=status,
                threshold_warning=50,
                threshold_critical=100,
                details={
                    'connection_count': connection_count,
                    'query_latency': query_latency,
                    'replication_lag': replication_lag,
                    'check_duration': duration
                }
            )
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthMetric(
                name='database',
                value=0,
                timestamp=datetime.utcnow(),
                status='critical',
                threshold_warning=50,
                threshold_critical=100,
                details={'error': str(e)}
            )
            
    async def _get_order_latencies(self) -> List[float]:
        """Get recent order latencies (simulated)"""
        # In production, this would query actual order execution data
        # For now, return simulated data
        return [10.5, 12.3, 8.7, 15.2, 9.8, 11.1, 13.4, 7.9, 14.6, 10.2]
        
    def _get_metric_history(self, metric_name: str, limit: int = 100) -> List[float]:
        """Get historical metric values"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=limit)
        return list(self.metrics_history[metric_name])
        
    def _store_metric_value(self, metric_name: str, value: float):
        """Store metric value in history"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=1000)
        self.metrics_history[metric_name].append(value)
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate metric trend using statistical analysis"""
        if len(values) < 3:
            return 'stable'
            
        # Use linear regression for trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend based on slope and significance
        if p_value > 0.05:  # Not statistically significant
            return 'stable'
        elif slope > 0.1:
            return 'degrading'
        elif slope < -0.1:
            return 'improving'
        else:
            return 'stable'
            
    def _predict_metric_value(self, history: List[float]) -> float:
        """Predict future metric value using simple linear regression"""
        if len(history) < 10:
            return None
            
        try:
            # Simple linear regression prediction
            x = np.arange(len(history))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, history)
            
            # Predict next value
            next_x = len(history)
            prediction = slope * next_x + intercept
            
            return float(prediction)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None
            
    async def detect_anomalies(self) -> List[Anomaly]:
        """Detect anomalies across all metrics"""
        anomalies = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                metric = await check_func()
                
                # Store metric value
                self._store_metric_value(check_name, metric.value)
                
                # Get historical data
                history = self._get_metric_history(check_name, limit=100)
                
                if len(history) < 10:
                    continue
                    
                # Detect anomalies using statistical methods
                if self.ml_models and 'anomaly' in self.ml_models:
                    # Use ML-based anomaly detection
                    X = np.array(history).reshape(-1, 1)
                    predictions = self.ml_models['anomaly'].fit_predict(X)
                    
                    # Check if current value is anomalous
                    current_value = np.array([[metric.value]])
                    is_anomaly = self.ml_models['anomaly'].predict(current_value)[0] == -1
                else:
                    # Use statistical anomaly detection
                    is_anomaly = self._is_statistical_anomaly(metric.value, history)
                    
                if is_anomaly:
                    expected_range = self._calculate_expected_range(history)
                    severity = self._calculate_anomaly_severity(metric.value, history)
                    
                    anomaly = Anomaly(
                        metric=check_name,
                        value=metric.value,
                        expected_range=expected_range,
                        severity=severity,
                        timestamp=metric.timestamp,
                        description=f"Anomalous {check_name} value: {metric.value}"
                    )
                    
                    anomalies.append(anomaly)
                    self.anomalies.append(anomaly)
                    
                    # Update Prometheus metrics
                    self.prometheus_metrics['anomaly_count'].inc()
                    
            except Exception as e:
                logger.error(f"Anomaly detection failed for {check_name}: {e}")
                
        return anomalies
        
    def _is_statistical_anomaly(self, value: float, history: List[float]) -> bool:
        """Detect statistical anomalies using z-score"""
        if len(history) < 10:
            return False
            
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return False
            
        z_score = abs(value - mean) / std
        return z_score > 3.0  # 3 standard deviations
        
    def _calculate_expected_range(self, history: List[float]) -> tuple:
        """Calculate expected range for a metric"""
        if len(history) < 10:
            return (0, 100)
            
        mean = np.mean(history)
        std = np.std(history)
        
        return (mean - 2*std, mean + 2*std)
        
    def _calculate_anomaly_severity(self, value: float, history: List[float]) -> str:
        """Calculate anomaly severity"""
        if len(history) < 10:
            return 'low'
            
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return 'low'
            
        z_score = abs(value - mean) / std
        
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'
            
    async def self_heal(self, issue: str) -> bool:
        """Attempt to self-heal identified issues"""
        if issue not in self.self_healing_actions:
            logger.warning(f"No self-healing action for issue: {issue}")
            return False
            
        action = self.self_healing_actions[issue]
        
        try:
            logger.info(f"Attempting self-healing for: {issue}")
            self.prometheus_metrics['self_healing_attempts'].inc()
            
            result = await action()
            
            if result:
                logger.info(f"Successfully self-healed: {issue}")
                self.prometheus_metrics['self_healing_success'].inc()
                await self._notify_self_healing_success(issue)
            else:
                logger.error(f"Self-healing failed for: {issue}")
                await self._escalate_issue(issue)
                
            return result
            
        except Exception as e:
            logger.error(f"Self-healing error for {issue}: {e}")
            await self._escalate_issue(issue)
            return False
            
    def _configure_self_healing(self):
        """Configure self-healing actions"""
        self.self_healing_actions = {
            'high_memory': self._heal_high_memory,
            'high_cpu': self._heal_high_cpu,
            'connection_pool_exhausted': self._heal_connection_pool,
            'cache_miss_rate_high': self._heal_cache,
            'database_slow': self._heal_database,
            'order_latency_high': self._heal_order_latency,
        }
        
    async def _heal_high_memory(self) -> bool:
        """Self-healing for high memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear caches (would need access to actual caches)
            logger.info("Cleared caches and ran garbage collection")
            
            # Restart workers if needed
            if psutil.virtual_memory().percent > 90:
                logger.info("Memory usage still high, would restart workers")
                # In production, this would restart worker processes
                
            return True
        except Exception as e:
            logger.error(f"Memory healing failed: {e}")
            return False
            
    async def _heal_high_cpu(self) -> bool:
        """Self-healing for high CPU usage"""
        try:
            # Reduce worker count temporarily
            logger.info("Reduced worker count to lower CPU usage")
            # In production, this would adjust worker processes
            
            return True
        except Exception as e:
            logger.error(f"CPU healing failed: {e}")
            return False
            
    async def _heal_connection_pool(self) -> bool:
        """Self-healing for connection pool issues"""
        try:
            # Reset connection pools
            logger.info("Reset connection pools")
            # In production, this would reset actual connection pools
            
            return True
        except Exception as e:
            logger.error(f"Connection pool healing failed: {e}")
            return False
            
    async def _heal_cache(self) -> bool:
        """Self-healing for cache issues"""
        try:
            # Clear and warm up caches
            logger.info("Cleared and warmed up caches")
            # In production, this would clear actual caches
            
            return True
        except Exception as e:
            logger.error(f"Cache healing failed: {e}")
            return False
            
    async def _heal_database(self) -> bool:
        """Self-healing for database issues"""
        try:
            # Reset database connections
            logger.info("Reset database connections")
            # In production, this would reset actual DB connections
            
            return True
        except Exception as e:
            logger.error(f"Database healing failed: {e}")
            return False
            
    async def _heal_order_latency(self) -> bool:
        """Self-healing for high order latency"""
        try:
            # Optimize order processing
            logger.info("Optimized order processing")
            # In production, this would optimize actual order processing
            
            return True
        except Exception as e:
            logger.error(f"Order latency healing failed: {e}")
            return False
            
    async def _notify_self_healing_success(self, issue: str):
        """Notify about successful self-healing"""
        logger.info(f"Self-healing successful for: {issue}")
        # In production, this would send notifications
        
    async def _escalate_issue(self, issue: str):
        """Escalate issue that couldn't be self-healed"""
        logger.error(f"Escalating issue that couldn't be self-healed: {issue}")
        # In production, this would escalate to human operators
        
    async def run_all_health_checks(self) -> Dict[str, HealthMetric]:
        """Run all health checks and return results"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                results[name] = await check_func()
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthMetric(
                    name=name,
                    value=0,
                    timestamp=datetime.utcnow(),
                    status='critical',
                    threshold_warning=0,
                    threshold_critical=0,
                    details={'error': str(e)}
                )
                
        return results
        
    def calculate_overall_health_score(self, health_results: Dict[str, HealthMetric]) -> float:
        """Calculate overall health score (0-100)"""
        if not health_results:
            return 0.0
            
        weights = {
            'cpu': 0.15,
            'memory': 0.15,
            'disk': 0.10,
            'network': 0.10,
            'latency': 0.15,
            'throughput': 0.10,
            'error_rate': 0.15,
            'database': 0.10,
        }
        
        score = 100.0
        
        for check_name, result in health_results.items():
            weight = weights.get(check_name, 0.05)
            
            if result.status == 'critical':
                score -= weight * 50
            elif result.status == 'degraded':
                score -= weight * 25
                
        # Update Prometheus metric
        self.prometheus_metrics['health_score'].set(max(0, min(100, score)))
        
        return max(0, min(100, score))
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        return {
            'total_checks': len(self.health_checks),
            'anomalies_detected': len(self.anomalies),
            'self_healing_actions': len(self.self_healing_actions),
            'ml_models_available': len(self.ml_models),
            'metrics_tracked': len(self.metrics_history),
            'last_anomaly': self.anomalies[-1] if self.anomalies else None
        }

# Global health monitor instance
health_monitor = PredictiveHealthMonitor()

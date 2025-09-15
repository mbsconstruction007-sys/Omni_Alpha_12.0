"""
Risk Database Module
Database operations for risk management data
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, asdict
import asyncpg
import pandas as pd

logger = structlog.get_logger()

@dataclass
class RiskRecord:
    """Risk data record"""
    id: str
    timestamp: datetime
    portfolio_value: float
    total_risk: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    tail_risk: float
    skewness: float
    kurtosis: float
    metadata: Dict

class RiskDatabase:
    """Risk database operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pool = None
        # Database will be initialized when first accessed
    
    async def initialize_database(self):
        """Initialize database connection"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config["DATABASE_URL"],
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("Risk database initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize risk database", error=str(e))
            raise
    
    async def _ensure_database_initialized(self):
        """Ensure database is initialized"""
        if self.pool is None:
            await self.initialize_database()
    
    async def _create_tables(self):
        """Create risk management tables"""
        async with self.pool.acquire() as conn:
            # Risk metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id VARCHAR(50) PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    portfolio_value DECIMAL(20, 4) NOT NULL,
                    total_risk DECIMAL(10, 4) NOT NULL,
                    var_95 DECIMAL(10, 4) NOT NULL,
                    var_99 DECIMAL(10, 4) NOT NULL,
                    expected_shortfall DECIMAL(10, 4) NOT NULL,
                    sharpe_ratio DECIMAL(10, 4) NOT NULL,
                    sortino_ratio DECIMAL(10, 4) NOT NULL,
                    max_drawdown DECIMAL(10, 4) NOT NULL,
                    current_drawdown DECIMAL(10, 4) NOT NULL,
                    volatility DECIMAL(10, 4) NOT NULL,
                    beta DECIMAL(10, 4) NOT NULL,
                    alpha DECIMAL(10, 4) NOT NULL,
                    information_ratio DECIMAL(10, 4) NOT NULL,
                    correlation_risk DECIMAL(10, 4) NOT NULL,
                    concentration_risk DECIMAL(10, 4) NOT NULL,
                    liquidity_risk DECIMAL(10, 4) NOT NULL,
                    tail_risk DECIMAL(10, 4) NOT NULL,
                    skewness DECIMAL(10, 4) NOT NULL,
                    kurtosis DECIMAL(10, 4) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Risk alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id VARCHAR(50) PRIMARY KEY,
                    rule_name VARCHAR(100) NOT NULL,
                    level VARCHAR(20) NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    data JSONB,
                    channels TEXT[] NOT NULL,
                    sent BOOLEAN DEFAULT FALSE,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    escalated BOOLEAN DEFAULT FALSE,
                    acknowledged_by VARCHAR(100),
                    acknowledged_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Circuit breaker events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                    id SERIAL PRIMARY KEY,
                    breaker_name VARCHAR(100) NOT NULL,
                    event_type VARCHAR(20) NOT NULL,
                    threshold_value DECIMAL(10, 4) NOT NULL,
                    actual_value DECIMAL(10, 4) NOT NULL,
                    reason TEXT,
                    actions_taken TEXT[],
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Stress test results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    id VARCHAR(50) PRIMARY KEY,
                    test_name VARCHAR(100) NOT NULL,
                    scenario_name VARCHAR(100) NOT NULL,
                    portfolio_loss DECIMAL(15, 4) NOT NULL,
                    loss_percentage DECIMAL(10, 4) NOT NULL,
                    worst_position VARCHAR(20),
                    worst_position_loss DECIMAL(15, 4),
                    correlation_impact DECIMAL(10, 4),
                    liquidity_impact DECIMAL(10, 4),
                    recovery_time_estimate INTEGER,
                    confidence_level DECIMAL(5, 4),
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # VaR calculations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS var_calculations (
                    id VARCHAR(50) PRIMARY KEY,
                    confidence_level DECIMAL(5, 4) NOT NULL,
                    time_horizon INTEGER NOT NULL,
                    method VARCHAR(50) NOT NULL,
                    var_value DECIMAL(10, 4) NOT NULL,
                    expected_shortfall DECIMAL(10, 4) NOT NULL,
                    historical_var DECIMAL(10, 4),
                    parametric_var DECIMAL(10, 4),
                    monte_carlo_var DECIMAL(10, 4),
                    confidence_interval_lower DECIMAL(10, 4),
                    confidence_interval_upper DECIMAL(10, 4),
                    backtest_results JSONB,
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_alerts_timestamp ON risk_alerts(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_alerts_level ON risk_alerts(level)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_timestamp ON circuit_breaker_events(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_stress_test_results_timestamp ON stress_test_results(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_var_calculations_timestamp ON var_calculations(timestamp)")
            
            logger.info("Risk database tables created successfully")
    
    async def store_risk_metrics(self, risk_record: RiskRecord) -> bool:
        """Store risk metrics in database"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO risk_metrics (
                        id, timestamp, portfolio_value, total_risk, var_95, var_99,
                        expected_shortfall, sharpe_ratio, sortino_ratio, max_drawdown,
                        current_drawdown, volatility, beta, alpha, information_ratio,
                        correlation_risk, concentration_risk, liquidity_risk, tail_risk,
                        skewness, kurtosis, metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22
                    )
                """, 
                risk_record.id,
                risk_record.timestamp,
                risk_record.portfolio_value,
                risk_record.total_risk,
                risk_record.var_95,
                risk_record.var_99,
                risk_record.expected_shortfall,
                risk_record.sharpe_ratio,
                risk_record.sortino_ratio,
                risk_record.max_drawdown,
                risk_record.current_drawdown,
                risk_record.volatility,
                risk_record.beta,
                risk_record.alpha,
                risk_record.information_ratio,
                risk_record.correlation_risk,
                risk_record.concentration_risk,
                risk_record.liquidity_risk,
                risk_record.tail_risk,
                risk_record.skewness,
                risk_record.kurtosis,
                json.dumps(risk_record.metadata)
                )
            
            logger.info("Risk metrics stored", id=risk_record.id)
            return True
            
        except Exception as e:
            logger.error("Failed to store risk metrics", error=str(e))
            return False
    
    async def get_risk_metrics_history(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 1000
    ) -> List[RiskRecord]:
        """Get risk metrics history"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM risk_metrics 
                    WHERE timestamp BETWEEN $1 AND $2 
                    ORDER BY timestamp DESC 
                    LIMIT $3
                """, start_date, end_date, limit)
                
                records = []
                for row in rows:
                    record = RiskRecord(
                        id=row['id'],
                        timestamp=row['timestamp'],
                        portfolio_value=float(row['portfolio_value']),
                        total_risk=float(row['total_risk']),
                        var_95=float(row['var_95']),
                        var_99=float(row['var_99']),
                        expected_shortfall=float(row['expected_shortfall']),
                        sharpe_ratio=float(row['sharpe_ratio']),
                        sortino_ratio=float(row['sortino_ratio']),
                        max_drawdown=float(row['max_drawdown']),
                        current_drawdown=float(row['current_drawdown']),
                        volatility=float(row['volatility']),
                        beta=float(row['beta']),
                        alpha=float(row['alpha']),
                        information_ratio=float(row['information_ratio']),
                        correlation_risk=float(row['correlation_risk']),
                        concentration_risk=float(row['concentration_risk']),
                        liquidity_risk=float(row['liquidity_risk']),
                        tail_risk=float(row['tail_risk']),
                        skewness=float(row['skewness']),
                        kurtosis=float(row['kurtosis']),
                        metadata=row['metadata'] or {}
                    )
                    records.append(record)
                
                logger.info("Risk metrics history retrieved", count=len(records))
                return records
                
        except Exception as e:
            logger.error("Failed to get risk metrics history", error=str(e))
            return []
    
    async def store_risk_alert(self, alert_data: Dict) -> bool:
        """Store risk alert in database"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO risk_alerts (
                        id, rule_name, level, title, message, timestamp, data, channels,
                        sent, acknowledged, escalated
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    )
                """,
                alert_data['id'],
                alert_data['rule_name'],
                alert_data['level'],
                alert_data['title'],
                alert_data['message'],
                alert_data['timestamp'],
                json.dumps(alert_data.get('data', {})),
                alert_data['channels'],
                alert_data.get('sent', False),
                alert_data.get('acknowledged', False),
                alert_data.get('escalated', False)
                )
            
            logger.info("Risk alert stored", id=alert_data['id'])
            return True
            
        except Exception as e:
            logger.error("Failed to store risk alert", error=str(e))
            return False
    
    async def get_risk_alerts(
        self, 
        start_date: datetime, 
        end_date: datetime,
        level: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get risk alerts from database"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM risk_alerts 
                    WHERE timestamp BETWEEN $1 AND $2
                """
                params = [start_date, end_date]
                
                if level:
                    query += " AND level = $3"
                    params.append(level)
                
                query += " ORDER BY timestamp DESC LIMIT $4"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                alerts = []
                for row in rows:
                    alert = {
                        'id': row['id'],
                        'rule_name': row['rule_name'],
                        'level': row['level'],
                        'title': row['title'],
                        'message': row['message'],
                        'timestamp': row['timestamp'].isoformat(),
                        'data': row['data'] or {},
                        'channels': row['channels'],
                        'sent': row['sent'],
                        'acknowledged': row['acknowledged'],
                        'escalated': row['escalated'],
                        'acknowledged_by': row['acknowledged_by'],
                        'acknowledged_at': row['acknowledged_at'].isoformat() if row['acknowledged_at'] else None
                    }
                    alerts.append(alert)
                
                logger.info("Risk alerts retrieved", count=len(alerts))
                return alerts
                
        except Exception as e:
            logger.error("Failed to get risk alerts", error=str(e))
            return []
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge a risk alert"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE risk_alerts 
                    SET acknowledged = TRUE, acknowledged_by = $1, acknowledged_at = NOW()
                    WHERE id = $2
                """, user, alert_id)
                
                if result == "UPDATE 1":
                    logger.info("Alert acknowledged", alert_id=alert_id, user=user)
                    return True
                else:
                    logger.warning("Alert not found", alert_id=alert_id)
                    return False
                    
        except Exception as e:
            logger.error("Failed to acknowledge alert", error=str(e))
            return False
    
    async def store_circuit_breaker_event(self, event_data: Dict) -> bool:
        """Store circuit breaker event"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO circuit_breaker_events (
                        breaker_name, event_type, threshold_value, actual_value,
                        reason, actions_taken, timestamp
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7
                    )
                """,
                event_data['breaker_name'],
                event_data['event_type'],
                event_data['threshold_value'],
                event_data['actual_value'],
                event_data.get('reason', ''),
                event_data.get('actions_taken', []),
                event_data['timestamp']
                )
            
            logger.info("Circuit breaker event stored", breaker=event_data['breaker_name'])
            return True
            
        except Exception as e:
            logger.error("Failed to store circuit breaker event", error=str(e))
            return False
    
    async def get_circuit_breaker_events(
        self, 
        start_date: datetime, 
        end_date: datetime,
        breaker_name: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get circuit breaker events"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM circuit_breaker_events 
                    WHERE timestamp BETWEEN $1 AND $2
                """
                params = [start_date, end_date]
                
                if breaker_name:
                    query += " AND breaker_name = $3"
                    params.append(breaker_name)
                
                query += " ORDER BY timestamp DESC LIMIT $4"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                events = []
                for row in rows:
                    event = {
                        'id': row['id'],
                        'breaker_name': row['breaker_name'],
                        'event_type': row['event_type'],
                        'threshold_value': float(row['threshold_value']),
                        'actual_value': float(row['actual_value']),
                        'reason': row['reason'],
                        'actions_taken': row['actions_taken'],
                        'timestamp': row['timestamp'].isoformat()
                    }
                    events.append(event)
                
                logger.info("Circuit breaker events retrieved", count=len(events))
                return events
                
        except Exception as e:
            logger.error("Failed to get circuit breaker events", error=str(e))
            return []
    
    async def store_stress_test_result(self, result_data: Dict) -> bool:
        """Store stress test result"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO stress_test_results (
                        id, test_name, scenario_name, portfolio_loss, loss_percentage,
                        worst_position, worst_position_loss, correlation_impact,
                        liquidity_impact, recovery_time_estimate, confidence_level, timestamp
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                    )
                """,
                result_data['id'],
                result_data['test_name'],
                result_data['scenario_name'],
                result_data['portfolio_loss'],
                result_data['loss_percentage'],
                result_data.get('worst_position'),
                result_data.get('worst_position_loss'),
                result_data.get('correlation_impact'),
                result_data.get('liquidity_impact'),
                result_data.get('recovery_time_estimate'),
                result_data.get('confidence_level'),
                result_data['timestamp']
                )
            
            logger.info("Stress test result stored", id=result_data['id'])
            return True
            
        except Exception as e:
            logger.error("Failed to store stress test result", error=str(e))
            return False
    
    async def store_var_calculation(self, var_data: Dict) -> bool:
        """Store VaR calculation result"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO var_calculations (
                        id, confidence_level, time_horizon, method, var_value,
                        expected_shortfall, historical_var, parametric_var,
                        monte_carlo_var, confidence_interval_lower,
                        confidence_interval_upper, backtest_results, timestamp
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    )
                """,
                var_data['id'],
                var_data['confidence_level'],
                var_data['time_horizon'],
                var_data['method'],
                var_data['var_value'],
                var_data['expected_shortfall'],
                var_data.get('historical_var'),
                var_data.get('parametric_var'),
                var_data.get('monte_carlo_var'),
                var_data.get('confidence_interval_lower'),
                var_data.get('confidence_interval_upper'),
                json.dumps(var_data.get('backtest_results', {})),
                var_data['timestamp']
                )
            
            logger.info("VaR calculation stored", id=var_data['id'])
            return True
            
        except Exception as e:
            logger.error("Failed to store VaR calculation", error=str(e))
            return False
    
    async def get_risk_dashboard_data(self) -> Dict:
        """Get data for risk dashboard"""
        try:
            async with self.pool.acquire() as conn:
                # Get latest risk metrics
                latest_metrics = await conn.fetchrow("""
                    SELECT * FROM risk_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                # Get recent alerts
                recent_alerts = await conn.fetch("""
                    SELECT * FROM risk_alerts 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """)
                
                # Get circuit breaker status
                circuit_events = await conn.fetch("""
                    SELECT * FROM circuit_breaker_events 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                # Get risk metrics summary
                risk_summary = await conn.fetchrow("""
                    SELECT 
                        AVG(total_risk) as avg_risk,
                        MAX(total_risk) as max_risk,
                        AVG(var_95) as avg_var,
                        MAX(current_drawdown) as max_drawdown,
                        AVG(sharpe_ratio) as avg_sharpe
                    FROM risk_metrics 
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                """)
                
                dashboard_data = {
                    'latest_metrics': dict(latest_metrics) if latest_metrics else {},
                    'recent_alerts': [dict(alert) for alert in recent_alerts],
                    'circuit_events': [dict(event) for event in circuit_events],
                    'risk_summary': dict(risk_summary) if risk_summary else {},
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                logger.info("Risk dashboard data retrieved")
                return dashboard_data
                
        except Exception as e:
            logger.error("Failed to get risk dashboard data", error=str(e))
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Cleanup old risk data"""
        try:
            async with self.pool.acquire() as conn:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                # Cleanup old risk metrics
                metrics_deleted = await conn.execute("""
                    DELETE FROM risk_metrics 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                # Cleanup old alerts
                alerts_deleted = await conn.execute("""
                    DELETE FROM risk_alerts 
                    WHERE timestamp < $1 AND acknowledged = TRUE
                """, cutoff_date)
                
                # Cleanup old circuit breaker events
                events_deleted = await conn.execute("""
                    DELETE FROM circuit_breaker_events 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                # Cleanup old stress test results
                stress_deleted = await conn.execute("""
                    DELETE FROM stress_test_results 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                # Cleanup old VaR calculations
                var_deleted = await conn.execute("""
                    DELETE FROM var_calculations 
                    WHERE timestamp < $1
                """, cutoff_date)
                
                logger.info("Old risk data cleaned up", 
                           metrics_deleted=metrics_deleted,
                           alerts_deleted=alerts_deleted,
                           events_deleted=events_deleted,
                           stress_deleted=stress_deleted,
                           var_deleted=var_deleted)
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
            logger.info("Risk database connection closed")

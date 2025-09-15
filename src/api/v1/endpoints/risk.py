"""
Risk Management API Endpoints
Real-time access to all risk metrics and controls
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import structlog
import asyncio

from src.risk_management.risk_engine import RiskEngine, RiskLevel
from src.risk_management.risk_metrics import RiskMetrics
from src.risk_management.var_calculator import VaRCalculator
from src.risk_management.stress_testing import StressTester
from src.risk_management.circuit_breaker import CircuitBreaker
from src.risk_management.risk_alerts import RiskAlerts
from src.risk_management.risk_database import RiskDatabase

logger = structlog.get_logger()
router = APIRouter(prefix="/risk", tags=["risk"])

# Global risk management components (in production, these would be dependency injected)
risk_engine = None
risk_metrics = None
var_calculator = None
stress_tester = None
circuit_breaker = None
risk_alerts = None
risk_database = None

# Request/Response Models

class RiskCheckRequest(BaseModel):
    """Request model for pre-trade risk check"""
    symbol: str = Field(..., description="Trading symbol")
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: float = Field(..., gt=0, description="Order price")
    side: str = Field(..., description="Order side: buy or sell")
    order_type: str = Field(..., description="Order type: market, limit, etc.")
    client_order_id: Optional[str] = Field(None, description="Client order ID")

class RiskCheckResponse(BaseModel):
    """Response model for risk check"""
    approved: bool
    risk_score: float
    warnings: List[str]
    rejections: List[str]
    recommendations: List[str]
    timestamp: str

class RiskMetricsResponse(BaseModel):
    """Response model for risk metrics"""
    portfolio_value: float
    total_risk: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
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
    timestamp: str

class PositionSizeRequest(BaseModel):
    """Request model for position sizing"""
    symbol: str = Field(..., description="Trading symbol")
    account_value: float = Field(..., gt=0, description="Account value")
    risk_per_trade: Optional[float] = Field(None, description="Risk per trade percentage")
    method: Optional[str] = Field("kelly_criterion", description="Position sizing method")

class PositionSizeResponse(BaseModel):
    """Response model for position sizing"""
    symbol: str
    recommended_shares: int
    position_value: float
    risk_amount: float
    risk_percentage: float
    method_used: str
    confidence_score: float
    warnings: List[str]

class VaRRequest(BaseModel):
    """Request model for VaR calculation"""
    confidence_level: float = Field(0.95, ge=0.90, le=0.999, description="Confidence level")
    time_horizon: int = Field(1, ge=1, le=30, description="Time horizon in days")
    method: Optional[str] = Field("all", description="VaR calculation method")

class VaRResponse(BaseModel):
    """Response model for VaR calculation"""
    var_value: float
    confidence_level: float
    time_horizon: int
    method: str
    expected_shortfall: float
    historical_var: float
    parametric_var: float
    monte_carlo_var: float
    confidence_interval: List[float]
    interpretation: str
    timestamp: str

class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    scenarios: Optional[List[str]] = Field(None, description="Specific scenarios to test")
    custom_scenarios: Optional[Dict[str, Dict]] = Field(None, description="Custom scenarios")

class StressTestResponse(BaseModel):
    """Response model for stress testing"""
    scenarios: Dict[str, float]
    worst_case_loss: float
    best_case_loss: float
    recommendation: str
    timestamp: str

class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status"""
    name: str
    type: str
    state: str
    enabled: bool
    threshold: float
    current_value: float
    threshold_breached: bool
    escalation_level: int
    trigger_count: int
    last_trigger: Optional[str]

class AlertResponse(BaseModel):
    """Alert response"""
    id: str
    rule_name: str
    level: str
    title: str
    message: str
    timestamp: str
    acknowledged: bool
    escalated: bool

# API Endpoints

@router.post("/check-pre-trade", response_model=RiskCheckResponse)
async def check_pre_trade_risk(request: RiskCheckRequest):
    """
    Pre-trade risk check endpoint
    Returns approval status and detailed risk report
    """
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not initialized")
        
        order = {
            "id": f"ORDER_{datetime.utcnow().timestamp()}",
            "symbol": request.symbol,
            "quantity": request.quantity,
            "price": request.price,
            "side": request.side,
            "order_type": request.order_type,
            "client_order_id": request.client_order_id
        }
        
        approved, risk_report = await risk_engine.check_pre_trade_risk(order)
        
        return RiskCheckResponse(
            approved=approved,
            risk_score=risk_report.risk_score,
            warnings=risk_report.warnings,
            rejections=risk_report.rejections,
            recommendations=risk_report.recommendations,
            timestamp=risk_report.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error("Pre-trade risk check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics():
    """
    Get current risk metrics for the portfolio
    """
    try:
        if not risk_metrics:
            raise HTTPException(status_code=503, detail="Risk metrics not initialized")
        
        metrics = await risk_metrics.calculate_comprehensive_metrics()
        
        return RiskMetricsResponse(
            portfolio_value=metrics.portfolio_value,
            total_risk=metrics.total_risk,
            var_95=metrics.var_95,
            var_99=metrics.var_99,
            expected_shortfall=metrics.expected_shortfall,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            max_drawdown=metrics.max_drawdown,
            current_drawdown=metrics.current_drawdown,
            volatility=metrics.volatility,
            beta=metrics.beta,
            alpha=metrics.alpha,
            information_ratio=metrics.information_ratio,
            correlation_risk=metrics.correlation_risk,
            concentration_risk=metrics.concentration_risk,
            liquidity_risk=metrics.liquidity_risk,
            tail_risk=metrics.tail_risk,
            skewness=metrics.skewness,
            kurtosis=metrics.kurtosis,
            timestamp=metrics.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get risk metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/position-size", response_model=PositionSizeResponse)
async def calculate_position_size(request: PositionSizeRequest):
    """
    Calculate optimal position size based on risk parameters
    """
    try:
        if not risk_engine:
            raise HTTPException(status_code=503, detail="Risk engine not initialized")
        
        result = await risk_engine.position_risk.calculate_optimal_position_size(
            symbol=request.symbol,
            account_value=request.account_value,
            entry_price=100.0,  # Would get from market data
            risk_per_trade=request.risk_per_trade,
            method=request.method
        )
        
        return PositionSizeResponse(
            symbol=request.symbol,
            recommended_shares=result.recommended_shares,
            position_value=result.position_value,
            risk_amount=result.risk_amount,
            risk_percentage=result.risk_percentage,
            method_used=result.method_used,
            confidence_score=result.confidence_score,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error("Position sizing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/var", response_model=VaRResponse)
async def calculate_value_at_risk(request: VaRRequest):
    """
    Calculate Value at Risk
    """
    try:
        if not var_calculator:
            raise HTTPException(status_code=503, detail="VaR calculator not initialized")
        
        var_result = await var_calculator.calculate_comprehensive_var(
            confidence_level=request.confidence_level,
            time_horizon=request.time_horizon,
            method=request.method
        )
        
        interpretation = f"There is a {request.confidence_level*100}% probability that the portfolio will not lose more than {var_result.var_value:.2f}% in {request.time_horizon} day(s)"
        
        return VaRResponse(
            var_value=var_result.var_value,
            confidence_level=var_result.confidence_level,
            time_horizon=var_result.time_horizon,
            method=var_result.method,
            expected_shortfall=var_result.expected_shortfall,
            historical_var=var_result.historical_var,
            parametric_var=var_result.parametric_var,
            monte_carlo_var=var_result.monte_carlo_var,
            confidence_interval=list(var_result.confidence_interval),
            interpretation=interpretation,
            timestamp=var_result.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error("VaR calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(request: StressTestRequest):
    """
    Run stress test scenarios
    """
    try:
        if not stress_tester:
            raise HTTPException(status_code=503, detail="Stress tester not initialized")
        
        if request.scenarios:
            # Run specific scenarios
            results = {}
            for scenario in request.scenarios:
                scenario_result = await stress_tester._run_scenario(
                    scenario, 
                    stress_tester.scenarios.get(scenario, {})
                )
                results[scenario] = scenario_result.loss_percentage
        else:
            # Run all scenarios
            results = await stress_tester.run_all_scenarios()
        
        worst_case = max(results.values()) if results else 0
        best_case = min(results.values()) if results else 0
        
        recommendation = "Reduce position sizes" if worst_case > 30 else "Risk levels acceptable"
        
        return StressTestResponse(
            scenarios=results,
            worst_case_loss=worst_case,
            best_case_loss=best_case,
            recommendation=recommendation,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Stress test failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """
    Get circuit breaker status
    """
    try:
        if not circuit_breaker:
            raise HTTPException(status_code=503, detail="Circuit breaker not initialized")
        
        status = await circuit_breaker.get_breaker_status()
        
        return {
            "breakers": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get circuit breaker status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(
    breaker_name: str = Query(..., description="Circuit breaker name"),
    confirmation: str = Query(..., description="Type 'CONFIRM' to reset")
):
    """
    Reset circuit breaker (requires confirmation)
    """
    try:
        if confirmation != "CONFIRM":
            raise HTTPException(status_code=400, detail="Confirmation required")
        
        if not circuit_breaker:
            raise HTTPException(status_code=503, detail="Circuit breaker not initialized")
        
        success = await circuit_breaker.reset_breaker(breaker_name, force=True)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset circuit breaker")
        
        return {
            "status": "reset",
            "breaker_name": breaker_name,
            "message": "Circuit breaker has been reset",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Circuit breaker reset failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_risk_alerts(
    limit: int = Query(10, ge=1, le=100, description="Number of alerts to return"),
    level: Optional[str] = Query(None, description="Filter by alert level")
):
    """
    Get recent risk alerts
    """
    try:
        if not risk_alerts:
            raise HTTPException(status_code=503, detail="Risk alerts not initialized")
        
        alerts = await risk_alerts.get_alert_history(limit=limit)
        
        if level:
            alerts = [alert for alert in alerts if alert["level"] == level]
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "critical_count": sum(1 for a in alerts if a.get("level") == "critical"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get risk alerts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user: str = Query("system", description="User acknowledging the alert")
):
    """
    Acknowledge a risk alert
    """
    try:
        if not risk_alerts:
            raise HTTPException(status_code=503, detail="Risk alerts not initialized")
        
        success = await risk_alerts.acknowledge_alert(alert_id, user)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "acknowledged_by": user,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to acknowledge alert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_risk_dashboard():
    """
    Get comprehensive risk dashboard data
    """
    try:
        # Get current risk metrics
        if risk_metrics:
            current_metrics = await risk_metrics.calculate_comprehensive_metrics()
        else:
            current_metrics = None
        
        # Get circuit breaker status
        if circuit_breaker:
            breaker_status = await circuit_breaker.get_breaker_status()
        else:
            breaker_status = {}
        
        # Get recent alerts
        if risk_alerts:
            recent_alerts = await risk_alerts.get_active_alerts()
            alert_stats = await risk_alerts.get_alert_statistics()
        else:
            recent_alerts = []
            alert_stats = {}
        
        # Get dashboard data from database
        if risk_database:
            dashboard_data = await risk_database.get_risk_dashboard_data()
        else:
            dashboard_data = {}
        
        return {
            "overview": {
                "current_metrics": current_metrics.__dict__ if current_metrics else {},
                "circuit_breakers": breaker_status,
                "active_alerts": len(recent_alerts),
                "alert_statistics": alert_stats
            },
            "recent_alerts": recent_alerts,
            "dashboard_data": dashboard_data,
            "recommendations": await _generate_risk_recommendations(current_metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get risk dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def risk_system_health():
    """
    Get risk system health status
    """
    try:
        health_status = {
            "risk_engine": risk_engine is not None,
            "risk_metrics": risk_metrics is not None,
            "var_calculator": var_calculator is not None,
            "stress_tester": stress_tester is not None,
            "circuit_breaker": circuit_breaker is not None,
            "risk_alerts": risk_alerts is not None,
            "risk_database": risk_database is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        all_healthy = all(health_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get risk system health", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions

async def _generate_risk_recommendations(current_metrics) -> List[str]:
    """Generate risk recommendations based on current state"""
    recommendations = []
    
    if not current_metrics:
        return ["Risk metrics not available"]
    
    if current_metrics.current_drawdown > 10:
        recommendations.append("Consider reducing position sizes due to high drawdown")
    
    if current_metrics.var_95 > 15:
        recommendations.append("Portfolio VaR is high - review risk exposure")
    
    if current_metrics.correlation_risk > 0.7:
        recommendations.append("High correlation detected - consider diversification")
    
    if current_metrics.liquidity_risk > 50:
        recommendations.append("Liquidity risk is elevated - review position sizes")
    
    if current_metrics.sharpe_ratio < 1.0:
        recommendations.append("Sharpe ratio is low - consider risk-adjusted returns")
    
    if not recommendations:
        recommendations.append("Risk levels are within acceptable parameters")
    
    return recommendations

# Initialize risk management components
def init_risk_management(config: Dict):
    """Initialize risk management system"""
    global risk_engine, risk_metrics, var_calculator, stress_tester, circuit_breaker, risk_alerts, risk_database
    
    try:
        # Initialize components
        risk_engine = RiskEngine(config)
        risk_metrics = RiskMetrics(config)
        var_calculator = VaRCalculator(config)
        stress_tester = StressTester(config)
        circuit_breaker = CircuitBreaker(config)
        risk_alerts = RiskAlerts(config)
        risk_database = RiskDatabase(config)
        
        logger.info("Risk management system initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize risk management system", error=str(e))
        raise

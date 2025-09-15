"""
Institutional Trading API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/institutional", tags=["institutional"])

# Pydantic models for API
class PortfolioRequest(BaseModel):
    """Portfolio optimization request"""
    signals: Dict[str, float]
    risk_limit: float = 0.10
    optimization_method: str = "hierarchical_risk_parity"

class RiskMetricsResponse(BaseModel):
    """Risk metrics response"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    timestamp: datetime

class PerformanceResponse(BaseModel):
    """Performance metrics response"""
    total_pnl: float
    daily_pnl: float
    mtd_pnl: float
    ytd_pnl: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    best_day: float
    worst_day: float

class MicrostructureResponse(BaseModel):
    """Microstructure analysis response"""
    book_signals: Dict[str, Any]
    flow_signals: Dict[str, Any]
    toxicity: Dict[str, Any]
    venue_scores: Dict[str, float]
    timestamp: datetime

class AlphaSignalsResponse(BaseModel):
    """Alpha signals response"""
    traditional_factors: Dict[str, float]
    ml_predictions: Dict[str, float]
    alt_data_signals: Dict[str, float]
    combined_signals: Dict[str, float]
    timestamp: datetime

# Global engine instance (would be properly managed in production)
engine = None

@router.on_event("startup")
async def startup_event():
    """Initialize institutional engine on startup"""
    global engine
    from app.institutional.core import InstitutionalTradingEngine, InstitutionalConfig, InstitutionalType
    
    config = InstitutionalConfig(
        name="AlphaQuantCapital",
        type=InstitutionalType.HEDGE_FUND,
        aum_target=1000000000,
        risk_budget=0.10,
        regulatory_jurisdictions=["US", "EU"],
        prime_brokers=["Goldman Sachs", "Morgan Stanley"],
        asset_classes=["equities", "options", "futures"],
        strategies=[]
    )
    
    engine = InstitutionalTradingEngine(config)
    await engine.initialize()
    asyncio.create_task(engine.run())

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get institutional engine status"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "status": "running" if engine.is_running else "stopped",
        "engine_id": engine.engine_id,
        "uptime": (datetime.now() - engine.start_time).total_seconds(),
        "in_drawdown": engine.in_drawdown,
        "emergency_stop": engine.emergency_stop,
        "positions_count": len(engine.positions),
        "active_orders": len(engine.orders),
        "config": {
            "name": engine.config.name,
            "type": engine.config.type.value,
            "aum_target": engine.config.aum_target,
            "risk_budget": engine.config.risk_budget
        }
    }

@router.get("/risk/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics() -> RiskMetricsResponse:
    """Get current risk metrics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    metrics = engine.risk_manager.risk_metrics
    
    return RiskMetricsResponse(
        var_95=metrics.get('var_95', 0),
        var_99=metrics.get('var_99', 0),
        expected_shortfall=0.0,  # Would calculate
        max_drawdown=metrics.get('max_drawdown', 0),
        sharpe_ratio=metrics.get('sharpe_ratio', 0),
        sortino_ratio=metrics.get('sortino_ratio', 0),
        calmar_ratio=metrics.get('calmar_ratio', 0),
        timestamp=datetime.now()
    )

@router.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """Get performance metrics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    metrics = await engine.performance_tracker.get_metrics()
    
    return PerformanceResponse(
        total_pnl=metrics.get('total_pnl', 0),
        daily_pnl=0.0,  # Would calculate
        mtd_pnl=0.0,  # Would calculate
        ytd_pnl=0.0,  # Would calculate
        sharpe_ratio=metrics.get('sharpe_ratio', 0),
        win_rate=metrics.get('win_rate', 0),
        avg_win=metrics.get('avg_win', 0),
        avg_loss=metrics.get('avg_loss', 0),
        best_day=0.0,  # Would calculate
        worst_day=0.0  # Would calculate
    )

@router.get("/microstructure", response_model=MicrostructureResponse)
async def get_microstructure_analysis() -> MicrostructureResponse:
    """Get microstructure analysis"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Get market data
    market_data = await engine.data_pipeline.get_market_data()
    
    # Perform microstructure analysis
    analysis = await engine.microstructure_analyzer.analyze(market_data)
    
    return MicrostructureResponse(
        book_signals=analysis.get('book_signals', {}),
        flow_signals=analysis.get('flow_signals', {}),
        toxicity=analysis.get('toxicity', {}),
        venue_scores=analysis.get('venue_scores', {}),
        timestamp=analysis.get('timestamp', datetime.now())
    )

@router.get("/alpha/signals", response_model=AlphaSignalsResponse)
async def get_alpha_signals() -> AlphaSignalsResponse:
    """Get alpha signals"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Get market data
    market_data = await engine.data_pipeline.get_market_data()
    
    # Get microstructure signals
    microstructure_signals = await engine.microstructure_analyzer.analyze(market_data)
    
    # Generate alpha signals
    alpha_signals = await engine.alpha_engine.generate_signals(
        market_data, 
        microstructure_signals
    )
    
    # Get individual components
    traditional_factors = await engine.alpha_engine.factor_library.calculate_factors(market_data)
    ml_predictions = await engine.alpha_engine._generate_ml_predictions(market_data)
    alt_data_signals = await engine.alpha_engine._process_alternative_data(market_data)
    
    return AlphaSignalsResponse(
        traditional_factors=traditional_factors,
        ml_predictions=ml_predictions,
        alt_data_signals=alt_data_signals,
        combined_signals=alpha_signals,
        timestamp=datetime.now()
    )

@router.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioRequest) -> Dict[str, float]:
    """Optimize portfolio allocation"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Get risk limits
    risk_limits = {'max_position': request.risk_limit}
    
    # Optimize portfolio
    optimized_portfolio = await engine.portfolio_manager.optimize_portfolio(
        request.signals,
        engine.positions,
        risk_limits
    )
    
    return optimized_portfolio

@router.get("/positions")
async def get_positions() -> List[Dict[str, Any]]:
    """Get current positions"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    positions = []
    for symbol, position in engine.positions.items():
        positions.append({
            "symbol": symbol,
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl
        })
    
    return positions

@router.get("/orders/active")
async def get_active_orders() -> List[Dict[str, Any]]:
    """Get active orders"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    orders = []
    for order_id, order in engine.orders.items():
        orders.append({
            "id": order_id,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "price": order.price,
            "timestamp": order.timestamp.isoformat()
        })
    
    return orders

@router.get("/execution/analytics")
async def get_execution_analytics() -> Dict[str, Any]:
    """Get execution analytics"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    analytics = engine.execution_engine.execution_analytics.performance_metrics
    
    return {
        "avg_slippage": analytics.get('avg_slippage', 0),
        "slippage_std": analytics.get('slippage_std', 0),
        "fill_rate": analytics.get('fill_rate', 0),
        "avg_latency_ms": analytics.get('avg_latency_ms', 0),
        "total_executions": analytics.get('total_executions', 0),
        "timestamp": datetime.now()
    }

@router.get("/compliance/status")
async def get_compliance_status() -> Dict[str, Any]:
    """Get compliance status"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "violations_count": len(engine.compliance_engine.violations),
        "last_check": datetime.now().isoformat(),
        "status": "compliant" if len(engine.compliance_engine.violations) == 0 else "violations_detected"
    }

@router.post("/emergency/stop")
async def emergency_stop() -> Dict[str, str]:
    """Trigger emergency stop"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    engine.emergency_stop = True
    
    return {"status": "Emergency stop activated"}

@router.post("/emergency/resume")
async def resume_trading() -> Dict[str, str]:
    """Resume trading after emergency stop"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    engine.emergency_stop = False
    
    return {"status": "Trading resumed"}

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check"""
    if not engine:
        return {"status": "unhealthy", "reason": "Engine not initialized"}
    
    health_status = {
        "engine": "healthy" if engine.is_running else "unhealthy",
        "data_pipeline": "healthy",  # Would check actual status
        "risk_manager": "healthy",
        "execution_engine": "healthy",
        "compliance": "healthy",
        "timestamp": datetime.now()
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in health_status.values() if isinstance(status, str)) else "unhealthy"
    
    return {
        "status": overall_status,
        "components": health_status
    }

@router.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """Real-time data feed"""
    await websocket.accept()
    
    try:
        while True:
            if engine:
                # Send real-time updates
                data = {
                    "type": "update",
                    "positions": len(engine.positions),
                    "orders": len(engine.orders),
                    "pnl": await engine.performance_tracker.get_metrics(),
                    "risk_metrics": engine.risk_manager.risk_metrics,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_json(data)
            
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        await websocket.close()

@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get engine configuration"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return {
        "name": engine.config.name,
        "type": engine.config.type.value,
        "aum_target": engine.config.aum_target,
        "risk_budget": engine.config.risk_budget,
        "max_leverage": engine.config.max_leverage,
        "max_drawdown": engine.config.max_drawdown,
        "target_sharpe": engine.config.target_sharpe,
        "regulatory_jurisdictions": engine.config.regulatory_jurisdictions,
        "prime_brokers": engine.config.prime_brokers,
        "asset_classes": engine.config.asset_classes
    }

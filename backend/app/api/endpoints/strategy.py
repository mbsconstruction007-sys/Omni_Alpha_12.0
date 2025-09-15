"""
Strategy API Endpoints - FastAPI Router for Strategy Management
Step 8: World's #1 Strategy Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ...strategy_engine.strategy_engine import StrategyEngine
from ...strategy_engine.core.strategy_config import StrategyConfig
from ...strategy_engine.models.strategy_models import (
    Strategy, StrategyType, StrategyStatus, StrategyPerformance,
    Signal, SignalType, SignalSource, TradingSignal,
    StrategyDiscovery, StrategyEvolution, BacktestResult
)
from ...strategy_engine.signal_aggregator import FusionMethod

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/strategy", tags=["Strategy Management"])

# Global strategy engine instance
strategy_engine: Optional[StrategyEngine] = None

def get_strategy_engine() -> StrategyEngine:
    """Get strategy engine instance"""
    if strategy_engine is None:
        raise HTTPException(status_code=503, detail="Strategy engine not initialized")
    return strategy_engine

def init_strategy_engine(config: StrategyConfig):
    """Initialize strategy engine"""
    global strategy_engine
    strategy_engine = StrategyEngine(config)
    logger.info("🚀 Strategy Engine initialized")

# Strategy Management Endpoints

@router.post("/create", response_model=Strategy)
async def create_strategy(
    strategy_data: Dict[str, Any],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Create a new strategy"""
    try:
        strategy = await engine.create_strategy(strategy_data)
        return strategy
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/start")
async def start_strategy(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Start a strategy"""
    try:
        success = await engine.start_strategy(strategy_id)
        if success:
            return {"message": "Strategy started successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to start strategy")
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/stop")
async def stop_strategy(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Stop a strategy"""
    try:
        success = await engine.stop_strategy(strategy_id)
        if success:
            return {"message": "Strategy stopped successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to stop strategy")
    except Exception as e:
        logger.error(f"Error stopping strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Delete a strategy"""
    try:
        success = await engine.delete_strategy(strategy_id)
        if success:
            return {"message": "Strategy deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete strategy")
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[Strategy])
async def get_all_strategies(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get all strategies"""
    try:
        strategies = engine.get_all_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[Strategy])
async def get_active_strategies(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get active strategies"""
    try:
        strategies = engine.get_active_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting active strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}", response_model=Strategy)
async def get_strategy(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get strategy by ID"""
    try:
        strategy = engine.get_strategy(strategy_id)
        if strategy is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except Exception as e:
        logger.error(f"Error getting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Signal Management Endpoints

@router.post("/signals/generate")
async def generate_signals(
    symbols: List[str],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Generate signals for symbols"""
    try:
        # This would typically be called internally by the strategy engine
        # For now, return a placeholder response
        return {"message": "Signal generation initiated", "symbols": symbols}
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals", response_model=List[Signal])
async def get_signals(
    limit: int = 100,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get recent signals"""
    try:
        signals = engine.get_signals(limit)
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/signals", response_model=List[Signal])
async def get_strategy_signals(
    strategy_id: str,
    limit: int = 100,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get signals for a specific strategy"""
    try:
        signals = engine.get_strategy_signals(strategy_id, limit)
        return signals
    except Exception as e:
        logger.error(f"Error getting strategy signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy Discovery Endpoints

@router.post("/discover", response_model=List[Strategy])
async def discover_strategies(
    discovery_params: Dict[str, Any],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Discover new strategies using AI"""
    try:
        strategies = await engine.discover_strategies(discovery_params)
        return strategies
    except Exception as e:
        logger.error(f"Error discovering strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/evolve", response_model=Strategy)
async def evolve_strategy(
    strategy_id: str,
    evolution_params: Dict[str, Any],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Evolve a strategy using genetic algorithms"""
    try:
        evolved_strategy = await engine.evolve_strategy(strategy_id, evolution_params)
        return evolved_strategy
    except Exception as e:
        logger.error(f"Error evolving strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting Endpoints

@router.post("/{strategy_id}/backtest", response_model=BacktestResult)
async def backtest_strategy(
    strategy_id: str,
    backtest_params: Dict[str, Any],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Backtest a strategy"""
    try:
        result = await engine.backtest_strategy(strategy_id, backtest_params)
        return result
    except Exception as e:
        logger.error(f"Error backtesting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints

@router.get("/{strategy_id}/analytics")
async def get_strategy_analytics(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get strategy analytics"""
    try:
        analytics = await engine.get_strategy_analytics(strategy_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting strategy analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/analytics")
async def get_portfolio_analytics(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get portfolio analytics"""
    try:
        analytics = await engine.get_portfolio_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Error getting portfolio analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Optimization Endpoints

@router.post("/{strategy_id}/optimize", response_model=Strategy)
async def optimize_strategy(
    strategy_id: str,
    optimization_params: Dict[str, Any],
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Optimize a strategy"""
    try:
        optimized_strategy = await engine.optimize_strategy(strategy_id, optimization_params)
        return optimized_strategy
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Management Endpoints

@router.get("/{strategy_id}/risk")
async def check_strategy_risk(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Check strategy risk"""
    try:
        risk_assessment = await engine.check_strategy_risk(strategy_id)
        return risk_assessment
    except Exception as e:
        logger.error(f"Error checking strategy risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance Endpoints

@router.get("/{strategy_id}/performance", response_model=StrategyPerformance)
async def get_strategy_performance(
    strategy_id: str,
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get strategy performance"""
    try:
        performance = engine.get_strategy_performance(strategy_id)
        if performance is None:
            raise HTTPException(status_code=404, detail="Strategy performance not found")
        return performance
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/all")
async def get_all_strategy_performance(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get all strategy performance"""
    try:
        performance = engine.get_all_strategy_performance()
        return performance
    except Exception as e:
        logger.error(f"Error getting all strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/metrics")
async def get_performance_metrics(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Get performance metrics"""
    try:
        metrics = engine.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint

@router.get("/health")
async def health_check(
    engine: StrategyEngine = Depends(get_strategy_engine)
):
    """Perform health check"""
    try:
        health_status = await engine.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

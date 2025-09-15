"""
Portfolio Management API Endpoints
Complete interface to portfolio operations
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.portfolio_management.portfolio_engine import PortfolioEngine
# from app.core.auth import get_current_user  # Uncomment when auth is implemented

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Initialize portfolio engine (in production, use dependency injection)
portfolio_engine = None

class PortfolioConstructRequest(BaseModel):
    """Request to construct new portfolio"""
    capital: float = Field(..., gt=0, description="Initial capital")
    objectives: Dict = Field(..., description="Investment objectives")
    constraints: Optional[Dict] = Field(None, description="Portfolio constraints")
    method: str = Field("ensemble", description="Optimization method")

class RebalanceRequest(BaseModel):
    """Request to rebalance portfolio"""
    force: bool = Field(False, description="Force rebalancing")
    tax_aware: bool = Field(True, description="Use tax-aware rebalancing")

class PositionRequest(BaseModel):
    """Request to add position"""
    symbol: str
    quantity: int = Field(..., gt=0)
    strategy: str = Field("portfolio", description="Strategy name")
    check_risk: bool = Field(True, description="Run risk checks")

@router.post("/construct")
async def construct_portfolio(
    request: PortfolioConstructRequest,
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Construct an optimal portfolio from scratch
    This is where wealth creation begins
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        portfolio = await portfolio_engine.construct_portfolio(
            capital=request.capital,
            objectives=request.objectives,
            constraints=request.constraints
        )
        
        return {
            "success": True,
            "portfolio_id": portfolio.portfolio_id,
            "positions": len(portfolio.positions),
            "total_value": portfolio.total_value,
            "cash": portfolio.cash,
            "metrics": portfolio.metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio construction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_portfolio(
    method: str = Query("ensemble", description="Optimization method"),
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Optimize existing portfolio
    Continuous improvement for maximum returns
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        results = await portfolio_engine.optimize_portfolio(method=method)
        
        return {
            "success": True,
            "optimization_method": method,
            "current_metrics": results["current_metrics"],
            "optimized_weights": results["optimized_weights"],
            "required_trades": len(results["required_trades"]),
            "estimated_costs": results["estimated_costs"],
            "expected_improvement": results["expected_improvement"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebalance")
async def rebalance_portfolio(
    request: RebalanceRequest,
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Rebalance portfolio to target weights
    Discipline that generates alpha
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        results = await portfolio_engine.rebalance_portfolio(
            force=request.force,
            tax_aware=request.tax_aware
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Portfolio rebalancing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_portfolio_analytics(
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Get comprehensive portfolio analytics
    Know exactly how your portfolio performs
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        metrics = portfolio_engine.real_time_metrics
        
        return {
            "portfolio_value": metrics.get("portfolio_value", 0),
            "performance": {
                "daily_return": metrics.get("daily_return", 0),
                "mtd_return": await portfolio_engine.analytics.calculate_daily_return(),
                "ytd_return": 0.0,  # Placeholder
                "total_return": 0.0  # Placeholder
            },
            "risk_metrics": {
                "volatility": metrics.get("volatility", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "var_95": metrics.get("var_95", 0),
                "cvar_95": metrics.get("cvar_95", 0)
            },
            "factor_exposures": metrics.get("factor_exposures", {}),
            "regime": portfolio_engine.current_regime.value,
            "regime_probability": metrics.get("regime_probability", {}),
            "correlation_matrix": metrics.get("correlation_matrix"),
            "updated_at": metrics.get("updated_at")
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/attribution")
async def get_performance_attribution(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Get performance attribution analysis
    Understand exactly where returns come from
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        attribution = await portfolio_engine.attributor.calculate_attribution(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "total_return": attribution.get("total_return", 0),
            "attribution": {
                "asset_allocation": attribution.get("asset_allocation", 0),
                "security_selection": attribution.get("security_selection", 0),
                "market_timing": attribution.get("market_timing", 0),
                "currency_effect": attribution.get("currency", 0)
            },
            "factor_attribution": attribution.get("factor_attribution", {}),
            "sector_attribution": attribution.get("sector_attribution", {}),
            "best_performers": attribution.get("best_performers", []),
            "worst_performers": attribution.get("worst_performers", []),
            "period": f"{start_date} to {end_date}" if start_date else "inception to date",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance attribution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tax-harvest")
async def run_tax_harvesting(
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Run tax loss harvesting
    The secret to keeping more of what you earn
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        if not portfolio_engine.config.get("TAX_OPTIMIZATION_ENABLED", False):
            return {"success": False, "reason": "Tax optimization disabled"}
        
        results = await portfolio_engine.tax_optimizer.harvest_losses(
            portfolio_engine.portfolio.positions
        )
        
        return {
            "success": True,
            "harvested_positions": results["harvested_positions"],
            "total_loss_harvested": results["total_loss_harvested"],
            "estimated_tax_savings": results["estimated_tax_savings"],
            "replacements_suggested": len([h for h in results["harvested_positions"] if "replacement" in h]),
            "timestamp": results["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Tax harvesting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions")
async def get_positions(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """Get current portfolio positions"""
    try:
        if not portfolio_engine or not portfolio_engine.portfolio:
            raise HTTPException(status_code=404, detail="No portfolio found")
        
        positions = portfolio_engine.portfolio.positions
        
        if strategy:
            positions = [p for p in positions if p.strategy == strategy]
        
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                    "current_weight": p.current_weight,
                    "target_weight": p.target_weight,
                    "strategy": p.strategy,
                    "holding_period_days": (datetime.utcnow() - p.entry_date).days
                }
                for p in positions
            ],
            "total_positions": len(positions),
            "total_value": sum(p.quantity * p.current_price for p in positions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/position")
async def add_position(
    request: PositionRequest,
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """Add new position to portfolio"""
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        result = await portfolio_engine.add_position(
            symbol=request.symbol,
            quantity=request.quantity,
            strategy=request.strategy,
            check_risk=request.check_risk
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to add position: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest")
async def run_backtest(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    initial_capital: float = Query(100000, description="Initial capital"),
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """
    Run portfolio backtest
    Test strategies before risking real money
    """
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        from app.portfolio_management.portfolio_backtester import PortfolioBacktester
        
        backtester = PortfolioBacktester(portfolio_engine.config)
        results = await backtester.run_backtest(
            strategy=portfolio_engine.portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        return {
            "performance": {
                "total_return": results["total_return"],
                "annualized_return": results["annualized_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "win_rate": results["win_rate"]
            },
            "statistics": results["statistics"],
            "equity_curve": results["equity_curve"],
            "trades": len(results["trades"]),
            "period": f"{start_date} to {end_date}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regime")
async def get_market_regime(
    # current_user = Depends(get_current_user)  # Uncomment when auth is implemented
) -> Dict:
    """Get current market regime detection"""
    try:
        if not portfolio_engine:
            raise HTTPException(status_code=503, detail="Portfolio engine not initialized")
        
        regime_probs = await portfolio_engine.regime_detector.get_regime_probabilities()
        
        return {
            "current_regime": portfolio_engine.current_regime.value,
            "regime_probabilities": regime_probs,
            "regime_parameters": portfolio_engine.config.get("REGIME_SPECIFIC_PARAMS", {}).get(
                portfolio_engine.current_regime.value, {}
            ),
            "recommendation": await portfolio_engine.regime_detector.get_regime_recommendation(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get regime: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize portfolio engine
def init_portfolio_engine(config: Dict):
    global portfolio_engine
    portfolio_engine = PortfolioEngine(config)
    import asyncio
    asyncio.create_task(portfolio_engine.initialize())
    logger.info("✅ Portfolio Management API initialized")

# backend/app/api/routers/risk.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RiskMetrics(BaseModel):
    var_95: float
    var_99: float
    max_position_size: float
    current_exposure: float
    risk_score: int

@router.get('/metrics', response_model=RiskMetrics)
async def get_risk_metrics():
    return RiskMetrics(
        var_95=5000.00,
        var_99=7500.00,
        max_position_size=10000.00,
        current_exposure=25000.00,
        risk_score=65
    )

@router.get('/limits')
async def get_risk_limits():
    return {
        'max_position_size': 10000,
        'max_daily_loss': 5000,
        'max_leverage': 2.0,
        'position_limit': 10
    }

# backend/app/api/routers/portfolio.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    positions_value: float
    daily_pnl: float
    total_pnl: float

@router.get('/summary', response_model=PortfolioSummary)
async def get_portfolio_summary():
    return PortfolioSummary(
        total_value=100000.00,
        cash_balance=50000.00,
        positions_value=50000.00,
        daily_pnl=500.00,
        total_pnl=5000.00
    )

@router.get('/performance')
async def get_performance():
    return {
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.10,
        'win_rate': 0.65,
        'profit_factor': 1.8
    }

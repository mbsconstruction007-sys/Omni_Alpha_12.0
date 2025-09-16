# backend/app/api/routers/market_data.py
'''Market data endpoints'''

from fastapi import APIRouter, Query, WebSocket
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

router = APIRouter()

class Quote(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime

class Bar(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime

@router.get('/quote/{symbol}', response_model=Quote)
async def get_quote(symbol: str):
    '''Get real-time quote for a symbol'''
    return Quote(
        symbol=symbol,
        bid=150.00,
        ask=150.02,
        last=150.01,
        volume=1000000,
        timestamp=datetime.now()
    )

@router.get('/bars/{symbol}', response_model=List[Bar])
async def get_bars(
    symbol: str,
    timeframe: str = '1D',
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(100, le=1000)
):
    '''Get historical bars'''
    # Mock data
    bars = []
    for i in range(min(limit, 10)):
        bars.append(Bar(
            symbol=symbol,
            open=150.0 + i,
            high=152.0 + i,
            low=149.0 + i,
            close=151.0 + i,
            volume=1000000 + i * 10000,
            timestamp=datetime.now() - timedelta(days=i)
        ))
    return bars

@router.websocket('/stream/{symbol}')
async def stream_market_data(websocket: WebSocket, symbol: str):
    '''WebSocket endpoint for streaming market data'''
    await websocket.accept()
    try:
        while True:
            # Send mock data every second
            await websocket.send_json({
                'symbol': symbol,
                'price': 150.00,
                'volume': 1000,
                'timestamp': datetime.now().isoformat()
            })
            await asyncio.sleep(1)
    except Exception as e:
        await websocket.close()

@router.get('/symbols')
async def get_symbols(exchange: Optional[str] = None):
    '''Get available symbols'''
    return {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'count': 5,
        'exchange': exchange or 'ALL'
    }

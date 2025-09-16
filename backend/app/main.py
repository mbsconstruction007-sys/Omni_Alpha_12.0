# backend/app/main.py
'''
Omni Alpha 12.0 - Main FastAPI Application
'''

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting Omni Alpha 12.0 API...')
    yield
    logger.info('Shutting down Omni Alpha 12.0 API...')

# Create FastAPI app
app = FastAPI(
    title='Omni Alpha 12.0 Trading System',
    description='Advanced algorithmic trading system with AI integration',
    version='12.0.0',
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Basic endpoints first (no router imports yet)
@app.get('/')
async def root():
    return {
        'name': 'Omni Alpha 12.0',
        'status': 'operational',
        'version': '12.0.0',
        'timestamp': datetime.now().isoformat()
    }

@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'services': {
            'api': 'running',
            'version': '12.0.0'
        }
    }

# Try to import routers - if they fail, API still works
try:
    from api.routers import trading, market_data, portfolio, risk, analytics, system
    app.include_router(trading.router, prefix='/api/v1/trading', tags=['trading'])
    app.include_router(market_data.router, prefix='/api/v1/market', tags=['market'])
    app.include_router(portfolio.router, prefix='/api/v1/portfolio', tags=['portfolio'])
    app.include_router(risk.router, prefix='/api/v1/risk', tags=['risk'])
    app.include_router(analytics.router, prefix='/api/v1/analytics', tags=['analytics'])
    app.include_router(system.router, prefix='/api/v1/system', tags=['system'])
    logger.info('All routers loaded successfully')
except ImportError as e:
    logger.warning(f'Could not import routers: {e}')
    logger.info('API running with basic endpoints only')

# WebSocket endpoint
@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f'Echo: {data}')
    except WebSocketDisconnect:
        logger.info('WebSocket disconnected')

if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8000,
        reload=True,
        log_level='info'
    )

# backend/app/api/routers/system.py
from fastapi import APIRouter
import psutil
import platform

router = APIRouter()

@router.get('/status')
async def get_system_status():
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'python_version': platform.python_version(),
        'system': platform.system()
    }

@router.get('/config')
async def get_config():
    return {
        'trading_mode': 'paper',
        'api_version': '12.0.0',
        'features_enabled': ['ai', 'risk_management', 'auto_trading']
    }

@router.post('/shutdown')
async def shutdown_system():
    return {'message': 'System shutdown initiated'}

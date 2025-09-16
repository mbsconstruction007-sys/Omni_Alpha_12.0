# run_api.py
'''Launch script for FastAPI'''

import uvicorn
from backend.app.main import app

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        reload=True,
        log_level='info'
    )

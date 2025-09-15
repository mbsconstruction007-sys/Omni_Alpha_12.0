"""
Run script for Institutional Trading System
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('institutional_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="Institutional Trading System",
        description="Step 11: Institutional Operations & Alpha Amplification",
        version="11.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from app.api.institutional_api import router as institutional_router
    app.include_router(institutional_router)
    
    @app.get("/")
    async def root():
        return {
            "system": "Institutional Trading System",
            "version": "11.0.0",
            "status": "operational",
            "description": "The most sophisticated institutional trading framework"
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return app

async def main():
    """Main entry point"""
    logger.info("Starting Institutional Trading System...")
    
    # Create FastAPI app
    app = create_app()
    
    # Run with uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8011,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Institutional Trading System...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

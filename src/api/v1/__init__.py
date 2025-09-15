"""
API v1 Package
"""

from fastapi import APIRouter
from .endpoints import orders

api_router = APIRouter()
api_router.include_router(orders.router, prefix="/api/v1", tags=["orders"])

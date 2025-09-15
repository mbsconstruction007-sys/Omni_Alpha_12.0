"""
Order Management API Endpoints
RESTful API for order operations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from src.oms.models import Order, OrderRequest, OrderUpdate, OrderStatus
from src.oms.manager import OrderManager
from src.database.connection import db_manager

router = APIRouter(prefix="/orders", tags=["orders"])

# Dependency injection
async def get_order_manager() -> OrderManager:
    """Get order manager instance"""
    # This would be properly initialized with all dependencies
    # For now, returning a placeholder that would be injected
    from src.oms.executor import OrderExecutor
    from src.oms.risk_checker import RiskChecker
    from src.oms.position_manager import PositionManager
    from src.database.repositories.order_repository import OrderRepository
    from src.brokers import BrokerManager
    
    # Initialize components (in production, this would be done via dependency injection)
    broker_manager = BrokerManager()
    executor = OrderExecutor(broker_manager, {})
    risk_checker = RiskChecker(broker_manager, {})
    position_manager = PositionManager(broker_manager)
    repository = OrderRepository(db_manager)
    
    return OrderManager(
        executor=executor,
        risk_checker=risk_checker,
        position_manager=position_manager,
        repository=repository,
        broker_manager=broker_manager
    )

@router.post("/", response_model=Order, status_code=status.HTTP_201_CREATED)
async def create_order(
    request: OrderRequest,
    manager: OrderManager = Depends(get_order_manager)
):
    """
    Create a new order
    
    - **symbol**: Trading symbol (e.g., AAPL, GOOGL)
    - **side**: Order side (buy/sell)
    - **quantity**: Number of shares
    - **order_type**: Type of order (market/limit/stop)
    - **limit_price**: Limit price for limit orders
    - **stop_price**: Stop price for stop orders
    """
    try:
        # Create order
        order = await manager.create_order(request)
        
        return order
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}"
        )

@router.get("/{order_id}", response_model=Order)
async def get_order(
    order_id: str,
    manager: OrderManager = Depends(get_order_manager)
):
    """Get order by ID"""
    order = await manager.get_order(order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return order

@router.get("/", response_model=List[Order])
async def list_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[OrderStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    manager: OrderManager = Depends(get_order_manager)
):
    """
    List orders with optional filters
    
    Query parameters:
    - **symbol**: Filter by trading symbol
    - **status**: Filter by order status
    - **start_date**: Filter orders created after this date
    - **end_date**: Filter orders created before this date
    - **limit**: Maximum number of results (default: 100, max: 1000)
    - **offset**: Pagination offset
    """
    # Get orders based on status
    if status and status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
        orders = await manager.get_active_orders(symbol=symbol)
    else:
        orders = await manager.get_order_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    # Apply pagination
    return orders[offset:offset + limit]

@router.patch("/{order_id}", response_model=Order)
async def modify_order(
    order_id: str,
    update: OrderUpdate,
    manager: OrderManager = Depends(get_order_manager)
):
    """
    Modify an existing order
    
    Only NEW and PARTIALLY_FILLED orders can be modified.
    """
    # Get order
    order = await manager.get_order(order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    try:
        # Modify order
        modified_order = await manager.modify_order(order_id, update)
        return modified_order
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify order: {str(e)}"
        )

@router.delete("/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(
    order_id: str,
    reason: str = Query("User requested", description="Cancellation reason"),
    manager: OrderManager = Depends(get_order_manager)
):
    """Cancel an order"""
    # Get order
    order = await manager.get_order(order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    try:
        # Cancel order
        await manager.cancel_order(order_id, reason=reason)
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}"
        )

@router.post("/cancel-all", response_model=Dict[str, Any])
async def cancel_all_orders(
    symbol: Optional[str] = Query(None, description="Cancel orders for specific symbol"),
    manager: OrderManager = Depends(get_order_manager)
):
    """Cancel all active orders"""
    try:
        # Get active orders
        orders = await manager.get_active_orders(symbol=symbol)
        
        # Cancel each order
        cancelled = []
        failed = []
        
        for order in orders:
            try:
                await manager.cancel_order(order.order_id, reason="Bulk cancellation")
                cancelled.append(order.order_id)
            except Exception as e:
                failed.append({
                    'order_id': order.order_id,
                    'error': str(e)
                })
        
        return {
            'cancelled': cancelled,
            'failed': failed,
            'total': len(orders)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel orders: {str(e)}"
        )

@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_order_metrics(
    manager: OrderManager = Depends(get_order_manager)
):
    """Get order metrics and statistics"""
    try:
        metrics = await manager.get_metrics()
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.get("/positions/current", response_model=List[Dict[str, Any]])
async def get_positions(
    manager: OrderManager = Depends(get_order_manager)
):
    """Get current positions"""
    try:
        # Get positions from position manager
        positions = await manager.position_manager.get_positions()
        
        return [
            {
                'symbol': p.symbol,
                'quantity': float(p.quantity),
                'side': p.side,
                'average_price': float(p.average_entry_price),
                'current_price': float(p.current_price) if p.current_price else None,
                'unrealized_pnl': float(p.unrealized_pnl) if p.unrealized_pnl else None,
                'realized_pnl': float(p.realized_pnl),
                'market_value': float(p.market_value) if p.market_value else None
            }
            for p in positions
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )

@router.get("/risk/metrics", response_model=Dict[str, Any])
async def get_risk_metrics(
    manager: OrderManager = Depends(get_order_manager)
):
    """Get current risk metrics"""
    try:
        # Get risk metrics from risk checker
        metrics = await manager.risk_checker.get_risk_metrics()
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get risk metrics: {str(e)}"
        )

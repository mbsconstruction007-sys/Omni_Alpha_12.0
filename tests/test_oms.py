"""
Order Management System Tests
Comprehensive test suite for OMS components
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.oms.models import (
    Order, OrderRequest, OrderStatus, 
    OrderType, OrderSide, TimeInForce
)
from src.oms.manager import OrderManager
from src.oms.executor import OrderExecutor
from src.oms.risk_checker import RiskChecker, RiskCheckResult

# Fixtures

@pytest.fixture
async def order_manager():
    """Create order manager instance for testing"""
    executor = Mock(spec=OrderExecutor)
    risk_checker = Mock(spec=RiskChecker)
    position_manager = Mock()
    repository = Mock()
    broker_manager = Mock()
    
    manager = OrderManager(
        executor=executor,
        risk_checker=risk_checker,
        position_manager=position_manager,
        repository=repository,
        broker_manager=broker_manager
    )
    
    await manager.start()
    yield manager
    await manager.stop()

@pytest.fixture
def sample_order_request():
    """Create sample order request"""
    return OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        time_in_force=TimeInForce.DAY
    )

# Order Creation Tests

@pytest.mark.asyncio
async def test_create_market_order(order_manager, sample_order_request):
    """Test creating a market order"""
    # Setup
    sample_order_request.order_type = OrderType.MARKET
    sample_order_request.limit_price = None
    
    order_manager.risk_checker.check_order = AsyncMock(
        return_value=Mock(passed=True, details={})
    )
    
    # Execute
    order = await order_manager.create_order(sample_order_request)
    
    # Assert
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.quantity == Decimal("100")
    assert order.order_type == OrderType.MARKET
    assert order.status == OrderStatus.PENDING_NEW
    assert order.order_id.startswith("OMNI5_")

@pytest.mark.asyncio
async def test_create_limit_order(order_manager, sample_order_request):
    """Test creating a limit order"""
    # Setup
    order_manager.risk_checker.check_order = AsyncMock(
        return_value=Mock(passed=True, details={})
    )
    
    # Execute
    order = await order_manager.create_order(sample_order_request)
    
    # Assert
    assert order.order_type == OrderType.LIMIT
    assert order.limit_price == Decimal("150.00")
    assert order.status == OrderStatus.PENDING_NEW

@pytest.mark.asyncio
async def test_order_rejected_by_risk_check(order_manager, sample_order_request):
    """Test order rejection due to failed risk check"""
    # Setup
    order_manager.risk_checker.check_order = AsyncMock(
        return_value=Mock(
            passed=False, 
            reason="Order value exceeds limit",
            details={'order_value': 15000}
        )
    )
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Order rejected"):
        await order_manager.create_order(sample_order_request)

# Order Modification Tests

@pytest.mark.asyncio
async def test_modify_order_quantity(order_manager):
    """Test modifying order quantity"""
    # Setup
    order = Order(
        order_id="test_order_1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        status=OrderStatus.NEW
    )
    
    order_manager.active_orders[order.order_id] = order
    order_manager.risk_checker.check_order = AsyncMock(
        return_value=Mock(passed=True, details={})
    )
    order_manager.executor.modify_order = AsyncMock()
    
    # Execute
    update = Mock(quantity=Decimal("200"), limit_price=None, stop_price=None)
    modified_order = await order_manager.modify_order(order.order_id, update)
    
    # Assert
    assert modified_order.quantity == Decimal("200")
    assert modified_order.status == OrderStatus.REPLACED

# Order Cancellation Tests

@pytest.mark.asyncio
async def test_cancel_order(order_manager):
    """Test cancelling an order"""
    # Setup
    order = Order(
        order_id="test_order_2",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        status=OrderStatus.NEW
    )
    
    order_manager.active_orders[order.order_id] = order
    order_manager.executor.cancel_order = AsyncMock()
    
    # Execute
    cancelled_order = await order_manager.cancel_order(
        order.order_id, 
        reason="User requested"
    )
    
    # Assert
    assert cancelled_order.status == OrderStatus.CANCELLED
    assert order.order_id not in order_manager.active_orders
    assert cancelled_order.cancelled_at is not None

@pytest.mark.asyncio
async def test_cannot_cancel_filled_order(order_manager):
    """Test that filled orders cannot be cancelled"""
    # Setup
    order = Order(
        order_id="test_order_3",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED
    )
    
    order_manager.active_orders[order.order_id] = order
    
    # Execute & Assert
    with pytest.raises(ValueError, match="Cannot cancel order in status"):
        await order_manager.cancel_order(order.order_id)

# Fill Handling Tests

@pytest.mark.asyncio
async def test_handle_partial_fill(order_manager):
    """Test handling a partial fill"""
    # Setup
    order = Order(
        order_id="test_order_4",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        filled_quantity=Decimal("0"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00"),
        status=OrderStatus.NEW
    )
    
    order_manager.active_orders[order.order_id] = order
    order_manager.position_manager.update_position = AsyncMock()
    
    fill = Mock(
        order_id="test_order_4",
        quantity=Decimal("50"),
        price=Decimal("149.50"),
        commission=Decimal("1.00")
    )
    
    # Execute
    await order_manager.handle_fill(fill)
    
    # Assert
    assert order.filled_quantity == Decimal("50")
    assert order.remaining_quantity == Decimal("50")
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.average_fill_price == Decimal("149.50")

@pytest.mark.asyncio
async def test_handle_complete_fill(order_manager):
    """Test handling a complete fill"""
    # Setup
    order = Order(
        order_id="test_order_5",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        filled_quantity=Decimal("0"),
        order_type=OrderType.MARKET,
        status=OrderStatus.NEW
    )
    
    order_manager.active_orders[order.order_id] = order
    order_manager.position_manager.update_position = AsyncMock()
    
    fill = Mock(
        order_id="test_order_5",
        quantity=Decimal("100"),
        price=Decimal("150.25"),
        commission=Decimal("2.00")
    )
    
    # Execute
    await order_manager.handle_fill(fill)
    
    # Assert
    assert order.filled_quantity == Decimal("100")
    assert order.remaining_quantity == Decimal("0")
    assert order.status == OrderStatus.FILLED
    assert order.order_id not in order_manager.active_orders
    assert order.filled_at is not None

# Risk Check Tests

@pytest.mark.asyncio
async def test_risk_check_order_value_limit():
    """Test order value limit risk check"""
    # Setup
    broker_manager = Mock()
    broker_manager.get_primary_broker = AsyncMock(return_value=Mock())
    
    risk_checker = RiskChecker(
        broker_manager=broker_manager,
        config={'max_order_value': 10000}
    )
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    # Execute
    result = await risk_checker.check_order(order)
    
    # Assert
    assert result.passed is False
    assert "exceeds limit" in result.reason

@pytest.mark.asyncio
async def test_risk_check_position_concentration():
    """Test position concentration risk check"""
    # Setup
    broker_manager = Mock()
    broker_manager.get_primary_broker = AsyncMock(return_value=Mock())
    
    risk_checker = RiskChecker(
        broker_manager=broker_manager,
        config={
            'max_order_value': 100000,
            'max_concentration': 0.20
        }
    )
    
    # Set existing positions
    risk_checker.positions = {'AAPL': Decimal('1000')}
    risk_checker.total_exposure = Decimal('10000')
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5000"),
        order_type=OrderType.MARKET
    )
    
    # Execute
    result = await risk_checker.check_order(order)
    
    # Assert
    assert result.passed is False
    assert "concentration" in result.reason.lower()

# Integration Tests

@pytest.mark.asyncio
async def test_full_order_lifecycle():
    """Test complete order lifecycle from creation to fill"""
    # This would be a more complex integration test
    # testing the full flow with mocked broker connections
    pass

# Performance Tests

@pytest.mark.asyncio
async def test_order_creation_performance(order_manager):
    """Test order creation performance"""
    # Setup
    order_manager.risk_checker.check_order = AsyncMock(
        return_value=Mock(passed=True, details={})
    )
    
    # Execute
    start_time = datetime.utcnow()
    
    tasks = []
    for i in range(100):
        request = OrderRequest(
            symbol=f"TEST{i}",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        tasks.append(order_manager.create_order(request))
    
    await asyncio.gather(*tasks)
    
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    
    # Assert - should handle 100 orders in under 1 second
    assert elapsed < 1.0
    assert len(order_manager.active_orders) == 100

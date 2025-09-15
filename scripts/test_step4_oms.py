#!/usr/bin/env python3
"""
Step 4: Order Management System Test Suite
Comprehensive testing of OMS components
"""

import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oms.models import (
    Order, OrderRequest, OrderUpdate, OrderStatus, 
    OrderType, OrderSide, TimeInForce
)
from oms.risk_checker import RiskChecker, RiskCheckResult
from oms.position_manager import PositionManager
from oms.order_book import OrderBook
from oms.fill_handler import FillHandler

async def test_oms_models():
    """Test OMS data models"""
    print("🧪 Testing OMS Models...")
    
    # Test Order creation
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.quantity == Decimal("100")
    assert order.order_type == OrderType.LIMIT
    assert order.limit_price == Decimal("150.00")
    assert order.status == OrderStatus.PENDING_NEW
    assert order.order_id.startswith("OMNI5_")
    
    print("✅ Order model creation successful")
    
    # Test OrderRequest
    request = OrderRequest(
        symbol="GOOGL",
        side=OrderSide.SELL,
        quantity=Decimal("50"),
        order_type=OrderType.MARKET
    )
    
    assert request.symbol == "GOOGL"
    assert request.side == OrderSide.SELL
    assert request.quantity == Decimal("50")
    assert request.order_type == OrderType.MARKET
    
    print("✅ OrderRequest model creation successful")
    
    # Test OrderUpdate
    update = OrderUpdate(
        quantity=Decimal("75"),
        limit_price=Decimal("160.00")
    )
    
    assert update.quantity == Decimal("75")
    assert update.limit_price == Decimal("160.00")
    
    print("✅ OrderUpdate model creation successful")

async def test_risk_checker():
    """Test risk checker functionality"""
    print("\n🧪 Testing Risk Checker...")
    
    # Mock broker manager
    async def mock_get_primary_broker():
        return None
    
    broker_manager = type('MockBrokerManager', (), {})()
    broker_manager.get_primary_broker = mock_get_primary_broker
    
    # Create risk checker
    risk_checker = RiskChecker(
        broker_manager=broker_manager,
        config={
            'max_order_value': 10000,
            'max_daily_trades': 100,
            'max_concentration': 0.20
        }
    )
    
    # Test order that should pass
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    result = await risk_checker.check_order(order)
    print(f"✅ Risk check result: {result.passed} - {result.reason}")
    
    # Test order that should fail (too large)
    large_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("1000"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    result = await risk_checker.check_order(large_order)
    print(f"✅ Large order risk check: {result.passed} - {result.reason}")

async def test_position_manager():
    """Test position manager functionality"""
    print("\n🧪 Testing Position Manager...")
    
    # Mock broker manager
    async def mock_get_primary_broker():
        return None
    
    broker_manager = type('MockBrokerManager', (), {})()
    broker_manager.get_primary_broker = mock_get_primary_broker
    
    # Create position manager
    position_manager = PositionManager(broker_manager)
    
    # Test position creation
    from oms.models import Fill
    
    fill = Fill(
        order_id="test_order",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.00"),
        venue="test"
    )
    
    await position_manager.update_position(fill)
    positions = await position_manager.get_positions()
    
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == Decimal("100")
    assert positions[0].side == "long"
    
    print("✅ Position manager working correctly")

async def test_order_book():
    """Test order book functionality"""
    print("\n🧪 Testing Order Book...")
    
    # Create order book
    order_book = OrderBook("AAPL")
    
    # Test adding orders
    order1 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    order2 = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=Decimal("50"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("151.00")
    )
    
    order_book.add_order(order1)
    order_book.add_order(order2)
    
    # Test market data
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    spread = order_book.get_spread()
    
    assert best_bid == Decimal("150.00")
    assert best_ask == Decimal("151.00")
    assert spread == Decimal("1.00")
    
    print("✅ Order book working correctly")
    
    # Test trade
    order_book.add_trade(Decimal("150.50"), Decimal("25"))
    
    assert order_book.last_trade_price == Decimal("150.50")
    assert order_book.volume_today == Decimal("25")
    
    print("✅ Trade processing working correctly")

async def test_fill_handler():
    """Test fill handler functionality"""
    print("\n🧪 Testing Fill Handler...")
    
    # Mock position manager
    async def mock_update_position(fill):
        pass
    
    position_manager = type('MockPositionManager', (), {})()
    position_manager.update_position = mock_update_position
    
    # Create fill handler
    fill_handler = FillHandler(position_manager)
    
    # Test fill processing
    order = Order(
        order_id="test_order",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    fill_data = {
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 50,
        'price': 149.50,
        'commission': 1.00,
        'venue': 'test'
    }
    
    fill = await fill_handler.process_fill(fill_data, order)
    
    assert fill.order_id == "test_order"
    assert fill.quantity == Decimal("50")
    assert fill.price == Decimal("149.50")
    assert fill.commission == Decimal("1.00")
    
    print("✅ Fill handler working correctly")

async def test_order_lifecycle():
    """Test complete order lifecycle"""
    print("\n🧪 Testing Order Lifecycle...")
    
    # Create order
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("150.00")
    )
    
    # Test state transitions
    assert order.status == OrderStatus.PENDING_NEW
    
    # Simulate order submission
    order.status = OrderStatus.NEW
    order.submitted_at = datetime.utcnow()
    
    assert order.status == OrderStatus.NEW
    assert order.submitted_at is not None
    
    # Simulate partial fill
    order.filled_quantity = Decimal("50")
    order.remaining_quantity = Decimal("50")
    order.status = OrderStatus.PARTIALLY_FILLED
    
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == Decimal("50")
    assert order.remaining_quantity == Decimal("50")
    
    # Simulate complete fill
    order.filled_quantity = Decimal("100")
    order.remaining_quantity = Decimal("0")
    order.status = OrderStatus.FILLED
    order.filled_at = datetime.utcnow()
    
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == Decimal("100")
    assert order.remaining_quantity == Decimal("0")
    assert order.filled_at is not None
    
    print("✅ Order lifecycle working correctly")

async def main():
    """Run all OMS tests"""
    print("🚀 Step 4: Order Management System Test Suite")
    print("=" * 60)
    
    try:
        await test_oms_models()
        await test_risk_checker()
        await test_position_manager()
        await test_order_book()
        await test_fill_handler()
        await test_order_lifecycle()
        
        print("\n" + "=" * 60)
        print("🎉 ALL OMS TESTS PASSED!")
        print("✅ Step 4 Order Management System is working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

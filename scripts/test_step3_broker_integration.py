#!/usr/bin/env python3
"""
Test Step 3: Broker Integration System
Comprehensive test for the multi-broker system
"""

import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brokers import BrokerManager, BrokerType, BrokerConfig, AlpacaBroker
from src.database.models import (
    Order, OrderType, OrderSide, TimeInForce, 
    AssetType, ExchangeType
)

async def test_broker_integration():
    """Test the complete broker integration system"""
    
    print("🚀 Testing Step 3: Broker Integration System")
    print("=" * 60)
    
    # Test 1: Broker Manager Initialization
    print("\n1️⃣ Testing Broker Manager Initialization...")
    manager = BrokerManager()
    assert not manager._initialized
    print("✅ Broker manager created successfully")
    
    # Test 2: Broker Configuration
    print("\n2️⃣ Testing Broker Configuration...")
    alpaca_config = BrokerConfig(
        name="alpaca_test",
        api_key="test_key",
        secret_key="test_secret",
        base_url="https://paper-api.alpaca.markets",
        paper_trading=True
    )
    print("✅ Broker configuration created successfully")
    
    # Test 3: Alpaca Broker Creation
    print("\n3️⃣ Testing Alpaca Broker Creation...")
    alpaca_broker = AlpacaBroker(alpaca_config)
    assert alpaca_broker.config.name == "alpaca_test"
    assert alpaca_broker.config.paper_trading == True
    print("✅ Alpaca broker created successfully")
    
    # Test 4: Order Validation
    print("\n4️⃣ Testing Order Validation...")
    test_order = Order(
        order_id="test_123",
        account_id="test_account",
        symbol="AAPL",
        asset_type=AssetType.STOCK,
        exchange=ExchangeType.NASDAQ,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("10"),
        time_in_force=TimeInForce.DAY
    )
    
    # Test validation (should pass)
    try:
        await alpaca_broker.validate_order(test_order)
        print("✅ Order validation passed")
    except Exception as e:
        print(f"❌ Order validation failed: {e}")
        return False
    
    # Test 5: Invalid Order Validation
    print("\n5️⃣ Testing Invalid Order Validation...")
    
    # Test with empty symbol (this will pass Pydantic validation but fail broker validation)
    invalid_order = Order(
        order_id="invalid_123",
        account_id="test_account",
        symbol="INVALID",  # Valid symbol for Pydantic
        asset_type=AssetType.STOCK,
        exchange=ExchangeType.NASDAQ,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,  # Limit order without price should fail
        quantity=Decimal("10"),
        time_in_force=TimeInForce.DAY
        # Missing limit_price for limit order
    )
    
    try:
        await alpaca_broker.validate_order(invalid_order)
        print("❌ Invalid order validation should have failed")
        return False
    except ValueError as e:
        print("✅ Invalid order validation correctly failed")
    
    # Test 6: Broker Manager Status
    print("\n6️⃣ Testing Broker Manager Status...")
    status = await manager.get_broker_status()
    assert 'initialized' in status
    assert 'routing_strategy' in status
    assert 'brokers' in status
    print("✅ Broker manager status retrieved successfully")
    
    # Test 7: Market Hours Check
    print("\n7️⃣ Testing Market Hours Check...")
    is_open = await alpaca_broker.is_market_open()
    print(f"Market is {'open' if is_open else 'closed'}")
    print("✅ Market hours check completed")
    
    # Test 8: Metrics and Health Check
    print("\n8️⃣ Testing Metrics and Health Check...")
    metrics = await alpaca_broker.get_metrics()
    assert 'broker' in metrics
    assert 'status' in metrics
    assert 'metrics' in metrics
    
    health = await alpaca_broker.health_check()
    assert 'broker' in health
    assert 'healthy' in health
    assert 'checks' in health
    print("✅ Metrics and health check completed")
    
    # Test 9: Callback Registration
    print("\n9️⃣ Testing Callback Registration...")
    callback_called = False
    
    async def test_callback(data):
        nonlocal callback_called
        callback_called = True
    
    alpaca_broker.register_callback('test_event', test_callback)
    await alpaca_broker._trigger_callbacks('test_event', 'test_data')
    assert callback_called
    print("✅ Callback registration and triggering works")
    
    # Test 10: Rate Limiter
    print("\n🔟 Testing Rate Limiter...")
    from src.brokers.base import RateLimiter
    rate_limiter = RateLimiter(rate=10, period=60)
    
    # Test token acquisition
    acquired = await rate_limiter.acquire()
    assert acquired
    print("✅ Rate limiter working correctly")
    
    print("\n" + "=" * 60)
    print("🎉 ALL BROKER INTEGRATION TESTS PASSED!")
    print("=" * 60)
    
    return True

async def test_with_real_credentials():
    """Test with real Alpaca credentials if available"""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key or api_key == "test_key":
        print("\n⚠️  Skipping real credentials test - no valid API keys found")
        print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables to test with real credentials")
        return True
    
    print("\n🔐 Testing with Real Alpaca Credentials...")
    
    try:
        # Create broker manager with real config
        manager = BrokerManager()
        
        configs = {
            BrokerType.ALPACA: BrokerConfig(
                name="alpaca_live",
                api_key=api_key,
                secret_key=secret_key,
                base_url="https://paper-api.alpaca.markets",
                paper_trading=True
            )
        }
        
        # Initialize broker
        await manager.initialize(configs)
        
        # Test account access
        account = await manager.get_account()
        if account:
            print(f"✅ Account connected: {account.account_id}")
            print(f"   Cash Balance: ${account.cash_balance}")
            print(f"   Buying Power: ${account.buying_power}")
        else:
            print("❌ Could not retrieve account information")
            
        # Test positions
        positions = await manager.get_positions()
        print(f"✅ Retrieved {len(positions)} positions")
        
        # Test orders
        orders = await manager.get_orders()
        print(f"✅ Retrieved {len(orders)} orders")
        
        # Get broker status
        status = await manager.get_broker_status()
        print(f"✅ Broker status: {status['brokers']}")
        
        # Cleanup
        await manager.shutdown()
        
        print("✅ Real credentials test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Real credentials test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Step 3: Broker Integration Test Suite")
    print("=" * 60)
    
    # Run basic tests
    success = asyncio.run(test_broker_integration())
    
    if success:
        # Run real credentials test if available
        asyncio.run(test_with_real_credentials())
    
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Step 3 Broker Integration Tests")
    sys.exit(0 if success else 1)

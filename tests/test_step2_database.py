"""
Comprehensive test suite for Step 2: Database Layer
"""

import pytest
import asyncio
import asyncpg
from decimal import Decimal
from datetime import datetime, timedelta
import uuid
import orjson
from faker import Faker

from src.database.connection import DatabaseManager
from src.database.models import (
    Order, Trade, Position, Strategy, TradingSignal,
    OrderStatus, OrderType, OrderSide, TimeInForce, AssetType, ExchangeType
)
from src.database.repositories.order_repository import OrderRepository

fake = Faker()

@pytest.fixture
async def db_manager():
    """Database manager fixture"""
    manager = DatabaseManager()
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.fixture
async def order_repo(db_manager):
    """Order repository fixture"""
    return OrderRepository()

class TestDatabaseConnection:
    """Test database connectivity and pooling"""
    
    @pytest.mark.asyncio
    async def test_postgres_connection(self, db_manager):
        """Test PostgreSQL connection"""
        result = await db_manager.fetchval("SELECT 1")
        assert result == 1
        
    @pytest.mark.asyncio
    async def test_timescale_connection(self, db_manager):
        """Test TimescaleDB connection"""
        async with db_manager.timescale_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
            
    @pytest.mark.asyncio
    async def test_redis_connection(self, db_manager):
        """Test Redis connection"""
        await db_manager.cache_set("test_key", "test_value", ttl=10)
        value = await db_manager.cache_get("test_key")
        assert value == "test_value"
        await db_manager.cache_delete("test_key")
        
    @pytest.mark.asyncio
    async def test_connection_pool_stats(self, db_manager):
        """Test connection pool statistics"""
        stats = await db_manager.get_pool_stats()
        
        assert 'postgres' in stats
        assert 'timescale' in stats
        assert 'redis' in stats
        assert 'mongo' in stats
        
        assert stats['postgres']['size'] >= 0
        assert stats['redis']['connected'] is True
        
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_manager):
        """Test transaction rollback"""
        try:
            async with db_manager.transaction() as conn:
                await conn.execute("CREATE TABLE test_table (id INT)")
                raise Exception("Test rollback")
        except Exception:
            pass
            
        # Table should not exist due to rollback
        async with db_manager.postgres_connection() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'test_table')"
            )
            assert result is False

class TestModels:
    """Test database models"""
    
    def test_order_model_validation(self):
        """Test Order model validation"""
        order = Order(
            order_id=str(uuid.uuid4()),
            account_id="TEST_ACCOUNT",
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.50"),
            time_in_force=TimeInForce.DAY
        )
        
        assert order.order_id
        assert order.quantity == Decimal("100")
        assert order.status == OrderStatus.PENDING
        assert order.remaining_quantity == Decimal("100")
        
    def test_order_json_serialization(self):
        """Test Order JSON serialization"""
        order = Order(
            order_id=str(uuid.uuid4()),
            account_id="TEST_ACCOUNT",
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100")
        )
        
        json_str = order.model_dump_json()
        assert json_str
        
        # Deserialize back
        order_dict = orjson.loads(json_str)
        assert order_dict['order_id'] == order.order_id
        assert float(order_dict['quantity']) == 100.0
        
    def test_position_pnl_calculation(self):
        """Test Position P&L calculations"""
        position = Position(
            position_id=str(uuid.uuid4()),
            account_id="TEST_ACCOUNT",
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            quantity=Decimal("100"),
            available_quantity=Decimal("100"),
            average_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
            realized_pnl=Decimal("0"),
            total_pnl=Decimal("500.00"),
            pnl_percentage=Decimal("3.33"),
            opened_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        assert position.unrealized_pnl == Decimal("500.00")
        assert position.pnl_percentage == Decimal("3.33")

class TestOrderRepository:
    """Test order repository operations"""
    
    @pytest.mark.asyncio
    async def test_create_order(self, order_repo):
        """Test order creation"""
        order = Order(
            order_id=f"TEST_{uuid.uuid4()}",
            account_id="TEST_ACCOUNT",
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.50"),
            time_in_force=TimeInForce.DAY,
            strategy_id="TEST_STRATEGY",
            tags=["test", "automated"],
            metadata={"source": "unit_test"}
        )
        
        created_order = await order_repo.create(order)
        assert created_order.order_id == order.order_id
        
        # Verify in database
        fetched_order = await order_repo.get_by_id(order.order_id)
        assert fetched_order
        assert fetched_order.symbol == "AAPL"
        assert fetched_order.quantity == Decimal("100")
        
    @pytest.mark.asyncio
    async def test_update_order_status(self, order_repo):
        """Test order status update"""
        # Create order
        order = Order(
            order_id=f"TEST_{uuid.uuid4()}",
            account_id="TEST_ACCOUNT",
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50")
        )
        
        await order_repo.create(order)
        
        # Update status
        success = await order_repo.update_status(
            order.order_id,
            OrderStatus.SUBMITTED,
            "Order sent to exchange"
        )
        assert success
        
        # Verify update
        updated_order = await order_repo.get_by_id(order.order_id)
        assert updated_order.status == OrderStatus.SUBMITTED
        assert updated_order.status_message == "Order sent to exchange"
        
    @pytest.mark.asyncio
    async def test_update_order_fill(self, order_repo):
        """Test order fill update"""
        # Create order
        order = Order(
            order_id=f"TEST_{uuid.uuid4()}",
            account_id="TEST_ACCOUNT",
            symbol="TSLA",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("250.00")
        )
        
        await order_repo.create(order)
        
        # Partial fill
        success = await order_repo.update_fill(
            order.order_id,
            Decimal("50"),
            Decimal("249.95")
        )
        assert success
        
        # Verify partial fill
        updated_order = await order_repo.get_by_id(order.order_id)
        assert updated_order.filled_quantity == Decimal("50")
        assert updated_order.remaining_quantity == Decimal("50")
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        
        # Complete fill
        success = await order_repo.update_fill(
            order.order_id,
            Decimal("100"),
            Decimal("249.97")
        )
        assert success
        
        # Verify complete fill
        updated_order = await order_repo.get_by_id(order.order_id)
        assert updated_order.filled_quantity == Decimal("100")
        assert updated_order.remaining_quantity == Decimal("0")
        assert updated_order.status == OrderStatus.FILLED
        
    @pytest.mark.asyncio
    async def test_cancel_order(self, order_repo):
        """Test order cancellation"""
        # Create order
        order = Order(
            order_id=f"TEST_{uuid.uuid4()}",
            account_id="TEST_ACCOUNT",
            symbol="GOOGL",
            asset_type=AssetType.STOCK,
            exchange=ExchangeType.NASDAQ,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("25"),
            limit_price=Decimal("2800.00")
        )
        
        await order_repo.create(order)
        
        # Cancel order
        success = await order_repo.cancel_order(
            order.order_id,
            "User requested cancellation"
        )
        assert success
        
        # Verify cancellation
        cancelled_order = await order_repo.get_by_id(order.order_id)
        assert cancelled_order.status == OrderStatus.CANCELLED
        assert cancelled_order.status_message == "User requested cancellation"
        assert cancelled_order.cancelled_at is not None
        
    @pytest.mark.asyncio
    async def test_get_active_orders(self, order_repo):
        """Test retrieving active orders"""
        account_id = f"TEST_ACCOUNT_{uuid.uuid4()}"
        
        # Create multiple orders with different statuses
        orders = []
        for i in range(5):
            order = Order(
                order_id=f"TEST_{uuid.uuid4()}",
                account_id=account_id,
                symbol=f"TEST{i}",
                asset_type=AssetType.STOCK,
                exchange=ExchangeType.NYSE,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("100"),
                limit_price=Decimal("100.00"),
                status=OrderStatus.PENDING if i < 3 else OrderStatus.FILLED
            )
            orders.append(order)
            await order_repo.create(order)
            
        # Get active orders
        active_orders = await order_repo.get_active_orders(account_id)
        
        # Should only return pending orders
        assert len(active_orders) == 3
        for order in active_orders:
            assert order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            
    @pytest.mark.asyncio
    async def test_get_order_stats(self, order_repo):
        """Test order statistics"""
        account_id = f"TEST_ACCOUNT_{uuid.uuid4()}"
        
        # Create test orders
        for i in range(10):
            order = Order(
                order_id=f"TEST_{uuid.uuid4()}",
                account_id=account_id,
                symbol="AAPL" if i % 2 == 0 else "GOOGL",
                asset_type=AssetType.STOCK,
                exchange=ExchangeType.NASDAQ,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                status=OrderStatus.FILLED if i < 7 else OrderStatus.CANCELLED,
                commission=Decimal("1.00"),
                fees=Decimal("0.50")
            )
            await order_repo.create(order)
            
        # Get statistics
        stats = await order_repo.get_order_stats(account_id)
        
        assert stats['total_orders'] == 10
        assert stats['filled_orders'] == 7
        assert stats['cancelled_orders'] == 3
        assert stats['fill_rate'] == 0.7
        assert stats['total_costs'] == 15.0  # 10 * (1.00 + 0.50)
        assert stats['unique_symbols'] == 2

class TestPerformance:
    """Test database performance"""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_manager):
        """Test bulk insert performance"""
        orders = []
        for i in range(1000):
            orders.append((
                f"PERF_TEST_{uuid.uuid4()}",
                "PERF_ACCOUNT",
                "AAPL",
                "stock",
                "nasdaq",
                "buy",
                "market",
                100.0,
                0.0,
                100.0,
                "pending"
            ))
            
        start_time = asyncio.get_event_loop().time()
        
        async with db_manager.postgres_connection() as conn:
            await conn.executemany(
                """
                INSERT INTO orders (
                    order_id, account_id, symbol, asset_type, exchange,
                    side, order_type, quantity, filled_quantity,
                    remaining_quantity, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                orders
            )
            
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should insert 1000 orders in less than 1 second
        assert elapsed < 1.0
        print(f"Bulk insert of 1000 orders took {elapsed:.3f} seconds")
        
    @pytest.mark.asyncio
    async def test_query_performance(self, db_manager):
        """Test query performance"""
        # Run a complex query
        query = """
            SELECT o.symbol, COUNT(*) as order_count,
                   AVG(o.quantity) as avg_quantity,
                   SUM(o.commission + o.fees) as total_costs
            FROM orders o
            WHERE o.created_at >= NOW() - INTERVAL '30 days'
            GROUP BY o.symbol
            HAVING COUNT(*) > 5
            ORDER BY order_count DESC
            LIMIT 100
        """
        
        start_time = asyncio.get_event_loop().time()
        results = await db_manager.fetch(query)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Complex query should complete in less than 100ms
        assert elapsed < 0.1
        print(f"Complex aggregation query took {elapsed:.3f} seconds")
        
    @pytest.mark.asyncio
    async def test_cache_performance(self, db_manager):
        """Test cache performance"""
        # Set 1000 cache entries
        start_time = asyncio.get_event_loop().time()
        
        for i in range(1000):
            await db_manager.cache_set(f"perf_test_{i}", f"value_{i}", ttl=60)
            
        set_elapsed = asyncio.get_event_loop().time() - start_time
        
        # Get 1000 cache entries
        start_time = asyncio.get_event_loop().time()
        
        for i in range(1000):
            value = await db_manager.cache_get(f"perf_test_{i}")
            assert value == f"value_{i}"
            
        get_elapsed = asyncio.get_event_loop().time() - start_time
        
        # Clean up
        for i in range(1000):
            await db_manager.cache_delete(f"perf_test_{i}")
            
        # Cache operations should be very fast
        assert set_elapsed < 0.5  # 1000 sets in < 500ms
        assert get_elapsed < 0.3  # 1000 gets in < 300ms
        
        print(f"Cache set (1000): {set_elapsed:.3f}s, get (1000): {get_elapsed:.3f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
High-performance order repository with caching
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
import orjson
from ..models import Order, OrderStatus, OrderType, OrderSide
from ..connection import db_manager
import structlog

logger = structlog.get_logger()

class OrderRepository:
    """
    Order repository with caching and optimized queries
    """
    
    def __init__(self):
        self.db = db_manager
        self.cache_ttl = 300  # 5 minutes
        
    async def create(self, order: Order) -> Order:
        """Create new order with caching"""
        query = """
            INSERT INTO orders (
                order_id, client_order_id, parent_order_id, account_id,
                symbol, asset_type, exchange, side, order_type,
                quantity, filled_quantity, remaining_quantity,
                limit_price, stop_price, average_fill_price,
                time_in_force, expire_time, status, status_message,
                max_slippage, position_size_pct, stop_loss, take_profit,
                commission, fees, slippage, strategy_id, signal_id,
                tags, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30
            ) RETURNING *
        """
        
        # Execute insert
        result = await self.db.fetchrow(
            query,
            order.order_id,
            order.client_order_id,
            order.parent_order_id,
            order.account_id,
            order.symbol,
            order.asset_type.value,
            order.exchange.value,
            order.side.value,
            order.order_type.value,
            float(order.quantity),
            float(order.filled_quantity),
            float(order.remaining_quantity),
            float(order.limit_price) if order.limit_price else None,
            float(order.stop_price) if order.stop_price else None,
            float(order.average_fill_price) if order.average_fill_price else None,
            order.time_in_force.value,
            order.expire_time,
            order.status.value,
            order.status_message,
            float(order.max_slippage) if order.max_slippage else None,
            float(order.position_size_pct) if order.position_size_pct else None,
            float(order.stop_loss) if order.stop_loss else None,
            float(order.take_profit) if order.take_profit else None,
            float(order.commission),
            float(order.fees),
            float(order.slippage),
            order.strategy_id,
            order.signal_id,
            order.tags,
            orjson.dumps(order.metadata).decode() if order.metadata else None
        )
        
        # Cache the order
        await self._cache_order(order)
        
        # Log creation
        logger.info(
            "Order created",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=float(order.quantity)
        )
        
        return order
        
    async def get_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID with caching"""
        # Try cache first
        cached = await self.db.cache_get(f"order:{order_id}")
        if cached:
            return Order.model_validate_json(cached)
            
        # Query database
        query = "SELECT * FROM orders WHERE order_id = $1"
        result = await self.db.fetchrow(query, order_id)
        
        if result:
            order = self._record_to_order(result)
            await self._cache_order(order)
            return order
            
        return None
        
    async def get_active_orders(self, account_id: str = None) -> List[Order]:
        """Get all active orders"""
        query = """
            SELECT * FROM orders
            WHERE status IN ('pending', 'submitted', 'partially_filled')
        """
        params = []
        
        if account_id:
            query += " AND account_id = $1"
            params.append(account_id)
            
        query += " ORDER BY created_at DESC"
        
        results = await self.db.fetch(query, *params)
        return [self._record_to_order(r) for r in results]
        
    async def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        message: str = None
    ) -> bool:
        """Update order status"""
        query = """
            UPDATE orders
            SET status = $2, status_message = $3, updated_at = NOW()
            WHERE order_id = $1
            RETURNING *
        """
        
        result = await self.db.fetchrow(query, order_id, status.value, message)
        
        if result:
            # Invalidate cache
            await self.db.cache_delete(f"order:{order_id}")
            
            logger.info(
                "Order status updated",
                order_id=order_id,
                status=status.value,
                message=message
            )
            return True
            
        return False
        
    async def update_fill(
        self,
        order_id: str,
        filled_quantity: Decimal,
        average_fill_price: Decimal
    ) -> bool:
        """Update order fill information"""
        query = """
            UPDATE orders
            SET filled_quantity = $2,
                remaining_quantity = quantity - $2,
                average_fill_price = $3,
                status = CASE
                    WHEN $2 >= quantity THEN 'filled'
                    WHEN $2 > 0 THEN 'partially_filled'
                    ELSE status
                END,
                filled_at = CASE
                    WHEN $2 >= quantity THEN NOW()
                    ELSE filled_at
                END,
                updated_at = NOW()
            WHERE order_id = $1
            RETURNING *
        """
        
        result = await self.db.fetchrow(
            query,
            order_id,
            float(filled_quantity),
            float(average_fill_price)
        )
        
        if result:
            # Invalidate cache
            await self.db.cache_delete(f"order:{order_id}")
            
            logger.info(
                "Order fill updated",
                order_id=order_id,
                filled_quantity=float(filled_quantity),
                average_fill_price=float(average_fill_price)
            )
            return True
            
        return False
        
    async def cancel_order(self, order_id: str, reason: str = None) -> bool:
        """Cancel an order"""
        query = """
            UPDATE orders
            SET status = 'cancelled',
                status_message = $2,
                cancelled_at = NOW(),
                updated_at = NOW()
            WHERE order_id = $1
            AND status IN ('pending', 'submitted', 'partially_filled')
            RETURNING *
        """
        
        result = await self.db.fetchrow(query, order_id, reason)
        
        if result:
            # Invalidate cache
            await self.db.cache_delete(f"order:{order_id}")
            
            logger.info(
                "Order cancelled",
                order_id=order_id,
                reason=reason
            )
            return True
            
        return False
        
    async def get_orders_by_strategy(
        self,
        strategy_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Order]:
        """Get orders for a specific strategy"""
        query = """
            SELECT * FROM orders
            WHERE strategy_id = $1
        """
        params = [strategy_id]
        
        if start_date:
            query += f" AND created_at >= ${len(params) + 1}"
            params.append(start_date)
            
        if end_date:
            query += f" AND created_at <= ${len(params) + 1}"
            params.append(end_date)
            
        query += " ORDER BY created_at DESC"
        
        results = await self.db.fetch(query, *params)
        return [self._record_to_order(r) for r in results]
        
    async def get_recent_orders(
        self,
        account_id: str,
        limit: int = 100
    ) -> List[Order]:
        """Get recent orders for an account"""
        query = """
            SELECT * FROM orders
            WHERE account_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        
        results = await self.db.fetch(query, account_id, limit)
        return [self._record_to_order(r) for r in results]
        
    async def get_order_stats(self, account_id: str) -> Dict[str, Any]:
        """Get order statistics for an account"""
        query = """
            SELECT
                COUNT(*) AS total_orders,
                COUNT(*) FILTER (WHERE status = 'filled') AS filled_orders,
                COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled_orders,
                COUNT(*) FILTER (WHERE status IN ('pending', 'submitted')) AS pending_orders,
                AVG(EXTRACT(EPOCH FROM (filled_at - submitted_at))) FILTER (WHERE filled_at IS NOT NULL) AS avg_fill_time_seconds,
                SUM(commission + fees) AS total_costs,
                COUNT(DISTINCT symbol) AS unique_symbols,
                COUNT(DISTINCT DATE(created_at)) AS trading_days
            FROM orders
            WHERE account_id = $1
            AND created_at >= NOW() - INTERVAL '30 days'
        """
        
        result = await self.db.fetchrow(query, account_id)
        
        return {
            'total_orders': result['total_orders'],
            'filled_orders': result['filled_orders'],
            'cancelled_orders': result['cancelled_orders'],
            'pending_orders': result['pending_orders'],
            'avg_fill_time_seconds': float(result['avg_fill_time_seconds']) if result['avg_fill_time_seconds'] else None,
            'total_costs': float(result['total_costs']) if result['total_costs'] else 0,
            'unique_symbols': result['unique_symbols'],
            'trading_days': result['trading_days'],
            'fill_rate': result['filled_orders'] / result['total_orders'] if result['total_orders'] > 0 else 0,
        }
        
    def _record_to_order(self, record) -> Order:
        """Convert database record to Order model"""
        return Order(
            id=record['id'],
            order_id=record['order_id'],
            client_order_id=record['client_order_id'],
            parent_order_id=record['parent_order_id'],
            account_id=record['account_id'],
            symbol=record['symbol'],
            asset_type=record['asset_type'],
            exchange=record['exchange'],
            side=record['side'],
            order_type=record['order_type'],
            quantity=Decimal(str(record['quantity'])),
            filled_quantity=Decimal(str(record['filled_quantity'])),
            remaining_quantity=Decimal(str(record['remaining_quantity'])),
            limit_price=Decimal(str(record['limit_price'])) if record['limit_price'] else None,
            stop_price=Decimal(str(record['stop_price'])) if record['stop_price'] else None,
            average_fill_price=Decimal(str(record['average_fill_price'])) if record['average_fill_price'] else None,
            time_in_force=record['time_in_force'],
            expire_time=record['expire_time'],
            submitted_at=record['submitted_at'],
            filled_at=record['filled_at'],
            cancelled_at=record['cancelled_at'],
            status=record['status'],
            status_message=record['status_message'],
            max_slippage=Decimal(str(record['max_slippage'])) if record['max_slippage'] else None,
            position_size_pct=Decimal(str(record['position_size_pct'])) if record['position_size_pct'] else None,
            stop_loss=Decimal(str(record['stop_loss'])) if record['stop_loss'] else None,
            take_profit=Decimal(str(record['take_profit'])) if record['take_profit'] else None,
            commission=Decimal(str(record['commission'])),
            fees=Decimal(str(record['fees'])),
            slippage=Decimal(str(record['slippage'])),
            strategy_id=record['strategy_id'],
            signal_id=record['signal_id'],
            tags=record['tags'] or [],
            metadata=orjson.loads(record['metadata']) if record['metadata'] else {},
            created_at=record['created_at'],
            updated_at=record['updated_at'],
        )
        
    async def _cache_order(self, order: Order):
        """Cache order data"""
        await self.db.cache_set(
            f"order:{order.order_id}",
            order.model_dump_json(),
            ttl=self.cache_ttl
        )

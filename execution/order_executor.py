"""
OMNI ALPHA 5.0 - ORDER EXECUTION SYSTEM
=======================================
Handles actual order placement and management with Alpaca integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Try to import Alpaca API
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, TrailingStopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca API not available, using simulation mode")
    ALPACA_AVAILABLE = False

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class ExecutionType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

@dataclass
class OrderRequest:
    symbol: str
    quantity: int
    side: str  # 'BUY' or 'SELL'
    order_type: ExecutionType = ExecutionType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    
@dataclass
class ExecutionResult:
    order_id: Optional[str]
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

class OrderExecutor:
    """Handles actual order placement and management"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.active_orders = {}
        self.order_history = []
        self.simulation_mode = not ALPACA_AVAILABLE or config.get('SIMULATION_MODE', False)
        
        # Initialize Alpaca client if available
        if ALPACA_AVAILABLE and not self.simulation_mode:
            try:
                self.client = TradingClient(
                    config.get('ALPACA_API_KEY'),
                    config.get('ALPACA_SECRET_KEY'),
                    paper=config.get('TRADING_MODE', 'paper') == 'paper'
                )
                # Test connection
                account = self.client.get_account()
                logger.info(f"Alpaca client initialized - Account: ${float(account.cash):.2f} cash")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
                self.simulation_mode = True
                self.client = None
        
        if self.simulation_mode:
            logger.info("Running in simulation mode - no real orders will be placed")
    
    async def execute_signal(self, signal: Dict, risk_params: Dict, account: Dict = None) -> ExecutionResult:
        """Execute trading signal with comprehensive order management"""
        
        try:
            # Validate signal
            if signal.get('confidence', 0) < 0.6:
                return ExecutionResult(
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    error_message="Signal confidence too low"
                )
            
            # Prepare order request
            order_request = self._prepare_order_request(signal, risk_params)
            
            if not order_request:
                return ExecutionResult(
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    error_message="Failed to prepare order request"
                )
            
            # Execute the order
            if self.simulation_mode:
                result = await self._simulate_order_execution(order_request, signal)
            else:
                result = await self._execute_real_order(order_request)
            
            # Set up bracket orders if main order was filled
            if result.status == OrderStatus.FILLED and result.order_id:
                await self._setup_bracket_orders(
                    order_request.symbol,
                    result.filled_price or order_request.limit_price or 0,
                    risk_params.get('quantity', order_request.quantity),
                    risk_params.get('stop_loss'),
                    risk_params.get('take_profit')
                )
            
            # Record execution
            self.order_history.append({
                'timestamp': datetime.now(),
                'symbol': order_request.symbol,
                'signal': signal,
                'risk_params': risk_params,
                'result': result,
                'order_request': order_request
            })
            
            logger.info(f"Order execution result: {order_request.symbol} {order_request.side} - {result.status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return ExecutionResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                error_message=f"Execution error: {e}"
            )
    
    def _prepare_order_request(self, signal: Dict, risk_params: Dict) -> Optional[OrderRequest]:
        """Prepare order request from signal and risk parameters"""
        
        try:
            symbol = signal.get('symbol')
            signal_type = signal.get('signal', 'HOLD')
            quantity = risk_params.get('quantity', 0)
            
            if not symbol or signal_type == 'HOLD' or quantity <= 0:
                return None
            
            # Determine order side
            side = 'BUY' if signal_type in ['BUY', 'WEAK_BUY'] else 'SELL'
            
            # Determine execution type based on signal confidence and market conditions
            confidence = signal.get('confidence', 0.6)
            
            if confidence >= 0.8:
                # High confidence - use market order for immediate execution
                order_type = ExecutionType.MARKET
                limit_price = None
            else:
                # Lower confidence - use limit order for better price
                order_type = ExecutionType.LIMIT
                entry_price = signal.get('entry_price', 0)
                
                # Set limit price with small buffer for better fill probability
                if side == 'BUY':
                    limit_price = entry_price * 1.001  # 0.1% above current price
                else:
                    limit_price = entry_price * 0.999  # 0.1% below current price
            
            return OrderRequest(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force="DAY"
            )
            
        except Exception as e:
            logger.error(f"Order preparation error: {e}")
            return None
    
    async def _execute_real_order(self, order_request: OrderRequest) -> ExecutionResult:
        """Execute real order through Alpaca"""
        
        try:
            if not self.client:
                raise Exception("Alpaca client not available")
            
            # Prepare Alpaca order request
            if order_request.order_type == ExecutionType.MARKET:
                alpaca_request = MarketOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=OrderSide.BUY if order_request.side == 'BUY' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
            elif order_request.order_type == ExecutionType.LIMIT:
                alpaca_request = LimitOrderRequest(
                    symbol=order_request.symbol,
                    qty=order_request.quantity,
                    side=OrderSide.BUY if order_request.side == 'BUY' else OrderSide.SELL,
                    limit_price=order_request.limit_price,
                    time_in_force=TimeInForce.DAY
                )
            else:
                raise Exception(f"Unsupported order type: {order_request.order_type}")
            
            # Submit order
            order = self.client.submit_order(alpaca_request)
            
            # Wait for order to be processed (with timeout)
            max_wait_time = 30  # 30 seconds
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < max_wait_time:
                try:
                    # Get updated order status
                    updated_order = self.client.get_order_by_id(order.id)
                    
                    if updated_order.status in ['filled', 'partially_filled']:
                        self.active_orders[order.id] = updated_order
                        
                        return ExecutionResult(
                            order_id=str(order.id),
                            status=OrderStatus.FILLED if updated_order.status == 'filled' else OrderStatus.PARTIALLY_FILLED,
                            filled_quantity=int(updated_order.filled_qty or 0),
                            filled_price=float(updated_order.filled_avg_price) if updated_order.filled_avg_price else None,
                            commission=0.0,  # Alpaca is commission-free
                            timestamp=datetime.now()
                        )
                    
                    elif updated_order.status in ['cancelled', 'rejected', 'expired']:
                        return ExecutionResult(
                            order_id=str(order.id),
                            status=OrderStatus.CANCELLED if updated_order.status == 'cancelled' else 
                                   OrderStatus.REJECTED if updated_order.status == 'rejected' else OrderStatus.EXPIRED,
                            error_message=f"Order {updated_order.status}"
                        )
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error checking order status: {e}")
                    break
            
            # If we get here, order is still pending
            self.active_orders[order.id] = order
            return ExecutionResult(
                order_id=str(order.id),
                status=OrderStatus.SUBMITTED,
                timestamp=datetime.now()
            )
            
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return ExecutionResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                error_message=f"API error: {e}"
            )
        except Exception as e:
            logger.error(f"Real order execution error: {e}")
            return ExecutionResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                error_message=f"Execution error: {e}"
            )
    
    async def _simulate_order_execution(self, order_request: OrderRequest, signal: Dict) -> ExecutionResult:
        """Simulate order execution for testing"""
        
        try:
            # Simulate order processing delay
            await asyncio.sleep(0.1)
            
            # Simulate fill based on order type and market conditions
            fill_probability = 0.95  # 95% fill rate in simulation
            
            if order_request.order_type == ExecutionType.MARKET:
                fill_probability = 0.98  # Higher fill rate for market orders
            
            # Random fill simulation
            import random
            if random.random() < fill_probability:
                # Simulate filled order
                entry_price = signal.get('entry_price', 100.0)
                
                # Add small random slippage
                slippage = random.uniform(-0.002, 0.002)  # ±0.2% slippage
                filled_price = entry_price * (1 + slippage)
                
                return ExecutionResult(
                    order_id=f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                    status=OrderStatus.FILLED,
                    filled_quantity=order_request.quantity,
                    filled_price=round(filled_price, 2),
                    commission=0.0,
                    timestamp=datetime.now()
                )
            else:
                # Simulate rejected order
                return ExecutionResult(
                    order_id=None,
                    status=OrderStatus.REJECTED,
                    error_message="Simulated market rejection"
                )
                
        except Exception as e:
            logger.error(f"Order simulation error: {e}")
            return ExecutionResult(
                order_id=None,
                status=OrderStatus.REJECTED,
                error_message=f"Simulation error: {e}"
            )
    
    async def _setup_bracket_orders(self, symbol: str, entry_price: float, quantity: int, 
                                  stop_loss: Optional[float], take_profit: Optional[float]):
        """Create bracket orders for risk management"""
        
        try:
            if not stop_loss and not take_profit:
                logger.info(f"No bracket orders needed for {symbol}")
                return
            
            if self.simulation_mode:
                logger.info(f"Simulated bracket orders for {symbol}: SL={stop_loss}, TP={take_profit}")
                return
            
            if not self.client:
                logger.warning("Cannot create bracket orders - no Alpaca client")
                return
            
            # Create stop loss order
            if stop_loss:
                try:
                    stop_order = StopOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=OrderSide.SELL,  # Assuming we're long
                        stop_price=stop_loss,
                        time_in_force=TimeInForce.GTC
                    )
                    
                    stop_result = self.client.submit_order(stop_order)
                    self.active_orders[stop_result.id] = stop_result
                    logger.info(f"Stop loss order placed for {symbol} at ${stop_loss:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to place stop loss order: {e}")
            
            # Create take profit order
            if take_profit:
                try:
                    profit_order = LimitOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=OrderSide.SELL,  # Assuming we're long
                        limit_price=take_profit,
                        time_in_force=TimeInForce.GTC
                    )
                    
                    profit_result = self.client.submit_order(profit_order)
                    self.active_orders[profit_result.id] = profit_result
                    logger.info(f"Take profit order placed for {symbol} at ${take_profit:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed to place take profit order: {e}")
                    
        except Exception as e:
            logger.error(f"Bracket order setup error: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        
        try:
            if self.simulation_mode:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                    logger.info(f"Simulated order cancellation: {order_id}")
                    return True
                return False
            
            if not self.client:
                return False
            
            self.client.cancel_order_by_id(order_id)
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders, optionally filtered by symbol"""
        
        cancelled_count = 0
        
        try:
            if self.simulation_mode:
                orders_to_cancel = list(self.active_orders.keys())
                for order_id in orders_to_cancel:
                    if await self.cancel_order(order_id):
                        cancelled_count += 1
                return cancelled_count
            
            if not self.client:
                return 0
            
            # Get all open orders
            open_orders = self.client.get_orders(status='open')
            
            for order in open_orders:
                if symbol is None or order.symbol == symbol:
                    try:
                        await self.cancel_order(str(order.id))
                        cancelled_count += 1
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order.id}: {e}")
            
            logger.info(f"Cancelled {cancelled_count} orders" + (f" for {symbol}" if symbol else ""))
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Bulk order cancellation error: {e}")
            return cancelled_count
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order"""
        
        try:
            if self.simulation_mode:
                if order_id in self.active_orders:
                    return {
                        'order_id': order_id,
                        'status': 'filled',  # Simulate filled
                        'symbol': 'SIM',
                        'quantity': 100
                    }
                return None
            
            if not self.client:
                return None
            
            order = self.client.get_order_by_id(order_id)
            
            return {
                'order_id': str(order.id),
                'status': order.status,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'filled_quantity': int(order.filled_qty or 0),
                'filled_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            }
            
        except Exception as e:
            logger.error(f"Order status check error: {e}")
            return None
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active orders"""
        
        try:
            if self.simulation_mode:
                return [
                    {
                        'order_id': order_id,
                        'status': 'open',
                        'symbol': 'SIM'
                    }
                    for order_id in self.active_orders.keys()
                ]
            
            if not self.client:
                return []
            
            open_orders = self.client.get_orders(status='open')
            
            return [
                {
                    'order_id': str(order.id),
                    'status': order.status,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': int(order.qty),
                    'order_type': order.order_type,
                    'created_at': order.created_at
                }
                for order in open_orders
            ]
            
        except Exception as e:
            logger.error(f"Active orders retrieval error: {e}")
            return []
    
    def get_execution_statistics(self) -> Dict:
        """Get execution performance statistics"""
        
        try:
            if not self.order_history:
                return {
                    'total_orders': 0,
                    'fill_rate': 0.0,
                    'average_slippage': 0.0,
                    'execution_time': 0.0
                }
            
            total_orders = len(self.order_history)
            filled_orders = [h for h in self.order_history if h['result'].status == OrderStatus.FILLED]
            fill_rate = len(filled_orders) / total_orders
            
            # Calculate average slippage (simplified)
            slippages = []
            for history in filled_orders:
                signal_price = history['signal'].get('entry_price', 0)
                filled_price = history['result'].filled_price
                if signal_price and filled_price:
                    slippage = abs(filled_price - signal_price) / signal_price
                    slippages.append(slippage)
            
            avg_slippage = np.mean(slippages) if slippages else 0.0
            
            return {
                'total_orders': total_orders,
                'filled_orders': len(filled_orders),
                'fill_rate': fill_rate,
                'average_slippage': avg_slippage,
                'simulation_mode': self.simulation_mode,
                'alpaca_available': ALPACA_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation error: {e}")
            return {'error': str(e)}
    
    async def emergency_close_all_positions(self) -> Dict:
        """Emergency procedure to close all positions"""
        
        logger.critical("🚨 EMERGENCY POSITION CLOSURE INITIATED")
        
        try:
            if self.simulation_mode:
                logger.info("Simulated emergency closure - all positions closed")
                return {
                    'success': True,
                    'positions_closed': 0,
                    'orders_cancelled': len(self.active_orders),
                    'simulation_mode': True
                }
            
            if not self.client:
                return {'success': False, 'error': 'No trading client available'}
            
            # Cancel all open orders first
            cancelled_orders = await self.cancel_all_orders()
            
            # Close all positions
            positions = self.client.get_all_positions()
            positions_closed = 0
            
            for position in positions:
                try:
                    # Create market order to close position
                    close_order = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=abs(int(position.qty)),
                        side=OrderSide.SELL if int(position.qty) > 0 else OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    self.client.submit_order(close_order)
                    positions_closed += 1
                    logger.info(f"Emergency close order placed for {position.symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to close position {position.symbol}: {e}")
            
            return {
                'success': True,
                'positions_closed': positions_closed,
                'orders_cancelled': cancelled_orders,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emergency closure error: {e}")
            return {'success': False, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    import asyncio
    
    config = {
        'ALPACA_API_KEY': 'your_key',
        'ALPACA_SECRET_KEY': 'your_secret',
        'TRADING_MODE': 'paper',
        'SIMULATION_MODE': True
    }
    
    executor = OrderExecutor(config)
    
    # Test signal
    signal = {
        'symbol': 'AAPL',
        'signal': 'BUY',
        'confidence': 0.75,
        'entry_price': 150.0
    }
    
    risk_params = {
        'quantity': 10,
        'stop_loss': 147.0,
        'take_profit': 156.0
    }
    
    async def test():
        result = await executor.execute_signal(signal, risk_params)
        print(f"Execution result: {result}")
    
    asyncio.run(test())

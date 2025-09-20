#!/usr/bin/env python3
"""
OMNI ALPHA 5.0 - SIMPLE TRADING BOT
===================================
Standalone trading bot with all essential features
No complex dependencies - just core trading functionality
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('trading_bot_config.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTradingStrategies:
    """Simple but effective trading strategies"""
    
    def __init__(self):
        self.min_confidence = 0.6
    
    def moving_average_crossover(self, data: pd.DataFrame) -> Dict:
        """MA crossover strategy"""
        if len(data) < 50:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        if prev['SMA_20'] <= prev['SMA_50'] and last['SMA_20'] > last['SMA_50']:
            return {'signal': 'BUY', 'confidence': 0.7, 'reason': 'Golden Cross'}
        elif prev['SMA_20'] >= prev['SMA_50'] and last['SMA_20'] < last['SMA_50']:
            return {'signal': 'SELL', 'confidence': 0.7, 'reason': 'Death Cross'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'No MA signal'}
    
    def rsi_strategy(self, data: pd.DataFrame) -> Dict:
        """RSI oversold/overbought strategy"""
        if len(data) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        last_rsi = data['RSI'].iloc[-1]
        
        if last_rsi < 30:
            return {'signal': 'BUY', 'confidence': 0.65, 'reason': f'RSI oversold ({last_rsi:.1f})'}
        elif last_rsi > 70:
            return {'signal': 'SELL', 'confidence': 0.65, 'reason': f'RSI overbought ({last_rsi:.1f})'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'RSI neutral ({last_rsi:.1f})'}
    
    def bollinger_bands(self, data: pd.DataFrame) -> Dict:
        """Bollinger Bands strategy"""
        if len(data) < 25:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        data['SMA'] = data['Close'].rolling(20).mean()
        data['STD'] = data['Close'].rolling(20).std()
        data['Upper'] = data['SMA'] + (data['STD'] * 2)
        data['Lower'] = data['SMA'] - (data['STD'] * 2)
        
        last = data.iloc[-1]
        
        if last['Close'] <= last['Lower']:
            return {'signal': 'BUY', 'confidence': 0.68, 'reason': 'Price at lower band'}
        elif last['Close'] >= last['Upper']:
            return {'signal': 'SELL', 'confidence': 0.68, 'reason': 'Price at upper band'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Price within bands'}
    
    def volume_breakout(self, data: pd.DataFrame) -> Dict:
        """Volume breakout strategy"""
        if len(data) < 25:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['Price_Change'] = data['Close'].pct_change()
        
        last = data.iloc[-1]
        
        if last['Volume'] > last['Volume_MA'] * 2 and abs(last['Price_Change']) > 0.02:
            if last['Price_Change'] > 0:
                return {'signal': 'BUY', 'confidence': 0.72, 'reason': 'Volume breakout up'}
            else:
                return {'signal': 'SELL', 'confidence': 0.72, 'reason': 'Volume breakout down'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Normal volume'}
    
    def combine_signals(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Combine all strategies"""
        signals = [
            self.moving_average_crossover(data),
            self.rsi_strategy(data),
            self.bollinger_bands(data),
            self.volume_breakout(data)
        ]
        
        # Count signals
        buy_signals = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals if s['signal'] == 'SELL')
        
        # Average confidence
        avg_confidence = np.mean([s['confidence'] for s in signals])
        
        if buy_signals >= 3:
            return {
                'signal': 'BUY',
                'confidence': min(0.85, avg_confidence * 1.2),
                'reason': f'{buy_signals}/4 strategies bullish',
                'symbol': symbol,
                'timestamp': datetime.now()
            }
        elif sell_signals >= 3:
            return {
                'signal': 'SELL',
                'confidence': min(0.85, avg_confidence * 1.2),
                'reason': f'{sell_signals}/4 strategies bearish',
                'symbol': symbol,
                'timestamp': datetime.now()
            }
        
        return {
            'signal': 'HOLD', 
            'confidence': 0.5, 
            'reason': 'Mixed signals',
            'symbol': symbol,
            'timestamp': datetime.now()
        }

class SimpleRiskManager:
    """Simple but effective risk management"""
    
    def __init__(self, config):
        self.max_daily_loss = config.get('MAX_DAILY_LOSS', 1000)
        self.max_position_size = config.get('MAX_POSITION_SIZE', 10000)
        self.max_positions = config.get('MAX_POSITIONS', 5)
        self.daily_pnl = 0
        self.daily_trades = 0
        self.circuit_breaker_active = False
    
    def check_risk(self, order: Dict, account: Dict, positions: Dict) -> Tuple[bool, str]:
        """Check if trade is allowed"""
        
        # Circuit breaker check
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"
        
        # Daily loss check
        if self.daily_pnl <= -self.max_daily_loss:
            self.circuit_breaker_active = True
            return False, "Daily loss limit reached"
        
        # Position count check
        if len(positions) >= self.max_positions:
            return False, f"Max {self.max_positions} positions allowed"
        
        # Position size check
        order_value = order['quantity'] * order['price']
        if order_value > self.max_position_size:
            return False, f"Position size ${order_value:.2f} too large"
        
        # Buying power check
        if order_value > account.get('buying_power', 0):
            return False, "Insufficient buying power"
        
        return True, "Risk checks passed"
    
    def calculate_position_size(self, signal: Dict, account: Dict, price: float) -> int:
        """Calculate position size using Kelly Criterion"""
        portfolio_value = account.get('portfolio_value', 100000)
        confidence = signal.get('confidence', 0.6)
        
        # Simple Kelly approximation
        win_rate = confidence
        avg_win = 0.03  # 3% average win
        avg_loss = 0.02  # 2% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Conservative Kelly
        
        position_value = portfolio_value * kelly_fraction
        shares = int(min(position_value, self.max_position_size) / price)
        
        return max(1, shares)
    
    def update_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
        if pnl != 0:  # Only count actual trades
            self.daily_trades += 1

class SimpleOrderExecutor:
    """Simple order executor with simulation mode"""
    
    def __init__(self, config):
        self.config = config
        self.simulation_mode = config.get('SIMULATION_MODE', True)
        self.orders = {}
        self.order_counter = 0
        
        # Try to initialize Alpaca client for real trading
        if not self.simulation_mode:
            try:
                import alpaca_trade_api as tradeapi
                self.api = tradeapi.REST(
                    config.get('ALPACA_API_KEY'),
                    config.get('ALPACA_SECRET_KEY'),
                    'https://paper-api.alpaca.markets' if config.get('TRADING_MODE') == 'paper' else 'https://api.alpaca.markets'
                )
                # Test connection
                account = self.api.get_account()
                logger.info(f"Alpaca connected: ${account.cash} available")
            except Exception as e:
                logger.warning(f"Alpaca connection failed: {e}, using simulation mode")
                self.simulation_mode = True
                self.api = None
        else:
            self.api = None
            logger.info("Running in simulation mode")
    
    async def execute_order(self, signal: Dict, quantity: int, stop_loss: float = None, take_profit: float = None) -> Dict:
        """Execute order (real or simulated)"""
        
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:04d}"
        
        if self.simulation_mode:
            # Simulate order execution
            await asyncio.sleep(0.1)  # Simulate execution delay
            
            # Simulate 95% fill rate
            import random
            if random.random() < 0.95:
                # Add small slippage
                slippage = random.uniform(-0.001, 0.001)
                fill_price = signal.get('entry_price', 0) * (1 + slippage)
                
                result = {
                    'order_id': order_id,
                    'status': 'FILLED',
                    'symbol': signal['symbol'],
                    'quantity': quantity,
                    'fill_price': round(fill_price, 2),
                    'timestamp': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                self.orders[order_id] = result
                logger.info(f"✅ SIMULATED: {signal['symbol']} {signal['signal']} {quantity} @ ${fill_price:.2f}")
                return result
            else:
                logger.warning(f"❌ SIMULATED: Order rejected for {signal['symbol']}")
                return {'status': 'REJECTED', 'reason': 'Simulated rejection'}
        
        else:
            # Real Alpaca execution
            try:
                if signal['signal'] == 'BUY':
                    order = self.api.submit_order(
                        symbol=signal['symbol'],
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                else:
                    order = self.api.submit_order(
                        symbol=signal['symbol'],
                        qty=quantity,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                
                # Wait for fill
                await asyncio.sleep(2)
                updated_order = self.api.get_order(order.id)
                
                if updated_order.status == 'filled':
                    result = {
                        'order_id': str(order.id),
                        'status': 'FILLED',
                        'symbol': signal['symbol'],
                        'quantity': int(updated_order.filled_qty),
                        'fill_price': float(updated_order.filled_avg_price),
                        'timestamp': datetime.now()
                    }
                    
                    self.orders[order.id] = result
                    logger.info(f"✅ REAL: {signal['symbol']} {signal['signal']} {quantity} @ ${result['fill_price']:.2f}")
                    return result
                
                return {'status': 'PENDING', 'order_id': str(order.id)}
                
            except Exception as e:
                logger.error(f"Real order execution failed: {e}")
                return {'status': 'FAILED', 'error': str(e)}

class SimpleTradingBot:
    """Simple but complete trading bot"""
    
    def __init__(self):
        self.config = self._load_config()
        self.strategies = SimpleTradingStrategies()
        self.risk_manager = SimpleRiskManager(self.config)
        self.executor = SimpleOrderExecutor(self.config)
        
        # Trading state
        self.positions = {}
        self.account = {'portfolio_value': 100000, 'buying_power': 50000, 'cash': 50000}
        self.daily_stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
        self.running = False
        self.start_time = datetime.now()
        
        # Market data cache
        self.data_cache = {}
        self.cache_time = {}
    
    def _load_config(self) -> Dict:
        """Load configuration from environment"""
        return {
            'SCAN_SYMBOLS': os.getenv('SCAN_SYMBOLS', 'AAPL,MSFT,GOOGL,TSLA,SPY').split(','),
            'SCAN_INTERVAL': int(os.getenv('SCAN_INTERVAL', 60)),
            'MIN_SIGNAL_CONFIDENCE': float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.65)),
            'MAX_DAILY_LOSS': float(os.getenv('MAX_DAILY_LOSS', 1000)),
            'MAX_POSITION_SIZE': float(os.getenv('MAX_POSITION_SIZE_DOLLARS', 10000)),
            'MAX_POSITIONS': int(os.getenv('MAX_POSITIONS', 5)),
            'SIMULATION_MODE': os.getenv('SIMULATION_MODE', 'true').lower() == 'true',
            'TRADING_MODE': os.getenv('TRADING_MODE', 'paper'),
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
            'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY')
        }
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data with caching"""
        try:
            # Check cache (5 minute expiry)
            if symbol in self.cache_time:
                if datetime.now() - self.cache_time[symbol] < timedelta(minutes=5):
                    return self.data_cache[symbol]
            
            # Get fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo")
            
            if not data.empty:
                self.data_cache[symbol] = data
                self.cache_time[symbol] = datetime.now()
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is open (simplified)"""
        now = datetime.now()
        
        # Check if weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours (9:30 AM - 4:00 PM EST)
        market_time = now.time()
        return time(9, 30) <= market_time <= time(16, 0)
    
    async def scan_and_trade(self):
        """Scan symbols and execute trades"""
        try:
            logger.info(f"🔍 Scanning {len(self.config['SCAN_SYMBOLS'])} symbols...")
            
            opportunities = []
            
            for symbol in self.config['SCAN_SYMBOLS']:
                try:
                    # Get market data
                    data = self.get_market_data(symbol)
                    if data is None or len(data) < 50:
                        continue
                    
                    # Generate signal
                    signal = self.strategies.combine_signals(data, symbol)
                    
                    # Filter by confidence
                    if signal['confidence'] < self.config['MIN_SIGNAL_CONFIDENCE']:
                        continue
                    
                    # Skip if already have position
                    if symbol in self.positions:
                        continue
                    
                    # Check risk
                    test_order = {
                        'symbol': symbol,
                        'quantity': 100,
                        'price': data['Close'].iloc[-1]
                    }
                    
                    can_trade, risk_reason = self.risk_manager.check_risk(test_order, self.account, self.positions)
                    if not can_trade:
                        logger.debug(f"Risk check failed for {symbol}: {risk_reason}")
                        continue
                    
                    # Calculate position size
                    current_price = data['Close'].iloc[-1]
                    signal['entry_price'] = current_price
                    quantity = self.risk_manager.calculate_position_size(signal, self.account, current_price)
                    
                    if quantity > 0:
                        opportunities.append({
                            'symbol': symbol,
                            'signal': signal,
                            'quantity': quantity,
                            'price': current_price
                        })
                        
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
            
            # Execute best opportunities
            opportunities.sort(key=lambda x: x['signal']['confidence'], reverse=True)
            
            for opp in opportunities[:2]:  # Max 2 trades per scan
                await self._execute_trade(opp)
                await asyncio.sleep(1)  # Small delay between trades
            
            logger.info(f"Scan complete: {len(opportunities)} opportunities found")
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
    
    async def _execute_trade(self, opportunity: Dict):
        """Execute a trading opportunity"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            quantity = opportunity['quantity']
            price = opportunity['price']
            
            # Calculate stop loss and take profit
            if signal['signal'] == 'BUY':
                stop_loss = price * 0.98  # 2% stop loss
                take_profit = price * 1.05  # 5% take profit
            else:
                stop_loss = price * 1.02
                take_profit = price * 0.95
            
            logger.info(f"🎯 Executing: {symbol} {signal['signal']} qty={quantity} @ ${price:.2f} (conf: {signal['confidence']:.1%})")
            
            # Execute order
            result = await self.executor.execute_order(signal, quantity, stop_loss, take_profit)
            
            if result.get('status') == 'FILLED':
                # Record position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': result['quantity'],
                    'entry_price': result['fill_price'],
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal': signal
                }
                
                self.daily_stats['trades'] += 1
                logger.info(f"✅ Position opened: {symbol} {quantity} shares @ ${result['fill_price']:.2f}")
                
                # Log trade
                self._log_trade(symbol, signal, result)
                
            else:
                logger.warning(f"❌ Trade failed: {symbol} - {result.get('reason', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            if not self.positions:
                return
            
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                try:
                    # Get current price
                    data = self.get_market_data(symbol)
                    if data is None:
                        continue
                    
                    current_price = data['Close'].iloc[-1]
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    
                    # Calculate P&L
                    unrealized_pnl = (current_price - entry_price) * quantity
                    pnl_percent = (current_price - entry_price) / entry_price
                    
                    # Check exit conditions
                    stop_loss = position.get('stop_loss')
                    take_profit = position.get('take_profit')
                    
                    # Stop loss check
                    if stop_loss and current_price <= stop_loss:
                        positions_to_close.append((symbol, 'stop_loss', unrealized_pnl))
                        continue
                    
                    # Take profit check
                    if take_profit and current_price >= take_profit:
                        positions_to_close.append((symbol, 'take_profit', unrealized_pnl))
                        continue
                    
                    # Time-based exit (4 hour max hold)
                    hold_time = datetime.now() - position['entry_time']
                    if hold_time > timedelta(hours=4):
                        positions_to_close.append((symbol, 'time_exit', unrealized_pnl))
                        continue
                    
                    logger.debug(f"Position {symbol}: P&L=${unrealized_pnl:.2f} ({pnl_percent:.1%})")
                    
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
            
            # Close positions
            for symbol, reason, pnl in positions_to_close:
                await self._close_position(symbol, reason, pnl)
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    async def _close_position(self, symbol: str, reason: str, pnl: float):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Simulate close order
            close_result = {
                'status': 'FILLED',
                'fill_price': position['entry_price'] + (pnl / position['quantity']),
                'quantity': position['quantity']
            }
            
            # Update stats
            if pnl > 0:
                self.daily_stats['wins'] += 1
            else:
                self.daily_stats['losses'] += 1
            
            self.daily_stats['pnl'] += pnl
            self.risk_manager.update_pnl(pnl)
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"✅ Position closed: {symbol} - {reason} - P&L: ${pnl:.2f}")
            
            # Log close
            self._log_close(symbol, position, close_result, reason, pnl)
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
    
    def _log_trade(self, symbol: str, signal: Dict, result: Dict):
        """Log trade execution"""
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'quantity': result['quantity'],
                'price': result['fill_price'],
                'reason': signal['reason']
            }
            
            with open('trade_log.json', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
                
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    def _log_close(self, symbol: str, position: Dict, result: Dict, reason: str, pnl: float):
        """Log position close"""
        try:
            close_log = {
                'timestamp': datetime.now().isoformat(),
                'action': 'CLOSE',
                'symbol': symbol,
                'reason': reason,
                'entry_price': position['entry_price'],
                'exit_price': result['fill_price'],
                'quantity': result['quantity'],
                'pnl': pnl,
                'hold_time': str(datetime.now() - position['entry_time'])
            }
            
            with open('trade_log.json', 'a') as f:
                f.write(json.dumps(close_log) + '\n')
                
        except Exception as e:
            logger.error(f"Close logging error: {e}")
    
    def print_status(self):
        """Print current bot status"""
        uptime = datetime.now() - self.start_time
        total_trades = self.daily_stats['wins'] + self.daily_stats['losses']
        win_rate = self.daily_stats['wins'] / total_trades if total_trades > 0 else 0
        
        print(f"\n📊 TRADING BOT STATUS")
        print(f"⏰ Uptime: {uptime}")
        print(f"💰 Daily P&L: ${self.daily_stats['pnl']:.2f}")
        print(f"📈 Trades: {self.daily_stats['trades']} (Win Rate: {win_rate:.1%})")
        print(f"📋 Positions: {len(self.positions)}")
        print(f"🛡️ Risk Level: {'NORMAL' if not self.risk_manager.circuit_breaker_active else 'CIRCUIT BREAKER'}")
        
        if self.positions:
            print(f"📍 Active Positions:")
            for symbol, pos in self.positions.items():
                hold_time = datetime.now() - pos['entry_time']
                print(f"   {symbol}: {pos['quantity']} shares @ ${pos['entry_price']:.2f} (held: {hold_time})")
    
    async def run(self):
        """Main trading loop"""
        logger.info("🚀 OMNI ALPHA 5.0 - SIMPLE TRADING BOT STARTED")
        logger.info("=" * 60)
        
        # Print initial configuration
        print(f"\n🔧 Configuration:")
        print(f"   Symbols: {self.config['SCAN_SYMBOLS']}")
        print(f"   Scan Interval: {self.config['SCAN_INTERVAL']} seconds")
        print(f"   Min Confidence: {self.config['MIN_SIGNAL_CONFIDENCE']:.1%}")
        print(f"   Max Daily Loss: ${self.config['MAX_DAILY_LOSS']}")
        print(f"   Max Positions: {self.config['MAX_POSITIONS']}")
        print(f"   Simulation Mode: {self.config['SIMULATION_MODE']}")
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # Check if market is open
                    if not self.is_market_open():
                        logger.info("Market closed, waiting...")
                        await asyncio.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # Scan for opportunities
                    await self.scan_and_trade()
                    
                    # Monitor existing positions
                    await self.monitor_positions()
                    
                    # Print status every 10 minutes
                    if datetime.now().minute % 10 == 0:
                        self.print_status()
                    
                    # Wait for next scan
                    await asyncio.sleep(self.config['SCAN_INTERVAL'])
                    
                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(30)
            
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down trading bot...")
        self.running = False
        
        # Print final statistics
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Total Trades: {self.daily_stats['trades']}")
        print(f"   Wins: {self.daily_stats['wins']}")
        print(f"   Losses: {self.daily_stats['losses']}")
        print(f"   Total P&L: ${self.daily_stats['pnl']:.2f}")
        print(f"   Active Positions: {len(self.positions)}")
        
        # Save final report
        final_report = {
            'session_stats': self.daily_stats,
            'final_positions': self.positions,
            'config': self.config,
            'end_time': datetime.now().isoformat()
        }
        
        with open(f'session_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info("✅ Trading bot shutdown complete")

# Main execution
async def main():
    """Main function"""
    bot = SimpleTradingBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 OMNI ALPHA 5.0 - SIMPLE TRADING BOT")
    print("=" * 50)
    print("Starting automated trading system...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    asyncio.run(main())


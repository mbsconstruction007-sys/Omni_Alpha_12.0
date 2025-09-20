#!/usr/bin/env python3
"""
OMNI ALPHA 5.0 - CLEAN TRADING BOT
==================================
Production-ready trading bot without Unicode issues
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

# Setup logging without Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingStrategies:
    """Production trading strategies"""
    
    def moving_average_signal(self, data: pd.DataFrame) -> Dict:
        """Moving average crossover"""
        if len(data) < 50:
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        if prev['SMA_20'] <= prev['SMA_50'] and last['SMA_20'] > last['SMA_50']:
            return {'signal': 'BUY', 'confidence': 0.7, 'reason': 'Golden Cross'}
        elif prev['SMA_20'] >= prev['SMA_50'] and last['SMA_20'] < last['SMA_50']:
            return {'signal': 'SELL', 'confidence': 0.7, 'reason': 'Death Cross'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'No MA signal'}
    
    def rsi_signal(self, data: pd.DataFrame) -> Dict:
        """RSI strategy"""
        if len(data) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        rsi = data['RSI'].iloc[-1]
        
        if rsi < 30:
            return {'signal': 'BUY', 'confidence': 0.65, 'reason': f'RSI oversold {rsi:.1f}'}
        elif rsi > 70:
            return {'signal': 'SELL', 'confidence': 0.65, 'reason': f'RSI overbought {rsi:.1f}'}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reason': f'RSI neutral {rsi:.1f}'}
    
    def combine_signals(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Combine strategies"""
        signals = [
            self.moving_average_signal(data),
            self.rsi_signal(data)
        ]
        
        buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
        avg_conf = np.mean([s['confidence'] for s in signals])
        
        if buy_count >= 2:
            return {'signal': 'BUY', 'confidence': min(0.8, avg_conf * 1.2), 'symbol': symbol}
        elif sell_count >= 2:
            return {'signal': 'SELL', 'confidence': min(0.8, avg_conf * 1.2), 'symbol': symbol}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'symbol': symbol}

class RiskManager:
    """Simple risk management"""
    
    def __init__(self, config):
        self.max_daily_loss = config.get('MAX_DAILY_LOSS', 1000)
        self.max_positions = config.get('MAX_POSITIONS', 5)
        self.daily_pnl = 0
        self.daily_trades = 0
        self.circuit_breaker = False
    
    def check_risk(self, symbol: str, quantity: int, price: float, positions: Dict) -> Tuple[bool, str]:
        """Risk validation"""
        
        if self.circuit_breaker:
            return False, "Circuit breaker active"
        
        if self.daily_pnl <= -self.max_daily_loss:
            self.circuit_breaker = True
            return False, "Daily loss limit reached"
        
        if len(positions) >= self.max_positions:
            return False, f"Max {self.max_positions} positions"
        
        position_value = quantity * price
        if position_value > 10000:
            return False, "Position too large"
        
        return True, "OK"
    
    def calculate_size(self, signal: Dict, price: float) -> int:
        """Calculate position size"""
        confidence = signal.get('confidence', 0.6)
        base_size = int(5000 / price)  # $5000 base position
        
        # Adjust by confidence
        adjusted_size = int(base_size * confidence)
        
        return max(1, min(adjusted_size, 100))  # Min 1, max 100 shares

class TradingBot:
    """Main trading bot"""
    
    def __init__(self):
        self.config = self._load_config()
        self.strategies = TradingStrategies()
        self.risk_manager = RiskManager(self.config)
        
        self.positions = {}
        self.daily_stats = {'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0}
        self.running = False
        self.start_time = datetime.now()
        
        # Data cache
        self.data_cache = {}
        self.cache_time = {}
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        return {
            'SCAN_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
            'SCAN_INTERVAL': 300,  # 5 minutes for demo
            'MIN_CONFIDENCE': 0.65,
            'MAX_DAILY_LOSS': 1000,
            'MAX_POSITIONS': 3,
            'SIMULATION_MODE': True
        }
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data"""
        try:
            # Check cache
            if symbol in self.cache_time:
                if datetime.now() - self.cache_time[symbol] < timedelta(minutes=10):
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
            logger.error(f"Data error for {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check market hours"""
        now = datetime.now()
        
        # For demo, let's say market is always open
        return True
        
        # Real market hours check:
        # if now.weekday() >= 5:
        #     return False
        # market_time = now.time()
        # return time(9, 30) <= market_time <= time(16, 0)
    
    async def scan_opportunities(self):
        """Scan for trading opportunities"""
        logger.info("Scanning for opportunities...")
        
        opportunities = []
        
        for symbol in self.config['SCAN_SYMBOLS']:
            try:
                # Get data
                data = self.get_data(symbol)
                if data is None or len(data) < 50:
                    continue
                
                # Generate signal
                signal = self.strategies.combine_signals(data, symbol)
                
                # Check confidence
                if signal['confidence'] < self.config['MIN_CONFIDENCE']:
                    continue
                
                # Skip if we have position
                if symbol in self.positions:
                    continue
                
                # Check risk
                current_price = data['Close'].iloc[-1]
                quantity = self.risk_manager.calculate_size(signal, current_price)
                
                can_trade, reason = self.risk_manager.check_risk(
                    symbol, quantity, current_price, self.positions
                )
                
                if can_trade:
                    opportunities.append({
                        'symbol': symbol,
                        'signal': signal,
                        'quantity': quantity,
                        'price': current_price
                    })
                    logger.info(f"Opportunity: {symbol} {signal['signal']} conf={signal['confidence']:.1%}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return opportunities
    
    async def execute_trades(self, opportunities: List[Dict]):
        """Execute trading opportunities"""
        
        for opp in opportunities[:2]:  # Max 2 trades per scan
            try:
                symbol = opp['symbol']
                signal = opp['signal']
                quantity = opp['quantity']
                price = opp['price']
                
                logger.info(f"EXECUTING: {symbol} {signal['signal']} {quantity} shares @ ${price:.2f}")
                
                # Simulate execution
                await asyncio.sleep(0.1)
                
                # Calculate stop loss and take profit
                if signal['signal'] == 'BUY':
                    stop_loss = price * 0.98
                    take_profit = price * 1.04
                else:
                    stop_loss = price * 1.02
                    take_profit = price * 0.96
                
                # Record position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal': signal
                }
                
                self.daily_stats['trades'] += 1
                
                logger.info(f"SUCCESS: {symbol} position opened - SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                
                # Log to file
                self._log_trade(symbol, signal, quantity, price)
                
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
    
    async def monitor_positions(self):
        """Monitor existing positions"""
        
        if not self.positions:
            return
        
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            try:
                # Get current price
                data = self.get_data(symbol)
                if data is None:
                    continue
                
                current_price = data['Close'].iloc[-1]
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # Calculate P&L
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Check exit conditions
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                if current_price <= stop_loss:
                    positions_to_close.append((symbol, 'stop_loss', pnl))
                elif current_price >= take_profit:
                    positions_to_close.append((symbol, 'take_profit', pnl))
                else:
                    # Check time-based exit (2 hours for demo)
                    hold_time = datetime.now() - position['entry_time']
                    if hold_time > timedelta(hours=2):
                        positions_to_close.append((symbol, 'time_exit', pnl))
                    else:
                        logger.info(f"Position {symbol}: P&L=${pnl:.2f} ({pnl_pct:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
        
        # Close positions
        for symbol, reason, pnl in positions_to_close:
            await self._close_position(symbol, reason, pnl)
    
    async def _close_position(self, symbol: str, reason: str, pnl: float):
        """Close position"""
        try:
            position = self.positions[symbol]
            
            # Update stats
            if pnl > 0:
                self.daily_stats['wins'] += 1
            else:
                self.daily_stats['losses'] += 1
            
            self.daily_stats['pnl'] += pnl
            self.risk_manager.daily_pnl += pnl
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"CLOSED: {symbol} - {reason} - P&L: ${pnl:.2f}")
            
            # Log close
            self._log_close(symbol, position, reason, pnl)
            
        except Exception as e:
            logger.error(f"Close error: {e}")
    
    def _log_trade(self, symbol: str, signal: Dict, quantity: int, price: float):
        """Log trade"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'quantity': quantity,
                'price': price
            }
            
            with open('trades.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def _log_close(self, symbol: str, position: Dict, reason: str, pnl: float):
        """Log position close"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'CLOSE',
                'symbol': symbol,
                'reason': reason,
                'entry_price': position['entry_price'],
                'quantity': position['quantity'],
                'pnl': pnl,
                'hold_time': str(datetime.now() - position['entry_time'])
            }
            
            with open('trades.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Close logging error: {e}")
    
    def print_status(self):
        """Print status"""
        uptime = datetime.now() - self.start_time
        total_trades = self.daily_stats['wins'] + self.daily_stats['losses']
        win_rate = self.daily_stats['wins'] / total_trades if total_trades > 0 else 0
        
        print(f"\nTRADING BOT STATUS")
        print(f"Uptime: {uptime}")
        print(f"Daily P&L: ${self.daily_stats['pnl']:.2f}")
        print(f"Trades: {total_trades} (Win Rate: {win_rate:.1%})")
        print(f"Active Positions: {len(self.positions)}")
        
        if self.positions:
            print("Positions:")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos['quantity']} @ ${pos['entry_price']:.2f}")
    
    async def run(self):
        """Main trading loop"""
        print("OMNI ALPHA 5.0 - TRADING BOT STARTED")
        print("Configuration:")
        print(f"  Symbols: {self.config['SCAN_SYMBOLS']}")
        print(f"  Scan Interval: {self.config['SCAN_INTERVAL']} seconds")
        print(f"  Min Confidence: {self.config['MIN_CONFIDENCE']:.1%}")
        print(f"  Simulation Mode: {self.config['SIMULATION_MODE']}")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # Check market (for demo, always open)
                    if not self.is_market_open():
                        logger.info("Market closed, waiting...")
                        await asyncio.sleep(300)
                        continue
                    
                    # Scan for opportunities
                    opportunities = await self.scan_opportunities()
                    
                    # Execute trades
                    if opportunities:
                        await self.execute_trades(opportunities)
                    
                    # Monitor positions
                    await self.monitor_positions()
                    
                    # Print status every 5 scans
                    if self.daily_stats['trades'] % 5 == 0 and self.daily_stats['trades'] > 0:
                        self.print_status()
                    
                    # Wait for next scan
                    logger.info(f"Waiting {self.config['SCAN_INTERVAL']} seconds for next scan...")
                    await asyncio.sleep(self.config['SCAN_INTERVAL'])
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    await asyncio.sleep(30)
            
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown"""
        print("\nShutting down trading bot...")
        self.running = False
        
        # Final stats
        print(f"\nFINAL STATISTICS:")
        print(f"Total Trades: {self.daily_stats['trades']}")
        print(f"Wins: {self.daily_stats['wins']}")
        print(f"Losses: {self.daily_stats['losses']}")
        print(f"Total P&L: ${self.daily_stats['pnl']:.2f}")
        print(f"Final Positions: {len(self.positions)}")
        
        logger.info("Trading bot shutdown complete")

# Quick test function
async def quick_test():
    """Quick test of strategies"""
    print("OMNI ALPHA 5.0 - QUICK STRATEGY TEST")
    print("=" * 50)
    
    strategies = TradingStrategies()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    for symbol in symbols:
        try:
            print(f"\nTesting {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            
            if not data.empty:
                signal = strategies.combine_signals(data, symbol)
                print(f"  Signal: {signal['signal']} (Confidence: {signal['confidence']:.1%})")
                print(f"  Current Price: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"  No data available")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nStrategy test complete!")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Quick test mode
        asyncio.run(quick_test())
    else:
        # Full trading bot
        bot = TradingBot()
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nTrading bot stopped by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()


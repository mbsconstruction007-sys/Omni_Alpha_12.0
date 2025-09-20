"""
OMNI ALPHA 5.0 - MAIN TRADING BOT
=================================
Complete trading bot orchestrator with all components integrated
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import signal
import json
import os
from pathlib import Path

# Import our working components
from strategies.working_strategies import TradingStrategies
from risk.risk_manager import RiskManager
from execution.order_executor import OrderExecutor

# Core system components
from config.settings import get_settings
from database.simple_connection import DatabaseManager
from data_collection.fixed_alpaca_collector import FixedAlpacaCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot orchestrator with complete functionality"""
    
    def __init__(self, config_override: Dict = None):
        self.config = self._load_config(config_override)
        self.running = False
        self.start_time = datetime.now()
        
        # Core components
        self.strategies = TradingStrategies(self.config)
        self.risk_manager = RiskManager(self.config)
        self.executor = OrderExecutor(self.config)
        
        # Data components
        self.database = None
        self.data_collector = None
        
        # Trading state
        self.positions = {}
        self.account = {}
        self.last_scan_time = None
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'start_balance': 0.0
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.cache_expiry = {}
        
        # Performance tracking
        self.performance_log = []
        
    def _load_config(self, config_override: Dict = None) -> Dict:
        """Load and validate configuration"""
        
        try:
            # Load from settings
            settings = get_settings()
            base_config = {
                # Alpaca API
                'ALPACA_API_KEY': getattr(settings, 'alpaca_key', None),
                'ALPACA_SECRET_KEY': getattr(settings, 'alpaca_secret', None),
                'TRADING_MODE': getattr(settings, 'trading_mode', 'paper'),
                
                # Risk Management
                'MAX_DAILY_LOSS': getattr(settings, 'max_daily_loss', 1000),
                'MAX_POSITION_SIZE': getattr(settings, 'max_position_size_dollars', 10000),
                'MAX_POSITIONS': 5,
                'MAX_DAILY_TRADES': 20,
                'MAX_PORTFOLIO_RISK': 0.02,
                'MAX_SINGLE_POSITION_RISK': 0.01,
                'MAX_SECTOR_CONCENTRATION': 0.3,
                
                # Trading Parameters
                'SCAN_SYMBOLS': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'NVDA'],
                'SCAN_INTERVAL': 60,  # seconds
                'MIN_SIGNAL_CONFIDENCE': 0.6,
                'POSITION_SIZE_METHOD': 'combined',
                
                # System Settings
                'SIMULATION_MODE': not bool(getattr(settings, 'alpaca_secret', None)),
                'DATABASE_ENABLED': True,
                'MONITORING_ENABLED': True,
                'LOG_LEVEL': 'INFO'
            }
            
            # Apply overrides
            if config_override:
                base_config.update(config_override)
            
            # Validation
            if not base_config['SIMULATION_MODE']:
                if not base_config['ALPACA_API_KEY'] or not base_config['ALPACA_SECRET_KEY']:
                    logger.warning("No Alpaca credentials found, forcing simulation mode")
                    base_config['SIMULATION_MODE'] = True
            
            return base_config
            
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            # Return minimal safe config
            return {
                'SIMULATION_MODE': True,
                'TRADING_MODE': 'paper',
                'SCAN_SYMBOLS': ['SPY'],
                'SCAN_INTERVAL': 60,
                'MAX_DAILY_LOSS': 100,
                'MAX_POSITION_SIZE': 1000,
                'MAX_POSITIONS': 2,
                'MIN_SIGNAL_CONFIDENCE': 0.7
            }
    
    async def initialize(self):
        """Initialize all bot components"""
        
        logger.info("🚀 OMNI ALPHA 5.0 - TRADING BOT INITIALIZATION")
        logger.info("=" * 60)
        
        try:
            # Initialize database
            if self.config.get('DATABASE_ENABLED'):
                try:
                    self.database = DatabaseManager(self.config)
                    await self.database.initialize()
                    logger.info("✅ Database initialized")
                except Exception as e:
                    logger.warning(f"Database initialization failed: {e}")
            
            # Initialize data collector
            try:
                self.data_collector = FixedAlpacaCollector(self.config)
                data_connected = await self.data_collector.initialize()
                
                if data_connected:
                    # Start streaming for our symbols
                    await self.data_collector.start_streaming(self.config['SCAN_SYMBOLS'][:5])
                    logger.info("✅ Data collector initialized and streaming")
                else:
                    logger.info("✅ Data collector initialized (demo mode)")
                    
            except Exception as e:
                logger.error(f"Data collector initialization failed: {e}")
                self.data_collector = None
            
            # Get initial account info
            await self._update_account_info()
            
            # Initialize daily stats
            self.daily_stats['start_balance'] = self.account.get('portfolio_value', 0)
            
            # Reset daily risk metrics if new day
            self.risk_manager.reset_daily_metrics()
            
            logger.info("✅ Trading bot initialization complete")
            self._print_startup_status()
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            raise
    
    async def run(self):
        """Main trading loop"""
        
        try:
            logger.info("🎯 Starting main trading loop...")
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            self.running = True
            
            # Main trading loop
            while self.running:
                try:
                    # Check if market is open
                    if not await self._is_market_open():
                        logger.info("Market is closed, waiting...")
                        await asyncio.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # Update account and positions
                    await self._update_account_info()
                    
                    # Scan for trading opportunities
                    await self._scan_and_trade()
                    
                    # Monitor existing positions
                    await self._monitor_positions()
                    
                    # Update performance metrics
                    await self._update_performance_metrics()
                    
                    # Log status periodically
                    if datetime.now().minute % 10 == 0:  # Every 10 minutes
                        self._log_status()
                    
                    # Wait before next scan
                    await asyncio.sleep(self.config['SCAN_INTERVAL'])
                    
                except Exception as e:
                    logger.error(f"Error in main trading loop: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds before retrying
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
        finally:
            await self.shutdown()
    
    async def _scan_and_trade(self):
        """Scan symbols and execute trades based on signals"""
        
        try:
            scan_start = datetime.now()
            opportunities = []
            
            for symbol in self.config['SCAN_SYMBOLS']:
                try:
                    # Get market data
                    data = await self._get_market_data(symbol)
                    
                    if data is None or len(data) < 50:
                        logger.debug(f"Insufficient data for {symbol}")
                        continue
                    
                    # Generate combined signal
                    signal = self.strategies.combine_signals(data, symbol)
                    
                    # Filter by minimum confidence
                    if signal.get('confidence', 0) < self.config['MIN_SIGNAL_CONFIDENCE']:
                        continue
                    
                    # Skip if we already have a position in this symbol
                    if symbol in self.positions:
                        logger.debug(f"Already have position in {symbol}, skipping")
                        continue
                    
                    # Pre-trade risk checks
                    test_order = {
                        'symbol': symbol,
                        'quantity': 100,  # Temporary for risk check
                        'price': data['close'].iloc[-1]
                    }
                    
                    can_trade, risk_reason, risk_details = self.risk_manager.check_pre_trade_risk(
                        test_order, self.account, self.positions
                    )
                    
                    if not can_trade:
                        logger.debug(f"Risk check failed for {symbol}: {risk_reason}")
                        continue
                    
                    # Calculate position size
                    sizing_result = self.risk_manager.calculate_position_size(
                        signal, self.account, data['close'].iloc[-1], data
                    )
                    
                    if sizing_result['quantity'] <= 0:
                        logger.debug(f"No position size calculated for {symbol}")
                        continue
                    
                    # Add to opportunities
                    opportunities.append({
                        'symbol': symbol,
                        'signal': signal,
                        'sizing': sizing_result,
                        'current_price': data['close'].iloc[-1],
                        'data': data
                    })
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort opportunities by confidence and execute best ones
            opportunities.sort(key=lambda x: x['signal']['confidence'], reverse=True)
            
            executed_trades = 0
            max_trades_per_scan = 3  # Limit trades per scan
            
            for opp in opportunities[:max_trades_per_scan]:
                if not self.running:
                    break
                
                try:
                    await self._execute_opportunity(opp)
                    executed_trades += 1
                    
                    # Small delay between trades
                    if executed_trades < len(opportunities):
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Error executing opportunity for {opp['symbol']}: {e}")
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            logger.info(f"Scan completed: {len(opportunities)} opportunities, {executed_trades} trades executed in {scan_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Scan and trade error: {e}")
    
    async def _execute_opportunity(self, opportunity: Dict):
        """Execute a trading opportunity"""
        
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            sizing = opportunity['sizing']
            
            logger.info(f"🎯 Executing trade: {symbol} {signal['signal']} (confidence: {signal['confidence']:.2%})")
            
            # Calculate risk parameters
            current_price = opportunity['current_price']
            position_type = 'LONG' if signal['signal'] in ['BUY', 'WEAK_BUY'] else 'SHORT'
            
            risk_params = {
                'quantity': sizing['quantity'],
                'stop_loss': self.risk_manager.calculate_stop_loss(
                    current_price, position_type, symbol, opportunity['data']
                ),
                'take_profit': self.risk_manager.calculate_take_profit(
                    current_price, position_type, signal['confidence']
                )
            }
            
            # Execute the trade
            execution_result = await self.executor.execute_signal(signal, risk_params, self.account)
            
            if execution_result.status.value in ['FILLED', 'PARTIALLY_FILLED']:
                # Record successful trade
                self.daily_stats['trades'] += 1
                
                # Update positions
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': execution_result.filled_quantity,
                    'entry_price': execution_result.filled_price,
                    'entry_time': execution_result.timestamp,
                    'signal': signal,
                    'risk_params': risk_params,
                    'order_id': execution_result.order_id
                }
                
                # Update risk manager
                trade_value = execution_result.filled_quantity * execution_result.filled_price
                self.risk_manager.update_daily_pnl(0, 1)  # No P&L yet, just trade count
                
                logger.info(f"✅ Trade executed: {symbol} {signal['signal']} "
                          f"qty={execution_result.filled_quantity} @ ${execution_result.filled_price:.2f}")
                
                # Log trade details
                self._log_trade_execution(symbol, signal, risk_params, execution_result)
                
            else:
                logger.warning(f"❌ Trade failed: {symbol} - {execution_result.error_message}")
                
        except Exception as e:
            logger.error(f"Trade execution error for {opportunity['symbol']}: {e}")
    
    async def _monitor_positions(self):
        """Monitor existing positions and manage risk"""
        
        try:
            if not self.positions:
                return
            
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                try:
                    # Get current market data
                    data = await self._get_market_data(symbol)
                    if data is None or len(data) == 0:
                        continue
                    
                    current_price = data['close'].iloc[-1]
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    
                    # Calculate current P&L
                    unrealized_pnl = (current_price - entry_price) * quantity
                    pnl_percent = (current_price - entry_price) / entry_price
                    
                    # Check stop loss
                    stop_loss = position['risk_params'].get('stop_loss')
                    if stop_loss and current_price <= stop_loss:
                        logger.info(f"🛑 Stop loss triggered for {symbol}: ${current_price:.2f} <= ${stop_loss:.2f}")
                        positions_to_close.append((symbol, 'stop_loss', unrealized_pnl))
                        continue
                    
                    # Check take profit
                    take_profit = position['risk_params'].get('take_profit')
                    if take_profit and current_price >= take_profit:
                        logger.info(f"🎯 Take profit triggered for {symbol}: ${current_price:.2f} >= ${take_profit:.2f}")
                        positions_to_close.append((symbol, 'take_profit', unrealized_pnl))
                        continue
                    
                    # Check time-based exit (hold for max 1 day)
                    hold_time = datetime.now() - position['entry_time']
                    if hold_time > timedelta(hours=6):  # 6 hour max hold
                        logger.info(f"⏰ Time-based exit for {symbol}: held for {hold_time}")
                        positions_to_close.append((symbol, 'time_exit', unrealized_pnl))
                        continue
                    
                    # Check for signal reversal
                    current_signal = self.strategies.combine_signals(data, symbol)
                    original_signal = position['signal']['signal']
                    
                    if (original_signal in ['BUY', 'WEAK_BUY'] and 
                        current_signal['signal'] in ['SELL', 'WEAK_SELL'] and 
                        current_signal['confidence'] > 0.7):
                        logger.info(f"🔄 Signal reversal for {symbol}: {original_signal} -> {current_signal['signal']}")
                        positions_to_close.append((symbol, 'signal_reversal', unrealized_pnl))
                        continue
                    
                    # Log position status
                    logger.debug(f"Position {symbol}: P&L=${unrealized_pnl:.2f} ({pnl_percent:.2%}), "
                               f"Price=${current_price:.2f}, Hold={hold_time}")
                    
                except Exception as e:
                    logger.error(f"Error monitoring position {symbol}: {e}")
            
            # Close positions that need to be closed
            for symbol, reason, pnl in positions_to_close:
                await self._close_position(symbol, reason, pnl)
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    async def _close_position(self, symbol: str, reason: str, pnl: float):
        """Close a position"""
        
        try:
            if symbol not in self.positions:
                logger.warning(f"Trying to close non-existent position: {symbol}")
                return
            
            position = self.positions[symbol]
            
            # Create close signal
            close_signal = {
                'symbol': symbol,
                'signal': 'SELL',  # Always sell to close long positions
                'confidence': 0.8,
                'reason': f'Position close: {reason}'
            }
            
            close_risk_params = {
                'quantity': position['quantity'],
                'stop_loss': None,
                'take_profit': None
            }
            
            # Execute close order
            execution_result = await self.executor.execute_signal(close_signal, close_risk_params, self.account)
            
            if execution_result.status.value in ['FILLED', 'PARTIALLY_FILLED']:
                # Calculate realized P&L
                realized_pnl = (execution_result.filled_price - position['entry_price']) * execution_result.filled_quantity
                
                # Update statistics
                if realized_pnl > 0:
                    self.daily_stats['wins'] += 1
                else:
                    self.daily_stats['losses'] += 1
                
                self.daily_stats['pnl'] += realized_pnl
                
                # Update risk manager
                self.risk_manager.update_daily_pnl(realized_pnl, 1)
                
                # Remove from positions
                del self.positions[symbol]
                
                logger.info(f"✅ Position closed: {symbol} - {reason} - P&L: ${realized_pnl:.2f}")
                
                # Log position close
                self._log_position_close(symbol, position, execution_result, reason, realized_pnl)
                
            else:
                logger.error(f"❌ Failed to close position {symbol}: {execution_result.error_message}")
                
        except Exception as e:
            logger.error(f"Position close error for {symbol}: {e}")
    
    async def _get_market_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Get market data for a symbol with caching"""
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            now = datetime.now()
            
            if (cache_key in self.market_data_cache and 
                cache_key in self.cache_expiry and 
                now < self.cache_expiry[cache_key]):
                return self.market_data_cache[cache_key]
            
            # Get fresh data
            data = None
            
            # Try to get from data collector first
            if self.data_collector:
                data = await self.data_collector.get_historical_data(symbol, days=180)
            
            # Fallback to yfinance
            if data is None or len(data) < 10:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty:
                        data.columns = data.columns.str.lower()
                        data = data.reset_index()
                        if 'date' in data.columns:
                            data = data.set_index('date')
                except Exception as e:
                    logger.debug(f"yfinance fallback failed for {symbol}: {e}")
            
            # Cache the data
            if data is not None and len(data) > 0:
                self.market_data_cache[cache_key] = data
                self.cache_expiry[cache_key] = now + timedelta(minutes=5)  # Cache for 5 minutes
                
                return data
            else:
                logger.warning(f"No market data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return None
    
    async def _update_account_info(self):
        """Update account information"""
        
        try:
            if self.executor.simulation_mode:
                # Simulate account info
                self.account = {
                    'portfolio_value': 100000 + self.daily_stats['pnl'],
                    'buying_power': 50000,
                    'cash': 50000,
                    'equity': 100000 + self.daily_stats['pnl']
                }
            else:
                # Get real account info from Alpaca
                if self.executor.client:
                    account = self.executor.client.get_account()
                    self.account = {
                        'portfolio_value': float(account.portfolio_value),
                        'buying_power': float(account.buying_power),
                        'cash': float(account.cash),
                        'equity': float(account.equity)
                    }
                    
                    # Get positions
                    positions = self.executor.client.get_all_positions()
                    self.positions = {}
                    
                    for pos in positions:
                        self.positions[pos.symbol] = {
                            'symbol': pos.symbol,
                            'quantity': int(pos.qty),
                            'entry_price': float(pos.avg_cost),
                            'current_price': float(pos.market_value) / int(pos.qty),
                            'unrealized_pnl': float(pos.unrealized_pnl)
                        }
                        
        except Exception as e:
            logger.error(f"Account update error: {e}")
    
    async def _is_market_open(self) -> bool:
        """Check if market is open"""
        
        try:
            if self.executor.simulation_mode:
                # Simple time check for simulation
                now = datetime.now().time()
                market_open = time(9, 30)  # 9:30 AM
                market_close = time(16, 0)  # 4:00 PM
                
                # Check if it's a weekday and within market hours
                weekday = datetime.now().weekday() < 5  # Monday = 0, Friday = 4
                return weekday and market_open <= now <= market_close
            
            else:
                # Get real market status from Alpaca
                if self.executor.client:
                    clock = self.executor.client.get_clock()
                    return clock.is_open
                
            return False
            
        except Exception as e:
            logger.error(f"Market status check error: {e}")
            return False
    
    async def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        
        try:
            current_time = datetime.now()
            
            # Calculate daily return
            if self.daily_stats['start_balance'] > 0:
                daily_return = self.daily_stats['pnl'] / self.daily_stats['start_balance']
            else:
                daily_return = 0.0
            
            # Calculate win rate
            total_closed_trades = self.daily_stats['wins'] + self.daily_stats['losses']
            win_rate = self.daily_stats['wins'] / total_closed_trades if total_closed_trades > 0 else 0.0
            
            # Add to performance log
            self.performance_log.append({
                'timestamp': current_time,
                'portfolio_value': self.account.get('portfolio_value', 0),
                'daily_pnl': self.daily_stats['pnl'],
                'daily_return': daily_return,
                'trades': self.daily_stats['trades'],
                'win_rate': win_rate,
                'active_positions': len(self.positions)
            })
            
            # Keep only last 1000 entries
            if len(self.performance_log) > 1000:
                self.performance_log = self.performance_log[-1000:]
                
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    def _log_status(self):
        """Log current bot status"""
        
        try:
            uptime = datetime.now() - self.start_time
            portfolio_value = self.account.get('portfolio_value', 0)
            
            logger.info(f"📊 Bot Status - Uptime: {uptime}, Portfolio: ${portfolio_value:.2f}, "
                       f"Positions: {len(self.positions)}, Daily P&L: ${self.daily_stats['pnl']:.2f}, "
                       f"Trades: {self.daily_stats['trades']}")
                       
        except Exception as e:
            logger.error(f"Status logging error: {e}")
    
    def _log_trade_execution(self, symbol: str, signal: Dict, risk_params: Dict, execution_result):
        """Log detailed trade execution"""
        
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'OPEN',
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'quantity': execution_result.filled_quantity,
                'price': execution_result.filled_price,
                'stop_loss': risk_params.get('stop_loss'),
                'take_profit': risk_params.get('take_profit'),
                'order_id': execution_result.order_id
            }
            
            # Save to file for analysis
            with open('trade_log.json', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
                
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    def _log_position_close(self, symbol: str, position: Dict, execution_result, reason: str, pnl: float):
        """Log position close details"""
        
        try:
            close_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'CLOSE',
                'reason': reason,
                'entry_price': position['entry_price'],
                'exit_price': execution_result.filled_price,
                'quantity': execution_result.filled_quantity,
                'hold_time': str(datetime.now() - position['entry_time']),
                'pnl': pnl,
                'pnl_percent': (pnl / (position['entry_price'] * position['quantity'])) * 100
            }
            
            # Save to file for analysis
            with open('trade_log.json', 'a') as f:
                f.write(json.dumps(close_log) + '\n')
                
        except Exception as e:
            logger.error(f"Position close logging error: {e}")
    
    def _print_startup_status(self):
        """Print bot startup status"""
        
        print("\n" + "=" * 60)
        print("🤖 OMNI ALPHA 5.0 - TRADING BOT STATUS")
        print("=" * 60)
        
        print(f"\n🚀 System Configuration:")
        print(f"   Trading Mode: {self.config['TRADING_MODE']}")
        print(f"   Simulation Mode: {self.config['SIMULATION_MODE']}")
        print(f"   Scan Symbols: {len(self.config['SCAN_SYMBOLS'])} symbols")
        print(f"   Scan Interval: {self.config['SCAN_INTERVAL']} seconds")
        
        print(f"\n💰 Account Information:")
        print(f"   Portfolio Value: ${self.account.get('portfolio_value', 0):,.2f}")
        print(f"   Buying Power: ${self.account.get('buying_power', 0):,.2f}")
        print(f"   Current Positions: {len(self.positions)}")
        
        print(f"\n🛡️ Risk Management:")
        print(f"   Max Daily Loss: ${self.config['MAX_DAILY_LOSS']:,}")
        print(f"   Max Position Size: ${self.config['MAX_POSITION_SIZE']:,}")
        print(f"   Max Positions: {self.config['MAX_POSITIONS']}")
        print(f"   Min Signal Confidence: {self.config['MIN_SIGNAL_CONFIDENCE']:.1%}")
        
        print(f"\n📈 Strategy Configuration:")
        strategy_status = self.strategies.get_strategy_performance()
        print(f"   Active Strategies: {len(strategy_status['strategies'])}")
        print(f"   Strategy Weights: {strategy_status['weights']}")
        
        print(f"\n🔧 Component Status:")
        print(f"   Database: {'✅ Connected' if self.database else '❌ Not available'}")
        print(f"   Data Collector: {'✅ Active' if self.data_collector else '❌ Not available'}")
        print(f"   Risk Manager: ✅ Active")
        print(f"   Order Executor: ✅ Active ({self.executor.simulation_mode and 'Simulation' or 'Live'})")
        
        print("\n" + "=" * 60)
        print("✅ OMNI ALPHA 5.0 TRADING BOT IS READY FOR OPERATION!")
        print("🎯 Starting automated trading with comprehensive risk management...")
        print("=" * 60 + "\n")
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        
        try:
            uptime = datetime.now() - self.start_time
            
            # Calculate performance metrics
            total_trades = self.daily_stats['wins'] + self.daily_stats['losses']
            win_rate = self.daily_stats['wins'] / total_trades if total_trades > 0 else 0.0
            
            return {
                'system': {
                    'uptime_seconds': uptime.total_seconds(),
                    'trading_mode': self.config['TRADING_MODE'],
                    'simulation_mode': self.config['SIMULATION_MODE'],
                    'running': self.running
                },
                'account': self.account,
                'positions': {
                    'count': len(self.positions),
                    'symbols': list(self.positions.keys()),
                    'total_value': sum(pos.get('quantity', 0) * pos.get('current_price', 0) 
                                     for pos in self.positions.values())
                },
                'daily_stats': {
                    **self.daily_stats,
                    'win_rate': win_rate,
                    'total_trades': total_trades
                },
                'risk_status': self.risk_manager.get_risk_status(),
                'execution_stats': self.executor.get_execution_statistics(),
                'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None
            }
            
        except Exception as e:
            logger.error(f"Status report error: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("🛑 Initiating trading bot shutdown...")
        self.running = False
        
        try:
            # Cancel all open orders
            cancelled_orders = await self.executor.cancel_all_orders()
            logger.info(f"Cancelled {cancelled_orders} open orders")
            
            # Close data collector
            if self.data_collector:
                await self.data_collector.close()
            
            # Close database connection
            if self.database:
                await self.database.close()
            
            # Save final performance log
            final_report = self.get_status_report()
            with open(f'final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            logger.info("✅ Trading bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Main execution
async def main():
    """Main execution function"""
    
    # Configuration override for testing
    config_override = {
        'SIMULATION_MODE': True,  # Set to False for live trading
        'SCAN_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
        'SCAN_INTERVAL': 60,
        'MAX_DAILY_LOSS': 500,
        'MAX_POSITIONS': 3,
        'MIN_SIGNAL_CONFIDENCE': 0.65
    }
    
    bot = TradingBot(config_override)
    
    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error(f"Bot execution error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

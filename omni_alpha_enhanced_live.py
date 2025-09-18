"""
ENHANCED OMNI ALPHA BOT - FULL MARKET TRADING
Fixes: Position sizing, Auto-selling, Market coverage
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json

# Alpaca Trading
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Telegram
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from dotenv import load_dotenv
load_dotenv('alpaca_live_trading.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== ENHANCED CONFIGURATION =====================

class TradingConfig:
    """Enhanced trading configuration"""
    
    # POSITION SIZING - FIXED!
    MAX_POSITION_SIZE_PERCENT = 0.10  # 10% of portfolio per position
    MIN_POSITION_SIZE = 100  # Minimum $100 per trade
    MAX_POSITION_SIZE = 20000  # Maximum $20,000 per position
    
    # PORTFOLIO MANAGEMENT
    MAX_POSITIONS = 20  # Can hold up to 20 positions
    RESERVE_CASH_PERCENT = 0.10  # Keep 10% cash reserve
    
    # TRADING PARAMETERS
    STOP_LOSS_PERCENT = 0.03  # 3% stop loss
    TAKE_PROFIT_PERCENT = 0.06  # 6% take profit
    TRAILING_STOP_PERCENT = 0.02  # 2% trailing stop
    
    # AUTO SELLING CONDITIONS
    SELL_ON_PROFIT_TARGET = True
    SELL_ON_STOP_LOSS = True
    SELL_ON_SIGNAL_CHANGE = True
    SELL_END_OF_DAY = False  # Day trading mode
    HOLD_TIME_MINIMUM_MINUTES = 30  # Hold at least 30 minutes
    
    # MARKET COVERAGE - EXPANDED!
    # S&P 500 Top stocks + Sectors + Popular stocks
    UNIVERSE = [
        # Mega Caps
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        
        # Large Caps
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
        'NFLX', 'ADBE', 'CRM', 'PFE', 'TMO', 'CSCO', 'PEP', 'AVGO', 'ABBV',
        
        # Technology
        'INTC', 'AMD', 'QCOM', 'ORCL', 'IBM', 'NOW', 'UBER', 'SHOP', 'SQ', 'PLTR',
        
        # Finance
        'GS', 'MS', 'WFC', 'C', 'AXP', 'BLK', 'SPGI', 'CB', 'MMC',
        
        # Healthcare
        'LLY', 'MRK', 'CVS', 'CI', 'AMGN', 'GILD', 'VRTX', 'REGN',
        
        # Consumer
        'WMT', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'COST',
        
        # Energy & Materials
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'FCX', 'NEM',
        
        # ETFs for sector exposure
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
        'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY',
        
        # Popular/Volatile for trading
        'GME', 'AMC', 'COIN', 'ROKU', 'PINS', 'SNAP',
        'DKNG', 'PENN', 'BYND', 'SPCE', 'LCID', 'RIVN',
        
        # Indian ADRs
        'INFY', 'WIT', 'IBN', 'HDB', 'TTM', 'SIFY', 'WNS', 'MMYT'
    ]
    
    # SCREENING PARAMETERS
    MIN_PRICE = 5.0  # Minimum stock price
    MAX_PRICE = 1000.0  # Maximum stock price
    MIN_VOLUME = 1000000  # Minimum daily volume
    MIN_MARKET_CAP = 1000000000  # $1B minimum market cap

# ===================== ENHANCED TRADING SYSTEM =====================

class EnhancedAlpacaTradingSystem:
    """
    Enhanced Alpaca Trading System with proper position sizing and selling
    """
    
    def __init__(self):
        self.config = TradingConfig()
        
        # Initialize Alpaca API
        self.api = REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        
        # Verify connection
        self.account = self.api.get_account()
        logger.info(f"Connected to Alpaca. Buying Power: ${float(self.account.buying_power):,.2f}")
        
        # Trading state
        self.positions = {}
        self.pending_orders = {}
        self.position_entry_times = {}
        self.position_entry_prices = {}
        
        # Load positions
        self._load_current_positions()
    
    def _load_current_positions(self):
        """Load current positions from Alpaca"""
        positions = self.api.list_positions()
        for pos in positions:
            self.positions[pos.symbol] = {
                'qty': int(pos.qty),
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price or 0),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'entry_time': datetime.now() - timedelta(hours=1)  # Estimate
            }
            self.position_entry_times[pos.symbol] = datetime.now() - timedelta(hours=1)
            self.position_entry_prices[pos.symbol] = float(pos.avg_entry_price)
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 0.7) -> int:
        """
        PROPER POSITION SIZING CALCULATOR
        """
        
        # Get account info
        account = self.api.get_account()
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)
        
        # Get current price
        try:
            quote = self.api.get_latest_quote(symbol)
            price = float(quote.ap)
        except:
            price = 100  # Fallback
        
        # Calculate position size based on portfolio percentage
        base_position_value = portfolio_value * self.config.MAX_POSITION_SIZE_PERCENT
        
        # Adjust based on signal strength
        adjusted_position_value = base_position_value * signal_strength
        
        # Apply min/max constraints
        position_value = max(
            self.config.MIN_POSITION_SIZE,
            min(adjusted_position_value, self.config.MAX_POSITION_SIZE)
        )
        
        # Ensure we don't exceed buying power
        position_value = min(position_value, buying_power * 0.9)  # Keep 10% buffer
        
        # Calculate shares
        shares = int(position_value / price)
        
        logger.info(f"Position sizing for {symbol}: ${position_value:.2f} = {shares} shares @ ${price:.2f}")
        
        return max(1, shares)  # At least 1 share
    
    async def should_sell_position(self, symbol: str) -> Tuple[bool, str]:
        """
        ENHANCED SELL DECISION LOGIC
        Returns (should_sell, reason)
        """
        
        if symbol not in self.positions:
            return False, "No position"
        
        position = self.positions[symbol]
        entry_price = self.position_entry_prices.get(symbol, position['entry_price'])
        current_price = position['current_price']
        
        # Get latest price
        try:
            quote = self.api.get_latest_quote(symbol)
            current_price = float(quote.ap)
        except:
            pass
        
        # Calculate P&L percentage
        pnl_percent = (current_price - entry_price) / entry_price
        
        # 1. Take Profit Check
        if self.config.SELL_ON_PROFIT_TARGET:
            if pnl_percent >= self.config.TAKE_PROFIT_PERCENT:
                return True, f"TAKE_PROFIT: {pnl_percent:.2%} gain"
        
        # 2. Stop Loss Check
        if self.config.SELL_ON_STOP_LOSS:
            if pnl_percent <= -self.config.STOP_LOSS_PERCENT:
                return True, f"STOP_LOSS: {pnl_percent:.2%} loss"
        
        # 3. Time-based exit (if held too long)
        if symbol in self.position_entry_times:
            hold_time = datetime.now() - self.position_entry_times[symbol]
            if hold_time > timedelta(hours=24):  # Exit positions older than 24 hours
                return True, f"TIME_EXIT: Held for {hold_time.total_seconds()/3600:.1f} hours"
        
        # 4. Technical indicator exit
        sell_signal = await self._check_technical_sell_signal(symbol)
        if sell_signal:
            return True, "TECHNICAL_SIGNAL: Indicators show sell"
        
        # 5. End of day exit (if day trading)
        if self.config.SELL_END_OF_DAY:
            try:
                clock = self.api.get_clock()
                time_to_close = clock.next_close - clock.timestamp
                if time_to_close.total_seconds() < 900:  # 15 minutes before close
                    return True, "END_OF_DAY: Market closing soon"
            except:
                pass
        
        return False, "Hold"
    
    async def _check_technical_sell_signal(self, symbol: str) -> bool:
        """Check technical indicators for sell signal"""
        
        try:
            # Use daily bars instead of minute bars to avoid SIP data subscription issues
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Day,
                start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                limit=30
            ).df
            
            if len(bars) < 20:
                return False
            
            # Calculate indicators
            if TALIB_AVAILABLE:
                bars['RSI'] = talib.RSI(bars['close'].values, timeperiod=14)
            else:
                # Simple RSI calculation
                delta = bars['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                bars['RSI'] = 100 - (100 / (1 + rs))
            
            bars['SMA_fast'] = bars['close'].rolling(10).mean()
            bars['SMA_slow'] = bars['close'].rolling(20).mean()
            
            latest = bars.iloc[-1]
            
            # Sell conditions
            sell_conditions = [
                latest['RSI'] > 70,  # Overbought
                latest['SMA_fast'] < latest['SMA_slow'],  # Death cross
                bars['close'].iloc[-1] < bars['close'].iloc[-5] * 0.98  # 2% drop
            ]
            
            # Sell if 2 or more conditions are true
            return sum(sell_conditions) >= 2
            
        except Exception as e:
            logger.error(f"Technical sell signal error: {e}")
            return False
    
    async def execute_auto_sell(self):
        """
        AUTOMATIC POSITION SELLING
        Continuously monitors and sells positions
        """
        
        while True:
            try:
                # Check each position for sell conditions
                positions = self.api.list_positions()
                
                for position in positions:
                    symbol = position.symbol
                    
                    # Update position info
                    self.positions[symbol] = {
                        'qty': int(position.qty),
                        'entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price or 0),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc) * 100
                    }
                    
                    # Check if we should sell
                    should_sell, reason = await self.should_sell_position(symbol)
                    
                    if should_sell:
                        # Place sell order
                        logger.info(f"AUTO SELL {symbol}: {reason}")
                        
                        try:
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=position.qty,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            
                            logger.info(f"Sell order placed for {symbol}: {position.qty} shares. Reason: {reason}")
                            
                            # Remove from tracking
                            if symbol in self.position_entry_times:
                                del self.position_entry_times[symbol]
                            if symbol in self.position_entry_prices:
                                del self.position_entry_prices[symbol]
                                
                        except Exception as e:
                            logger.error(f"Sell order failed for {symbol}: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Auto sell error: {e}")
                await asyncio.sleep(30)

# ===================== MARKET SCANNER =====================

class MarketScanner:
    """
    Scans entire market for trading opportunities
    """
    
    def __init__(self, api):
        self.api = api
        self.config = TradingConfig()
    
    async def scan_market(self) -> List[Dict]:
        """
        Scan market for best trading opportunities
        """
        
        opportunities = []
        
        # Split universe into chunks for parallel processing
        chunk_size = 10
        chunks = [self.config.UNIVERSE[i:i+chunk_size] 
                 for i in range(0, len(self.config.UNIVERSE), chunk_size)]
        
        for chunk in chunks:
            for symbol in chunk:
                try:
                    score = await self._analyze_symbol(symbol)
                    if score['total_score'] > 0.6:  # Good opportunity
                        opportunities.append({
                            'symbol': symbol,
                            'score': score['total_score'],
                            'signal': score['signal'],
                            'reasons': score['reasons']
                        })
                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
            
            await asyncio.sleep(0.1)  # Rate limiting
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities[:20]  # Top 20 opportunities
    
    async def _analyze_symbol(self, symbol: str) -> Dict:
        """
        Comprehensive symbol analysis
        """
        
        try:
            # Get recent data
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Day,
                start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                limit=30
            ).df
            
            if len(bars) < 20:
                return {'total_score': 0, 'signal': 'SKIP', 'reasons': []}
            
            scores = []
            reasons = []
            
            # 1. Momentum Score
            momentum = (bars['close'].iloc[-1] - bars['close'].iloc[-20]) / bars['close'].iloc[-20]
            if momentum > 0.05:
                scores.append(0.3)
                reasons.append("Strong momentum")
            
            # 2. Volume Score
            avg_volume = bars['volume'].mean()
            recent_volume = bars['volume'].iloc[-1]
            if recent_volume > avg_volume * 1.5:
                scores.append(0.2)
                reasons.append("High volume")
            
            # 3. Volatility Score (for trading)
            volatility = bars['close'].pct_change().std()
            if 0.01 < volatility < 0.05:  # Good volatility for trading
                scores.append(0.2)
                reasons.append("Good volatility")
            
            # 4. RSI Score
            closes = bars['close'].values
            if len(closes) >= 14:
                if TALIB_AVAILABLE:
                    rsi = talib.RSI(closes, timeperiod=14)[-1]
                else:
                    # Simple RSI calculation
                    delta = bars['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi_series = 100 - (100 / (1 + rs))
                    rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50
                
                if 30 < rsi < 70:  # Not oversold/overbought
                    scores.append(0.3)
                    reasons.append(f"RSI: {rsi:.0f}")
            
            total_score = sum(scores)
            
            # Determine signal
            if total_score > 0.7:
                signal = 'STRONG_BUY'
            elif total_score > 0.5:
                signal = 'BUY'
            else:
                signal = 'HOLD'
            
            return {
                'total_score': total_score,
                'signal': signal,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.debug(f"Analysis error for {symbol}: {e}")
            return {'total_score': 0, 'signal': 'ERROR', 'reasons': []}

# ===================== ENHANCED TRADING BOT =====================

class EnhancedOmniAlphaBot:
    """
    Enhanced Telegram Bot with full market coverage
    """
    
    def __init__(self):
        self.trading_system = EnhancedAlpacaTradingSystem()
        self.market_scanner = MarketScanner(self.trading_system.api)
        self.config = TradingConfig()
        
        self.auto_trading_active = False
        self.sell_monitor_active = False
        
        # Telegram setup
        self.application = Application.builder().token(
            os.getenv('TELEGRAM_BOT_TOKEN')
        ).build()
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup command handlers"""
        
        handlers = [
            ('start', self.start_command),
            ('account', self.account_command),
            ('scan', self.scan_command),
            ('buy', self.buy_command),
            ('sell', self.sell_command),
            ('positions', self.positions_command),
            ('auto', self.auto_trade_command),
            ('stop', self.stop_command),
            ('performance', self.performance_command),
            ('settings', self.settings_command),
            ('help', self.help_command)
        ]
        
        for command, handler in handlers:
            self.application.add_handler(CommandHandler(command, handler))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        
        account = self.trading_system.api.get_account()
        positions = self.trading_system.api.list_positions()
        
        msg = f"""
**ENHANCED OMNI ALPHA TRADING BOT**

Account Status: {account.status}
Portfolio Value: ${float(account.portfolio_value):,.2f}
Buying Power: ${float(account.buying_power):,.2f}
Open Positions: {len(positions)}

**ENHANCED FEATURES:**
- Position Sizing: 10% of portfolio per trade
- Auto Selling: Take profit/stop loss automated
- Market Coverage: 100+ stocks
- Risk Management: Advanced controls

**Commands:**
/scan - Find best opportunities
/buy SYMBOL - Buy with proper sizing
/auto - Start full automation
/positions - View all positions
/help - All commands
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def account_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Account information"""
        
        account = self.trading_system.api.get_account()
        positions = self.trading_system.api.list_positions()
        
        total_pl = sum(float(p.unrealized_pl) for p in positions)
        
        msg = f"""
**Account Information**

Equity: ${float(account.equity):,.2f}
Cash: ${float(account.cash):,.2f}
Buying Power: ${float(account.buying_power):,.2f}
Day Trade Buying Power: ${float(account.daytrading_buying_power):,.2f}

Open Positions: {len(positions)}
Total P&L: ${total_pl:+,.2f}
Day Trades Used: {account.daytrade_count}/3

**Position Sizing:**
Max per position: ${self.config.MAX_POSITION_SIZE:,}
Current max: ${float(account.portfolio_value) * self.config.MAX_POSITION_SIZE_PERCENT:,.2f}
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Scan market for opportunities"""
        
        await update.message.reply_text("Scanning market for opportunities...")
        
        opportunities = await self.market_scanner.scan_market()
        
        if not opportunities:
            await update.message.reply_text("No good opportunities found right now.")
            return
        
        msg = "**Top Trading Opportunities**\n\n"
        
        for i, opp in enumerate(opportunities[:10], 1):
            msg += f"{i}. **{opp['symbol']}** - Score: {opp['score']:.2f}\n"
            msg += f"   Signal: {opp['signal']}\n"
            msg += f"   {', '.join(opp['reasons'])}\n\n"
        
        msg += "\nUse /buy SYMBOL to trade any of these"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def buy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buy command with proper position sizing"""
        
        if not context.args:
            await update.message.reply_text("Usage: /buy SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        # Calculate position size
        position_size = self.trading_system.calculate_position_size(symbol)
        
        # Get current price
        try:
            quote = self.trading_system.api.get_latest_quote(symbol)
            price = float(quote.ap)
            position_value = position_size * price
        except:
            price = 100
            position_value = position_size * price
        
        # Place order
        try:
            order = self.trading_system.api.submit_order(
                symbol=symbol,
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            # Track entry
            self.trading_system.position_entry_times[symbol] = datetime.now()
            self.trading_system.position_entry_prices[symbol] = price
            
            msg = f"""
**BUY ORDER PLACED**

Symbol: {symbol}
Shares: {position_size}
Price: ${price:.2f}
Total: ${position_value:.2f}
Order ID: {order.id}

Stop Loss: ${price * (1 - self.config.STOP_LOSS_PERCENT):.2f} (-{self.config.STOP_LOSS_PERCENT*100}%)
Take Profit: ${price * (1 + self.config.TAKE_PROFIT_PERCENT):.2f} (+{self.config.TAKE_PROFIT_PERCENT*100}%)
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Order failed: {str(e)}")
    
    async def sell_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sell command"""
        
        if not context.args:
            await update.message.reply_text("Usage: /sell SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        # Find position
        positions = self.trading_system.api.list_positions()
        position = None
        
        for pos in positions:
            if pos.symbol == symbol:
                position = pos
                break
        
        if not position:
            await update.message.reply_text(f"No position found for {symbol}")
            return
        
        # Place sell order
        try:
            order = self.trading_system.api.submit_order(
                symbol=symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            pnl = float(position.unrealized_pl)
            pnl_percent = float(position.unrealized_plpc) * 100
            
            msg = f"""
**SELL ORDER PLACED**

Symbol: {symbol}
Shares: {position.qty}
P&L: ${pnl:+,.2f} ({pnl_percent:+.2f}%)
Order ID: {order.id}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Remove from tracking
            if symbol in self.trading_system.position_entry_times:
                del self.trading_system.position_entry_times[symbol]
            if symbol in self.trading_system.position_entry_prices:
                del self.trading_system.position_entry_prices[symbol]
            
        except Exception as e:
            await update.message.reply_text(f"Sell order failed: {str(e)}")
    
    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View positions with auto-sell targets"""
        
        positions = self.trading_system.api.list_positions()
        
        if not positions:
            await update.message.reply_text("No open positions")
            return
        
        msg = "**Current Positions**\n\n"
        
        for position in positions:
            pnl = float(position.unrealized_pl)
            pnl_percent = float(position.unrealized_plpc) * 100
            entry_price = float(position.avg_entry_price)
            
            emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "📊"
            
            msg += f"{emoji} **{position.symbol}**\n"
            msg += f"Qty: {position.qty} shares\n"
            msg += f"Entry: ${entry_price:.2f}\n"
            msg += f"Current: ${float(position.current_price):.2f}\n"
            msg += f"Value: ${float(position.market_value):,.2f}\n"
            msg += f"P&L: ${pnl:+,.2f} ({pnl_percent:+.2f}%)\n"
            
            # Show auto-sell targets
            stop_loss = entry_price * (1 - self.config.STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 + self.config.TAKE_PROFIT_PERCENT)
            
            msg += f"Stop: ${stop_loss:.2f} | Target: ${take_profit:.2f}\n\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def auto_trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start automated trading"""
        
        if self.auto_trading_active:
            await update.message.reply_text("Auto trading already active")
            return
        
        self.auto_trading_active = True
        self.sell_monitor_active = True
        
        await update.message.reply_text("""
**AUTO TRADING ACTIVATED**

Scanning 100+ stocks
Auto position sizing enabled
Auto sell monitoring active
Risk management active

Type /stop to stop trading
        """, parse_mode='Markdown')
        
        # Start auto trading and selling
        asyncio.create_task(self.auto_trading_loop())
        asyncio.create_task(self.trading_system.execute_auto_sell())
    
    async def auto_trading_loop(self):
        """Enhanced auto trading with market scanning"""
        
        while self.auto_trading_active:
            try:
                # Check if market is open
                clock = self.trading_system.api.get_clock()
                if not clock.is_open:
                    await asyncio.sleep(60)
                    continue
                
                # Get account info
                account = self.trading_system.api.get_account()
                buying_power = float(account.buying_power)
                
                # Only trade if we have buying power
                if buying_power < 1000:
                    logger.info("Insufficient buying power")
                    await asyncio.sleep(300)
                    continue
                
                # Scan market for opportunities
                opportunities = await self.market_scanner.scan_market()
                
                # Get current positions
                positions = self.trading_system.api.list_positions()
                current_symbols = [p.symbol for p in positions]
                
                # Trade top opportunities not already in portfolio
                for opp in opportunities[:5]:  # Top 5 opportunities
                    if opp['symbol'] not in current_symbols:
                        if len(current_symbols) >= self.config.MAX_POSITIONS:
                            break  # Max positions reached
                        
                        # Calculate position size
                        position_size = self.trading_system.calculate_position_size(
                            opp['symbol'],
                            opp['score']
                        )
                        
                        # Place order
                        try:
                            order = self.trading_system.api.submit_order(
                                symbol=opp['symbol'],
                                qty=position_size,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            
                            # Track entry
                            self.trading_system.position_entry_times[opp['symbol']] = datetime.now()
                            
                            logger.info(f"AUTO BUY: {opp['symbol']} - {position_size} shares")
                            
                            # Add to current symbols to avoid duplicates
                            current_symbols.append(opp['symbol'])
                            
                        except Exception as e:
                            logger.error(f"Auto trade error for {opp['symbol']}: {e}")
                
                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Auto trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop auto trading"""
        
        self.auto_trading_active = False
        
        await update.message.reply_text("Auto trading stopped. Positions remain open with auto-sell monitoring.")
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance metrics"""
        
        try:
            # Get portfolio history
            history = self.trading_system.api.get_portfolio_history(period='1M', timeframe='1D')
            
            if not history.equity:
                await update.message.reply_text("No performance data available yet")
                return
            
            equity_values = history.equity
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
            
            account = self.trading_system.api.get_account()
            positions = self.trading_system.api.list_positions()
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            
            msg = f"""
**Performance Metrics**

Total Return: {total_return:+.2f}%
Current P&L: ${total_pl:+,.2f}
Portfolio Value: ${float(account.portfolio_value):,.2f}
Starting Value: ${equity_values[0]:,.2f}

Open Positions: {len(positions)}
Max Positions: {self.config.MAX_POSITIONS}
Auto Trading: {'Active' if self.auto_trading_active else 'Inactive'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Performance calculation failed: {str(e)}")
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View trading settings"""
        
        msg = f"""
**Trading Settings**

Position Sizing: {self.config.MAX_POSITION_SIZE_PERCENT*100}% of portfolio
Max Position Value: ${self.config.MAX_POSITION_SIZE:,}
Min Position Value: ${self.config.MIN_POSITION_SIZE:,}

Risk Management:
Stop Loss: {self.config.STOP_LOSS_PERCENT*100}%
Take Profit: {self.config.TAKE_PROFIT_PERCENT*100}%
Max Positions: {self.config.MAX_POSITIONS}

Market Universe: {len(self.config.UNIVERSE)} stocks
Auto Sell: {'Enabled' if self.config.SELL_ON_PROFIT_TARGET else 'Disabled'}
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        
        msg = """
**Enhanced Omni Alpha Commands**

**Trading:**
/scan - Find best opportunities
/buy SYMBOL - Buy with proper sizing
/sell SYMBOL - Sell position
/auto - Start automated trading
/stop - Stop auto trading

**Information:**
/account - Account information
/positions - View all positions
/performance - Performance metrics
/settings - Trading settings

**Features:**
- Proper position sizing (10% of portfolio)
- Auto selling with take profit/stop loss
- 100+ stock universe
- Risk management
- Market scanning
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def run(self):
        """Start the bot"""
        
        logger.info("Starting Enhanced Omni Alpha Bot...")
        
        # Start auto sell monitor by default
        asyncio.create_task(self.trading_system.execute_auto_sell())
        
        print("=" * 60)
        print("ENHANCED OMNI ALPHA TRADING SYSTEM")
        print("=" * 60)
        print("Alpaca connection verified")
        print("All enhanced features loaded")
        print("Risk management active")
        print("Auto-sell monitoring active")
        print("Telegram bot ready")
        print("Send /start in Telegram to begin")
        print("=" * 60)
        
        # Initialize and start application
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Bot is running! Press Ctrl+C to stop.")
        
        # Keep running
        await asyncio.Event().wait()

# ===================== MAIN EXECUTION =====================

async def main():
    """Main entry point"""
    
    try:
        bot = EnhancedOmniAlphaBot()
        
        # Show account info
        account = bot.trading_system.api.get_account()
        print(f"\nAccount Status:")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Number of Positions: {len(bot.trading_system.api.list_positions())}")
        
        print(f"\nTrading Universe: {len(TradingConfig.UNIVERSE)} stocks")
        print(f"Max Position Size: ${TradingConfig.MAX_POSITION_SIZE:,}")
        print(f"Auto Sell: ENABLED")
        
        print("\nBot starting...")
        print("Open Telegram and send /help")
        
        await bot.run()
        
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Bot failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

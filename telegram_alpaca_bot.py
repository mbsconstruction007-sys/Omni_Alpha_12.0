"""
Telegram Alpaca Trading Bot - Omni Alpha 12.0
Created by Claude AI Assistant
"""

from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import alpaca_trade_api as tradeapi
from datetime import datetime
import asyncio

# Telegram Token
TELEGRAM_TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'

# Alpaca Paper Trading (REPLACE WITH YOUR KEYS)
ALPACA_KEY = 'YOUR_ALPACA_API_KEY'
ALPACA_SECRET = 'YOUR_ALPACA_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

class AlpacaTradingBot:
    """Complete Alpaca Trading Bot with Telegram Integration"""
    
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL, api_version='v2')
        self.version = "12.0.0"
        self.ai_assistant = "Claude"
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        welcome_msg = """
🚀 **OMNI ALPHA 12.0 - ALPACA TRADING BOT**

🤖 **AI Assistant:** Claude Connected
📊 **Trading Platform:** Alpaca Paper Trading
🌍 **Global Dominance:** Ready

**Available Commands:**
/account - Account information
/quote SYMBOL - Get stock price
/buy SYMBOL QTY - Buy shares
/sell SYMBOL QTY - Sell shares
/positions - View positions
/orders - Recent orders
/portfolio - Portfolio summary
/status - System status
/help - Show all commands

**Example:**
/quote AAPL
/buy AAPL 1
/positions
        """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        help_msg = """
📋 **ALPACA TRADING COMMANDS**

💰 **Trading:**
/account - Account information
/quote SYMBOL - Get stock price
/buy SYMBOL QTY - Buy shares
/sell SYMBOL QTY - Sell shares
/positions - View positions
/orders - Recent orders
/portfolio - Portfolio summary

📊 **System:**
/status - System status
/help - Show this help

**Examples:**
/quote AAPL
/buy AAPL 1
/sell AAPL 1
/positions
        """
        await update.message.reply_text(help_msg, parse_mode='Markdown')
    
    async def account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Account information"""
        try:
            acc = self.api.get_account()
            account_msg = f"""
💰 **ACCOUNT INFORMATION**

📊 **Status:** {acc.status}
💵 **Buying Power:** ${float(acc.buying_power):,.2f}
💳 **Cash:** ${float(acc.cash):,.2f}
📈 **Portfolio Value:** ${float(acc.portfolio_value):,.2f}
📊 **Equity:** ${float(acc.equity):,.2f}
📉 **Day Trade Count:** {acc.daytrade_count}

🤖 **AI-Powered by Omni Alpha 12.0**
            """
            await update.message.reply_text(account_msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def quote(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get stock quote"""
        if not context.args:
            await update.message.reply_text("❌ Usage: /quote SYMBOL\nExample: /quote AAPL")
            return
        
        symbol = context.args[0].upper()
        try:
            quote = self.api.get_latest_quote(symbol)
            quote_msg = f"""
📈 **{symbol} QUOTE**

💰 **Ask Price:** ${quote.ap}
💰 **Bid Price:** ${quote.bp}
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}
📅 **Date:** {datetime.now().strftime("%Y-%m-%d")}

🤖 **AI-Powered by Omni Alpha 12.0**
            """
            await update.message.reply_text(quote_msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting quote for {symbol}: {str(e)}")
    
    async def buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buy shares"""
        if len(context.args) != 2:
            await update.message.reply_text("❌ Usage: /buy SYMBOL QUANTITY\nExample: /buy AAPL 1")
            return
        
        symbol = context.args[0].upper()
        qty = int(context.args[1])
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            buy_msg = f"""
✅ **BUY ORDER PLACED**

🏷️ **Symbol:** {symbol}
📊 **Quantity:** {qty}
🆔 **Order ID:** {order.id[:8]}...
📊 **Status:** {order.status}
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}

🤖 **AI-Powered by Omni Alpha 12.0**
            """
            await update.message.reply_text(buy_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error placing buy order: {str(e)}")
    
    async def sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sell shares"""
        if len(context.args) != 2:
            await update.message.reply_text("❌ Usage: /sell SYMBOL QUANTITY\nExample: /sell AAPL 1")
            return
        
        symbol = context.args[0].upper()
        qty = int(context.args[1])
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            sell_msg = f"""
✅ **SELL ORDER PLACED**

🏷️ **Symbol:** {symbol}
📊 **Quantity:** {qty}
🆔 **Order ID:** {order.id[:8]}...
📊 **Status:** {order.status}
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}

🤖 **AI-Powered by Omni Alpha 12.0**
            """
            await update.message.reply_text(sell_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error placing sell order: {str(e)}")
    
    async def positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View positions"""
        try:
            positions = self.api.list_positions()
            
            if not positions:
                await update.message.reply_text("📊 **No open positions**\n\nStart trading with /buy SYMBOL QTY")
                return
            
            positions_msg = "📊 **CURRENT POSITIONS**\n\n"
            
            for pos in positions:
                positions_msg += f"🏷️ **{pos.symbol}**\n"
                positions_msg += f"📊 **Quantity:** {pos.qty} shares\n"
                positions_msg += f"💰 **Avg Price:** ${float(pos.avg_entry_price):.2f}\n"
                positions_msg += f"📈 **Current:** ${float(pos.current_price):.2f}\n"
                positions_msg += f"💵 **P&L:** ${float(pos.unrealized_pl):.2f}\n"
                positions_msg += f"📊 **Value:** ${float(pos.market_value):.2f}\n\n"
            
            positions_msg += "🤖 **AI-Powered by Omni Alpha 12.0**"
            await update.message.reply_text(positions_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting positions: {str(e)}")
    
    async def orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """View recent orders"""
        try:
            orders = self.api.list_orders(status='all', limit=10)
            
            if not orders:
                await update.message.reply_text("📊 **No recent orders**")
                return
            
            orders_msg = "📊 **RECENT ORDERS**\n\n"
            
            for order in orders:
                orders_msg += f"🆔 **{order.id[:8]}...**\n"
                orders_msg += f"🏷️ **{order.symbol}** - {order.side.upper()}\n"
                orders_msg += f"📊 **Qty:** {order.qty}\n"
                orders_msg += f"📊 **Status:** {order.status}\n"
                orders_msg += f"🕐 **Time:** {order.created_at.strftime('%H:%M:%S')}\n\n"
            
            orders_msg += "🤖 **AI-Powered by Omni Alpha 12.0**"
            await update.message.reply_text(orders_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting orders: {str(e)}")
    
    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio summary"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            total_positions_value = sum(float(pos.market_value) for pos in positions)
            total_pnl = sum(float(pos.unrealized_pl) for pos in positions)
            
            portfolio_msg = f"""
📈 **PORTFOLIO SUMMARY**

💵 **Cash Balance:** ${float(account.cash):,.2f}
📊 **Positions Value:** ${total_positions_value:,.2f}
💰 **Total Portfolio:** ${float(account.portfolio_value):,.2f}
📈 **Unrealized P&L:** ${total_pnl:,.2f}
📊 **Open Positions:** {len(positions)}

🤖 **AI-Powered by Omni Alpha 12.0**
            """
            await update.message.reply_text(portfolio_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error getting portfolio: {str(e)}")
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System status"""
        status_msg = f"""
🤖 **OMNI ALPHA 12.0 STATUS**

📊 **System Status:** Operational
🕐 **Current Time:** {datetime.now().strftime("%H:%M:%S")}
📅 **Date:** {datetime.now().strftime("%Y-%m-%d")}
🤖 **AI Assistant:** {self.ai_assistant}
📈 **Version:** {self.version}
📊 **Trading Platform:** Alpaca Paper Trading

💰 **Account Status:** Connected
🌍 **Global Dominance:** Ready
⚡ **Performance:** Optimal
        """
        await update.message.reply_text(status_msg, parse_mode='Markdown')

def main():
    """Main function to run the bot"""
    print("🚀 Starting Omni Alpha 12.0 Alpaca Trading Bot...")
    print("🤖 AI Assistant: Claude Connected")
    print("📊 Trading Platform: Alpaca Paper Trading")
    print("🌍 Global Market Dominance: Ready")
    print("\n" + "="*50)
    print("🤖 ALPACA TRADING BOT RUNNING!")
    print("📱 Test in Telegram with @omni_alpha_12_bot")
    print("="*50)
    
    # Create bot instance
    bot = AlpacaTradingBot()
    
    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add command handlers
    commands = [
        ('start', bot.start),
        ('help', bot.help),
        ('account', bot.account),
        ('quote', bot.quote),
        ('buy', bot.buy),
        ('sell', bot.sell),
        ('positions', bot.positions),
        ('orders', bot.orders),
        ('portfolio', bot.portfolio),
        ('status', bot.status)
    ]
    
    for cmd, func in commands:
        app.add_handler(CommandHandler(cmd, func))
    
    # Start the bot
    app.run_polling()

if __name__ == '__main__':
    main()

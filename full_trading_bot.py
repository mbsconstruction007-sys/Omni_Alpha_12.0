# full_trading_bot.py
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from datetime import datetime
import random

TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'

class TradingBot:
    def __init__(self):
        self.positions = {}
        self.balance = 100000
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            'Welcome to Omni Alpha Trading Bot!\n\n'
            'Commands:\n'
            '/balance - Check balance\n'
            '/buy [symbol] [qty] - Buy stocks\n'
            '/sell [symbol] [qty] - Sell stocks\n'
            '/positions - View positions\n'
            '/market - Market overview'
        )
    
    async def balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f'Balance: ')
    
    async def buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) != 2:
            await update.message.reply_text('Usage: /buy SYMBOL QUANTITY')
            return
        
        symbol = context.args[0].upper()
        qty = int(context.args[1])
        price = random.uniform(50, 200)
        cost = price * qty
        
        if cost > self.balance:
            await update.message.reply_text('Insufficient funds!')
            return
        
        self.balance -= cost
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += qty
        
        await update.message.reply_text(
            f'Bought {qty} {symbol} @ \n'
            f'Total: \n'
            f'Balance: '
        )
    
    async def positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.positions:
            await update.message.reply_text('No positions')
            return
        
        msg = 'Positions:\n'
        for symbol, qty in self.positions.items():
            msg += f'{symbol}: {qty} shares\n'
        await update.message.reply_text(msg)
    
    async def market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f'Market Overview {datetime.now().strftime("%H:%M")}\n\n'
            f'SPY: +1.2%\n'
            f'QQQ: +1.5%\n'
            f'DIA: +0.8%\n'
            f'VIX: 15.23'
        )

bot = TradingBot()
app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler('start', bot.start))
app.add_handler(CommandHandler('balance', bot.balance))
app.add_handler(CommandHandler('buy', bot.buy))
app.add_handler(CommandHandler('positions', bot.positions))
app.add_handler(CommandHandler('market', bot.market))

print('Trading bot running!')
app.run_polling()

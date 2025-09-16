# omni_alpha_telegram_bot.py
'''Omni Alpha 12.0 - Telegram Trading Bot'''

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Load environment variables
load_dotenv()

# Get credentials from .env
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class OmniAlphaBot:
    def __init__(self):
        self.trading_active = False
        self.positions = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Welcome message with menu'''
        keyboard = [
            [InlineKeyboardButton("📊 Status", callback_data='status'),
             InlineKeyboardButton("💼 Portfolio", callback_data='portfolio')],
            [InlineKeyboardButton("📈 Start Trading", callback_data='start_trading'),
             InlineKeyboardButton("🛑 Stop Trading", callback_data='stop_trading')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_msg = f'''
🚀 **Omni Alpha 12.0 Trading Bot**

Welcome! I'm your AI-powered trading assistant.

Commands:
/status - System status
/portfolio - View portfolio
/positions - Current positions
/trade - Execute trade
/help - Show all commands
        '''
        await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Show system status'''
        status_msg = f'''
📊 **System Status**
━━━━━━━━━━━━━━━
🟢 Status: Online
⏰ Time: {datetime.now().strftime('%H:%M:%S')}
📈 Trading: {'Active' if self.trading_active else 'Inactive'}
🔥 Mode: Paper Trading
💻 Version: 12.0.0
🤖 AI: Operational
        '''
        await update.message.reply_text(status_msg, parse_mode='Markdown')
    
    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Show portfolio overview'''
        portfolio_msg = '''
💼 **Portfolio Overview**
━━━━━━━━━━━━━━━
💰 Balance: ,000
📊 Positions: 3 active
📈 Today P&L: +,234 (+1.23%)
🎯 Win Rate: 68%
📉 Max DD: -5.2%
        '''
        await update.message.reply_text(portfolio_msg, parse_mode='Markdown')
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Handle button clicks'''
        query = update.callback_query
        await query.answer()
        
        if query.data == 'status':
            await self.status(query, context)
        elif query.data == 'portfolio':
            await self.portfolio(query, context)
        elif query.data == 'start_trading':
            self.trading_active = True
            await query.edit_message_text("✅ Trading Started!")
        elif query.data == 'stop_trading':
            self.trading_active = False
            await query.edit_message_text("🛑 Trading Stopped!")

# Initialize bot
bot = OmniAlphaBot()

def main():
    '''Run the bot'''
    # Create application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("status", bot.status))
    app.add_handler(CommandHandler("portfolio", bot.portfolio))
    app.add_handler(CallbackQueryHandler(bot.button_handler))
    
    # Start bot
    print('✅ Omni Alpha Telegram Bot Starting...')
    print(f'📱 Chat ID: {CHAT_ID}')
    print('Send /start to your bot to begin!')
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

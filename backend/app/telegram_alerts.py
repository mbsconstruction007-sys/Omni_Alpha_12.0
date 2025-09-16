# backend/app/telegram_alerts.py
'''Alert system for Telegram notifications'''

from datetime import datetime
import asyncio
from telegram import Bot

class TelegramAlerts:
    def __init__(self, bot_token: str, chat_id: int):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
    
    async def send_trade_alert(self, trade_info: dict):
        message = f'''
🔔 **Trade Alert**
Symbol: {trade_info.get('symbol')}
Action: {trade_info.get('action')}
Quantity: {trade_info.get('quantity')}
Price: 
Time: {datetime.now().strftime('%H:%M:%S')}
        '''
        await self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
    
    async def send_price_alert(self, symbol: str, price: float, condition: str):
        message = f'''
📊 **Price Alert**
{symbol} {condition} 
Time: {datetime.now().strftime('%H:%M:%S')}
        '''
        await self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
    
    async def send_error_alert(self, error: str):
        message = f'''
❌ **Error Alert**
{error}
Time: {datetime.now().strftime('%H:%M:%S')}
        '''
        await self.bot.send_message(self.chat_id, message, parse_mode='Markdown')

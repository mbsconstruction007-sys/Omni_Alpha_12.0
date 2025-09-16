"""
Alert Sender - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import asyncio
from telegram import Bot
from datetime import datetime

# Bot Configuration
TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'

# Replace with your actual chat ID (get it from get_my_chat_id.py)
CHAT_ID = 'YOUR_CHAT_ID'  # Update this with your actual chat ID

async def send_alert(message: str):
    """Send an alert message to Telegram"""
    try:
        bot = Bot(TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        print(f"✅ Alert sent successfully: {message}")
    except Exception as e:
        print(f"❌ Error sending alert: {str(e)}")

async def send_system_status():
    """Send system status alert"""
    status_message = f"""
🚀 **OMNI ALPHA 12.0 ALERT**

📊 **System Status:** Operational
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}
📅 **Date:** {datetime.now().strftime("%Y-%m-%d")}
🤖 **AI Assistant:** Claude Connected
🌍 **Global Dominance:** Ready

✅ **All systems operational!**
    """
    await send_alert(status_message)

async def send_trade_alert(symbol: str, action: str, quantity: int, price: float):
    """Send trade execution alert"""
    trade_message = f"""
📈 **TRADE ALERT**

🔄 **Action:** {action}
🏷️ **Symbol:** {symbol}
📊 **Quantity:** {quantity}
💰 **Price:** ${price:.2f}
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}

🤖 **AI-Powered by Omni Alpha 12.0**
    """
    await send_alert(trade_message)

async def send_error_alert(error_message: str):
    """Send error alert"""
    error_alert = f"""
⚠️ **SYSTEM ALERT**

❌ **Error:** {error_message}
🕐 **Time:** {datetime.now().strftime("%H:%M:%S")}
📅 **Date:** {datetime.now().strftime("%Y-%m-%d")}

🔧 **Action Required:** Check system logs
    """
    await send_alert(error_alert)

async def main():
    """Main function for testing alerts"""
    print("🚀 Omni Alpha 12.0 Alert System")
    print("="*50)
    
    if CHAT_ID == 'YOUR_CHAT_ID':
        print("❌ Please update CHAT_ID in this file first!")
        print("Run: python get_my_chat_id.py")
        return
    
    # Test different alert types
    print("📤 Sending test alerts...")
    
    # System status alert
    await send_system_status()
    await asyncio.sleep(1)
    
    # Trade alert
    await send_trade_alert("AAPL", "BUY", 10, 150.25)
    await asyncio.sleep(1)
    
    # Success message
    success_message = """
✅ **ALERT SYSTEM READY**

🤖 **Omni Alpha 12.0 Alert System**
📱 **Telegram Integration:** Connected
🚀 **Global Dominance:** Ready
🌍 **AI-Powered Trading:** Active

**Ready for global market dominance!**
    """
    await send_alert(success_message)
    
    print("✅ All test alerts sent successfully!")

if __name__ == '__main__':
    asyncio.run(main())

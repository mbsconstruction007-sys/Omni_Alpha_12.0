"""
Get Chat ID for Intelligent Trader - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import asyncio
from telegram import Bot

async def get_chat_id():
    """Get the chat ID for the intelligent trader"""
    TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
    
    try:
        bot = Bot(TOKEN)
        
        # Get bot info
        bot_info = await bot.get_me()
        print(f"🤖 Bot Info: {bot_info.first_name} (@{bot_info.username})")
        
        print("\n" + "="*50)
        print("📱 INSTRUCTIONS:")
        print("1. Go to Telegram")
        print("2. Search for your bot: @" + bot_info.username)
        print("3. Send any message to the bot (e.g., /start)")
        print("4. Wait 5 seconds, then press Enter here")
        print("="*50)
        
        input("Press Enter after sending a message to your bot...")
        
        # Get updates to find chat ID
        updates = await bot.get_updates()
        
        if updates:
            latest_update = updates[-1]
            if latest_update.message:
                chat_id = latest_update.message.chat_id
                user_info = latest_update.message.from_user
                
                print(f"\n✅ SUCCESS! Chat ID Found:")
                print(f"🆔 Your Chat ID: {chat_id}")
                print(f"👤 Your Name: {user_info.first_name} {user_info.last_name or ''}")
                
                # Update the intelligent trader file
                print(f"\n🔧 Updating intelligent_10min_trader.py...")
                
                with open('intelligent_10min_trader.py', 'r') as f:
                    content = f.read()
                
                # Replace the chat ID
                updated_content = content.replace("CHAT_ID = 'YOUR_CHAT_ID'", f"CHAT_ID = '{chat_id}'")
                
                with open('intelligent_10min_trader.py', 'w') as f:
                    f.write(updated_content)
                
                print("✅ Chat ID updated in intelligent_10min_trader.py")
                print(f"\n🚀 Ready to run: python intelligent_10min_trader.py")
                
                return chat_id
            else:
                print("❌ No message found in updates")
        else:
            print("❌ No updates found. Make sure you sent a message to the bot.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    asyncio.run(get_chat_id())

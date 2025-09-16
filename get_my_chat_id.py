"""
Get My Chat ID - Telegram Bot Helper
Created by Claude AI Assistant
"""

import asyncio
from telegram import Bot

async def get_chat_id():
    """Get the chat ID for the current user"""
    TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
    
    try:
        bot = Bot(TOKEN)
        
        # Get bot info
        bot_info = await bot.get_me()
        print(f"🤖 Bot Info: {bot_info.first_name} (@{bot_info.username})")
        print(f"🆔 Bot ID: {bot_info.id}")
        
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
                print(f"📱 Username: @{user_info.username or 'Not set'}")
                
                print(f"\n📋 Use this Chat ID in your alert script:")
                print(f"chat_id = '{chat_id}'")
                
                return chat_id
            else:
                print("❌ No message found in updates")
        else:
            print("❌ No updates found. Make sure you sent a message to the bot.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your bot token is correct and the bot is running.")

if __name__ == '__main__':
    asyncio.run(get_chat_id())

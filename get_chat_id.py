"""
Get Telegram Chat ID Script
Helps you find your Telegram chat ID for notifications
"""

import requests
import os
import json
from dotenv import load_dotenv

def get_telegram_chat_id():
    """Get Telegram chat ID from bot updates"""
    
    print("📱 TELEGRAM CHAT ID FINDER")
    print("=" * 40)
    
    # Load environment
    load_dotenv('alpaca_live_trading.env')
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment")
        print("📝 Please set your bot token in alpaca_live_trading.env")
        return
    
    print("🤖 Bot Token: Found ✅")
    print(f"   Token: ...{bot_token[-8:]}")
    
    print(f"\n📋 INSTRUCTIONS:")
    print("1. Open Telegram")
    print("2. Find your bot (search for the username you created)")
    print("3. Send ANY message to your bot (e.g., 'hello')")
    print("4. Then press Enter here to get your Chat ID")
    
    input("\nPress Enter after sending a message to your bot...")
    
    try:
        print(f"\n🔍 Fetching updates from Telegram...")
        
        # Get bot updates
        response = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getUpdates",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}")
            print("🔧 Check your bot token")
            return
        
        data = response.json()
        
        if not data.get('ok'):
            print(f"❌ Telegram API Error: {data}")
            return
        
        if not data.get('result'):
            print("❌ No messages found!")
            print("📝 Make sure you:")
            print("   1. Created the bot with @BotFather")
            print("   2. Sent a message to your bot")
            print("   3. Used the correct bot token")
            return
        
        # Get the latest message
        latest_update = data['result'][-1]
        message = latest_update.get('message', {})
        
        if not message:
            print("❌ No message found in update")
            return
        
        chat_id = message['chat']['id']
        user_info = message['from']
        
        print(f"\n✅ SUCCESS! Found your information:")
        print(f"   Chat ID: {chat_id}")
        print(f"   Name: {user_info.get('first_name', 'Unknown')} {user_info.get('last_name', '')}")
        print(f"   Username: @{user_info.get('username', 'None')}")
        
        # Test sending a message
        print(f"\n🧪 Testing bot communication...")
        
        test_response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                'chat_id': chat_id,
                'text': '🎉 Chat ID found successfully!\n\nYour Omni Alpha bot is ready for trading notifications!'
            },
            timeout=10
        )
        
        if test_response.status_code == 200:
            print("✅ Test message sent successfully!")
        else:
            print(f"⚠️ Test message failed: {test_response.status_code}")
        
        # Save to file for future use
        chat_info = {
            'chat_id': chat_id,
            'user_info': user_info,
            'bot_token': f"...{bot_token[-8:]}",  # Masked for security
            'timestamp': str(requests.get('http://worldtimeapi.org/api/timezone/UTC').json().get('datetime', 'unknown'))
        }
        
        with open('telegram_chat_info.json', 'w') as f:
            json.dump(chat_info, f, indent=2)
        
        print(f"\n💾 Chat information saved to: telegram_chat_info.json")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Your bot is ready for trading!")
        print("2. Run: python omni_alpha_enhanced_live.py")
        print("3. Send /start in Telegram to begin trading")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("🔧 Check your internet connection")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_bot_token():
    """Test if bot token is valid"""
    
    load_dotenv('alpaca_live_trading.env')
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("❌ No bot token found")
        return False
    
    try:
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot_info = data['result']
                print(f"✅ Bot token valid: @{bot_info['username']}")
                return True
        
        print(f"❌ Invalid bot token")
        return False
        
    except Exception as e:
        print(f"❌ Bot token test failed: {e}")
        return False

def main():
    """Main function"""
    
    # First test the bot token
    if test_bot_token():
        get_telegram_chat_id()
    else:
        print(f"\n📝 TO FIX:")
        print("1. Go to @BotFather in Telegram")
        print("2. Send: /newbot")
        print("3. Create your bot and get the token")
        print("4. Add to alpaca_live_trading.env:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")

if __name__ == "__main__":
    main()

"""
Complete Alpaca Trading Test - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import os
import sys
import time
from datetime import datetime

def check_setup():
    """Check if setup is complete"""
    print("🔍 CHECKING SETUP...")
    print("-" * 30)
    
    # Check if alpaca_keys.py exists
    if not os.path.exists('alpaca_keys.py'):
        print("❌ API keys not found!")
        print("Please run: python setup_alpaca_keys.py")
        return False
    
    # Check if alpaca-trade-api is installed
    try:
        import alpaca_trade_api
        print("✅ Alpaca SDK installed")
    except ImportError:
        print("❌ Alpaca SDK not installed!")
        print("Please run: pip install alpaca-trade-api")
        return False
    
    # Check if python-telegram-bot is installed
    try:
        import telegram
        print("✅ Telegram bot library installed")
    except ImportError:
        print("❌ Telegram bot library not installed!")
        print("Please run: pip install python-telegram-bot")
        return False
    
    print("✅ Setup check complete!")
    return True

def test_alpaca_connection():
    """Test Alpaca connection"""
    print("\n🔌 TESTING ALPACA CONNECTION...")
    print("-" * 40)
    
    try:
        from alpaca_keys import API_KEY, SECRET_KEY, BASE_URL
        import alpaca_trade_api as tradeapi
        
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        account = api.get_account()
        
        print("✅ Alpaca connection successful!")
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Alpaca connection failed: {str(e)}")
        return False

def test_telegram_bot():
    """Test Telegram bot connection"""
    print("\n📱 TESTING TELEGRAM BOT...")
    print("-" * 35)
    
    try:
        import requests
        
        # Test bot API
        bot_token = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
        response = requests.get(f'https://api.telegram.org/bot{bot_token}/getMe')
        
        if response.status_code == 200:
            bot_info = response.json()
            print("✅ Telegram bot connection successful!")
            print(f"Bot Name: {bot_info['result']['first_name']}")
            print(f"Bot Username: @{bot_info['result']['username']}")
            return True
        else:
            print("❌ Telegram bot connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Telegram bot test failed: {str(e)}")
        return False

def run_5min_trade_test():
    """Run the 5-minute trade test"""
    print("\n🚀 RUNNING 5-MINUTE TRADE TEST...")
    print("-" * 45)
    
    try:
        # Import and run the trade test
        from quick_5min_trade import main as trade_main
        trade_main()
        return True
        
    except Exception as e:
        print(f"❌ 5-minute trade test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 OMNI ALPHA 12.0 - COMPLETE TESTING SUITE")
    print("=" * 60)
    print(f"⏰ Test Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    # Step 2: Test Alpaca connection
    if not test_alpaca_connection():
        print("\n❌ Alpaca connection failed. Please check your API keys.")
        return
    
    # Step 3: Test Telegram bot
    if not test_telegram_bot():
        print("\n❌ Telegram bot connection failed.")
        return
    
    # Step 4: Run 5-minute trade test
    print("\n🎯 ALL TESTS PASSED! Running 5-minute trade test...")
    time.sleep(2)
    
    if run_5min_trade_test():
        print("\n🎉 COMPLETE TEST SUITE PASSED!")
        print("=" * 60)
        print("✅ Setup complete")
        print("✅ Alpaca connection working")
        print("✅ Telegram bot connected")
        print("✅ 5-minute trade test successful")
        print()
        print("🚀 READY FOR GLOBAL MARKET DOMINANCE!")
        print("🌍 Omni Alpha 12.0 is fully operational!")
    else:
        print("\n❌ 5-minute trade test failed.")

if __name__ == '__main__':
    main()

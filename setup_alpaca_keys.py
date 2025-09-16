"""
Alpaca API Keys Setup - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import os
import alpaca_trade_api as tradeapi
from datetime import datetime

def setup_alpaca_keys():
    """Interactive setup for Alpaca API keys"""
    print("🚀 OMNI ALPHA 12.0 - ALPACA API KEYS SETUP")
    print("=" * 60)
    print()
    
    print("📋 STEP 1: GET YOUR ALPACA API KEYS")
    print("-" * 40)
    print("1. Go to: https://alpaca.markets/")
    print("2. Sign up for FREE account")
    print("3. Go to: https://app.alpaca.markets/paper/dashboard/overview")
    print("4. Click 'API Keys' in the sidebar")
    print("5. Generate new API keys")
    print("6. Copy your API_KEY and SECRET_KEY")
    print()
    
    # Get API keys from user
    api_key = input("Enter your Alpaca API Key: ").strip()
    secret_key = input("Enter your Alpaca Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("❌ API keys cannot be empty!")
        return False
    
    print()
    print("🔧 STEP 2: TESTING CONNECTION")
    print("-" * 40)
    
    try:
        # Test connection
        api = tradeapi.REST(api_key, secret_key, 'https://paper-api.alpaca.markets', api_version='v2')
        
        # Get account info
        account = api.get_account()
        
        print("✅ CONNECTION SUCCESSFUL!")
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Save keys to file
        print()
        print("💾 STEP 3: SAVING API KEYS")
        print("-" * 40)
        
        with open('alpaca_keys.py', 'w') as f:
            f.write(f"""# Alpaca API Keys - Omni Alpha 12.0
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

API_KEY = '{api_key}'
SECRET_KEY = '{secret_key}'
BASE_URL = 'https://paper-api.alpaca.markets'
""")
        
        print("✅ API keys saved to 'alpaca_keys.py'")
        print("✅ Ready to run trading tests!")
        
        return True
        
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {str(e)}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("1. Check if your API keys are correct")
        print("2. Make sure you're using Paper Trading keys (not Live Trading)")
        print("3. Verify your account is active")
        print("4. Check your internet connection")
        return False

def test_trading():
    """Test basic trading functionality"""
    print()
    print("🧪 STEP 4: TESTING TRADING FUNCTIONALITY")
    print("-" * 50)
    
    try:
        from alpaca_keys import API_KEY, SECRET_KEY, BASE_URL
        
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        
        # Get a quote
        print("📈 Getting SPY quote...")
        spy_quote = api.get_latest_quote('SPY')
        print(f"SPY Ask: ${spy_quote.ap}")
        print(f"SPY Bid: ${spy_quote.bp}")
        
        # Check account
        account = api.get_account()
        print(f"Available Cash: ${float(account.cash):,.2f}")
        
        print("✅ Trading functionality test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Trading test failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    if setup_alpaca_keys():
        if test_trading():
            print()
            print("🎉 SETUP COMPLETE!")
            print("=" * 60)
            print("✅ Alpaca API keys configured")
            print("✅ Connection tested successfully")
            print("✅ Trading functionality verified")
            print()
            print("🚀 READY TO RUN:")
            print("python alpaca_paper_trading.py")
            print("python telegram_alpaca_bot.py")
            print("python quick_5min_trade.py")
            print()
            print("🌍 Global Market Dominance Ready!")
        else:
            print("❌ Setup incomplete - trading test failed")
    else:
        print("❌ Setup failed - please check your API keys")

if __name__ == '__main__':
    main()

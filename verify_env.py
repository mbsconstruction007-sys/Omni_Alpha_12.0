"""
Environment Configuration Verification Script
Checks if all required environment variables are properly set
"""

import os
from dotenv import load_dotenv

def verify_environment():
    """Verify all required environment variables"""
    
    print("🔍 ENVIRONMENT CONFIGURATION CHECK")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv('alpaca_live_trading.env')
    
    # Required variables
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'ALPACA_BASE_URL',
        'TELEGRAM_BOT_TOKEN'
    ]
    
    # Optional variables with defaults
    optional_vars = {
        'MAX_POSITION_SIZE_PERCENT': '10',
        'MAX_POSITIONS': '20',
        'STOP_LOSS_PERCENT': '3',
        'TAKE_PROFIT_PERCENT': '6',
        'SCAN_INTERVAL_SECONDS': '300'
    }
    
    all_required_set = True
    
    # Check required variables
    print("\n📋 REQUIRED VARIABLES:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first 8 and last 4 characters for security
            if len(value) > 12:
                masked_value = f"{value[:8]}{'*' * (len(value)-12)}{value[-4:]}"
            else:
                masked_value = f"{'*' * len(value)}"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"❌ {var}: NOT SET")
            all_required_set = False
    
    # Check optional variables
    print(f"\n⚙️ OPTIONAL VARIABLES (with defaults):")
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"✅ {var}: {value}")
    
    # Test connections if all required vars are set
    if all_required_set:
        print(f"\n🔗 CONNECTION TESTS:")
        
        # Test Alpaca connection
        try:
            import alpaca_trade_api as tradeapi
            
            api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url=os.getenv('ALPACA_BASE_URL'),
                api_version='v2'
            )
            
            account = api.get_account()
            print(f"✅ Alpaca API: Connected (Status: {account.status})")
            print(f"   💰 Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
        except Exception as e:
            print(f"❌ Alpaca API: Failed - {str(e)}")
        
        # Test Telegram bot
        try:
            import requests
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info['ok']:
                    print(f"✅ Telegram Bot: Connected (@{bot_info['result']['username']})")
                else:
                    print(f"❌ Telegram Bot: API Error - {bot_info}")
            else:
                print(f"❌ Telegram Bot: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Telegram Bot: Failed - {str(e)}")
    
    # Final summary
    print(f"\n" + "=" * 50)
    if all_required_set:
        print("🎉 CONFIGURATION STATUS: READY")
        print("✅ All required environment variables are set")
        print("✅ You can run the enhanced trading bot now!")
        print(f"\n🚀 Next steps:")
        print("   1. Run: python omni_alpha_enhanced_live.py")
        print("   2. Open Telegram and send /start to your bot")
        print("   3. Use /help to see all commands")
    else:
        print("⚠️ CONFIGURATION STATUS: INCOMPLETE")
        print("❌ Some required environment variables are missing")
        print("📝 Please update your alpaca_live_trading.env file")
        print(f"\n📋 Required format:")
        print("   ALPACA_API_KEY=your_alpaca_key_here")
        print("   ALPACA_SECRET_KEY=your_alpaca_secret_here")
        print("   ALPACA_BASE_URL=https://paper-api.alpaca.markets")
        print("   TELEGRAM_BOT_TOKEN=your_telegram_token_here")

def main():
    """Main verification function"""
    
    try:
        verify_environment()
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("📝 Make sure alpaca_live_trading.env file exists")

if __name__ == "__main__":
    main()

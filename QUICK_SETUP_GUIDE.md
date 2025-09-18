# 🚀 OMNI ALPHA ENHANCED - COMPLETE SETUP GUIDE

## 📋 QUICK SETUP CHECKLIST

### ✅ Step 1: Get Alpaca Keys (REQUIRED)
1. Go to: https://app.alpaca.markets/signup
2. Sign up for FREE paper trading account
3. Go to: https://app.alpaca.markets/paper/dashboard/overview  
4. Click "Generate API Key"
5. Copy both keys (API Key & Secret Key)

### ✅ Step 2: Get Telegram Bot Token (REQUIRED)
1. Open Telegram app
2. Search for @BotFather
3. Send: `/newbot`
4. Choose a name: `OmniAlphaBot`
5. Choose username: `OmniAlpha123_bot`
6. Copy the bot token

### ✅ Step 3: Create Environment File

Create `alpaca_live_trading.env` with your keys:

```bash
# MINIMUM REQUIRED - Bot won't run without these
ALPACA_API_KEY=PK6NQI7HSGQ7B38PYLG8
ALPACA_SECRET_KEY=gu15JAAvNMqbDGJ8m14ePtHOy3TgnAD7vHkvg74C
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN=8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk

# OPTIONAL TRADING PARAMETERS
MAX_POSITION_SIZE_PERCENT=10
MAX_POSITIONS=20
STOP_LOSS_PERCENT=3
TAKE_PROFIT_PERCENT=6
SCAN_INTERVAL_SECONDS=300
```

### ✅ Step 4: Install Required Packages

```bash
pip install alpaca-trade-api python-telegram-bot python-dotenv yfinance pandas numpy requests asyncio
```

### ✅ Step 5: Run the Enhanced Bot

```bash
python omni_alpha_enhanced_live.py
```

---

## 🔍 VERIFICATION SCRIPTS

### Verify Environment Setup

Create `verify_env.py`:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('alpaca_live_trading.env')

# Check required variables
required = [
    'ALPACA_API_KEY',
    'ALPACA_SECRET_KEY', 
    'ALPACA_BASE_URL',
    'TELEGRAM_BOT_TOKEN'
]

print("ENV Configuration Check:")
print("=" * 40)

all_set = True
for var in required:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {'*' * 10}{value[-4:]}")  # Show last 4 chars
    else:
        print(f"❌ {var}: NOT SET")
        all_set = False

if all_set:
    print("\n✅ All required ENV variables are set!")
    print("You can run the bot now!")
else:
    print("\n❌ Some ENV variables are missing!")
    print("Please set them before running the bot.")
```

### Get Your Telegram Chat ID

Create `get_chat_id.py`:

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv('alpaca_live_trading.env')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

print("Getting your Telegram Chat ID...")
print("1. Send a message to your bot first")
print("2. Then run this script")

try:
    response = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
    data = response.json()

    if data['result']:
        chat_id = data['result'][0]['message']['chat']['id']
        print(f"\n✅ Your Chat ID: {chat_id}")
        print(f"💬 From: {data['result'][0]['message']['from']['first_name']}")
    else:
        print("❌ No messages found. Send a message to your bot first!")
        
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## 📦 COMPLETE REQUIREMENTS.TXT

Create `requirements.txt`:

```txt
alpaca-trade-api==3.1.1
python-telegram-bot==20.7
python-dotenv==1.0.0
yfinance==0.2.33
pandas==2.1.4
numpy==1.24.3
requests==2.31.0
pytz==2023.3
asyncio==3.4.3
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎯 ONE-COMMAND SETUP

```bash
# Create environment file (replace with your keys)
cat > alpaca_live_trading.env << 'EOF'
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN=your_telegram_token_here
EOF

# Install packages
pip install alpaca-trade-api python-telegram-bot python-dotenv yfinance pandas numpy

# Run the bot
python omni_alpha_enhanced_live.py
```

---

## 🚨 TROUBLESHOOTING

### Issue 1: "API Key not found"

```python
# Test if environment is loading
import os
from dotenv import load_dotenv

load_dotenv('alpaca_live_trading.env')
print(f"API Key: {os.getenv('ALPACA_API_KEY')}")  # Should print your key
```

### Issue 2: "Invalid API Key"

```bash
# Test Alpaca connection
curl -X GET https://paper-api.alpaca.markets/v2/account \
  -H "APCA-API-KEY-ID: your_key_here" \
  -H "APCA-API-SECRET-KEY: your_secret_here"
```

### Issue 3: "Telegram bot not responding"

```bash
# Test bot token
curl https://api.telegram.org/bot{your_token}/getMe
```

### Issue 4: "Module not found"

```bash
# Install missing packages
pip install --upgrade alpaca-trade-api python-telegram-bot python-dotenv
```

---

## 🔒 SECURITY BEST PRACTICES

### Protect Your Keys

```bash
# Add to .gitignore
echo "*.env" >> .gitignore
echo "alpaca_live_trading.env" >> .gitignore

# Set file permissions (Linux/Mac)
chmod 600 alpaca_live_trading.env
```

### Use Environment-Specific Files

```bash
alpaca_live_trading.env      # Paper trading
alpaca_live_trading.prod.env # Real money (when ready)
alpaca_live_trading.test.env # Testing
```

---

## 📊 EXPECTED RESULTS

After successful setup, you should see:

```
============================================================
ENHANCED OMNI ALPHA TRADING SYSTEM
============================================================

Account Status:
   Buying Power: $198,084.90
   Portfolio Value: $99,941.33
   Number of Positions: 2

Trading Universe: 100 stocks
Max Position Size: $20,000
Auto Sell: ENABLED

Bot starting...
Open Telegram and send /help
============================================================
Alpaca connection verified
All enhanced features loaded
Risk management active
Auto-sell monitoring active
Telegram bot ready
Send /start in Telegram to begin
============================================================
```

## 📱 TELEGRAM COMMANDS TO TEST

After the bot starts, send these in Telegram:

```
/start - Initialize system
/account - View account info  
/scan - Find opportunities in 100+ stocks
/buy AAPL - Buy with proper position sizing
/positions - View all positions with auto-sell targets
/auto - Start full automation
/help - All commands
```

## 🎊 SUCCESS INDICATORS

✅ **Position Sizing Fixed**: $7,000-$20,000 per trade (not $478)
✅ **Auto-Selling Active**: Take profit/stop loss monitoring
✅ **100+ Stock Coverage**: Full market scanning
✅ **Risk Management**: Advanced controls active
✅ **Real-time Monitoring**: Every position tracked

## 💡 NEXT STEPS

1. **Test Manual Trading**: Use `/buy SYMBOL` commands
2. **Enable Automation**: Send `/auto` to start full automation  
3. **Monitor Performance**: Use `/positions` and `/performance`
4. **Adjust Settings**: Modify environment variables as needed

## 🆘 SUPPORT

If you encounter issues:

1. Run `python verify_env.py` to check configuration
2. Check the log file: `enhanced_trading.log`
3. Verify your Alpaca account at: https://app.alpaca.markets/paper/dashboard/overview
4. Test your Telegram bot with @BotFather

---

**🚀 YOUR ENHANCED TRADING SYSTEM IS NOW READY FOR $198K PORTFOLIO MANAGEMENT!**

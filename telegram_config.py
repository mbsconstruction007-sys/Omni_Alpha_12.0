# telegram_config.py
'''Telegram Bot Configuration'''

# Bot Settings
BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'  # Replace with your token from BotFather
ADMIN_CHAT_ID = YOUR_CHAT_ID  # Replace with your Telegram user ID

# Alert Settings
ENABLE_PRICE_ALERTS = True
ENABLE_TRADE_ALERTS = True
ENABLE_ERROR_ALERTS = True
ENABLE_DAILY_SUMMARY = True

# Trading Settings
ALLOW_REMOTE_TRADING = True
REQUIRE_CONFIRMATION = True
MAX_TRADE_SIZE = 10000

# Command Permissions
AUTHORIZED_USERS = [
    YOUR_CHAT_ID,  # Add your Telegram user ID
    # Add more authorized user IDs
]

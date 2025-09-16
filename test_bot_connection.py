# test_bot_connection.py
import os
from dotenv import load_dotenv
import requests

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Test the bot token
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getMe'
response = requests.get(url)

if response.status_code == 200:
    bot_info = response.json()['result']
    print('✅ Bot Connected Successfully!')
    print(f"Bot Name: {bot_info['first_name']}")
    print(f"Bot Username: @{bot_info['username']}")
    print(f"Bot ID: {bot_info['id']}")
else:
    print('❌ Connection failed. Check your token.')

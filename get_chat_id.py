# get_chat_id.py
'''Script to get your Telegram chat ID'''

import requests

BOT_TOKEN =8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk

# Send a message to your bot first, then run this
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
response = requests.get(url)
data = response.json()

if data['result']:
    chat_id = data['result'][0]['message']['chat']['id']
    print(f'Your Chat ID: {chat_id}')
    print(f'Add this to your config: ADMIN_CHAT_ID = {chat_id}')
else:
    print('No messages found. Send /start to your bot first!')

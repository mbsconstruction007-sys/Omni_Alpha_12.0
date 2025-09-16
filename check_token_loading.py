# check_token_loading.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token
token = os.getenv('TELEGRAM_BOT_TOKEN')

if token:
    print(f'Token found: {token}')
    print(f'Token length: {len(token)}')
    print(f'Token format looks correct: {len(token.split(":")) == 2}')
else:
    print('ERROR: Token not found in environment!')
    print('Make sure .env file exists and contains:')
    print('TELEGRAM_BOT_TOKEN=your_actual_token_here')

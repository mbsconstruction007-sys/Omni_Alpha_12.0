# test_env_loading.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Try to get the token
token = os.getenv('TELEGRAM_BOT_TOKEN')

print(f"Token found: {token is not None}")
if token:
    print(f"Token starts with: {token[:10]}...")
    print(f"Token length: {len(token)}")
else:
    print("Token is None - check .env file")

# Show all environment variables that start with TELEGRAM
for key, value in os.environ.items():
    if 'TELEGRAM' in key:
        print(f"{key} = {value[:20]}..." if len(value) > 20 else f"{key} = {value}")

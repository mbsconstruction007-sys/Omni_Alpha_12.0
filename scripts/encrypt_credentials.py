#!/usr/bin/env python3
"""
CREDENTIAL ENCRYPTION HELPER
============================
Safely encrypt your API keys for storage in .env files
"""

import os
import sys
import getpass
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except ImportError:
    print("❌ Missing cryptography library!")
    print("Install with: pip install cryptography")
    sys.exit(1)

def generate_encryption_key():
    """Generate a new encryption key"""
    return Fernet.generate_key().decode()

def encrypt_value(value: str, key: str) -> str:
    """Encrypt a value using the provided key"""
    fernet = Fernet(key.encode())
    return fernet.encrypt(value.encode()).decode()

def decrypt_value(encrypted_value: str, key: str) -> str:
    """Decrypt a value using the provided key"""
    fernet = Fernet(key.encode())
    return fernet.decrypt(encrypted_value.encode()).decode()

def main():
    print("="*60)
    print("🔐 OMNI ALPHA 5.0 - CREDENTIAL ENCRYPTION TOOL")
    print("="*60)
    
    # Check if .env.local exists
    env_file = Path(".env.local")
    if not env_file.exists():
        print("⚠️  .env.local not found. Creating from template...")
        template = Path("step1_environment_template.env")
        if template.exists():
            env_file.write_text(template.read_text())
        else:
            env_file.touch()
    
    # Generate or get encryption key
    print("\n1. Generate new encryption key")
    print("2. Use existing encryption key")
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        encryption_key = generate_encryption_key()
        print(f"\n✅ New encryption key generated!")
        print(f"📋 SAVE THIS KEY (add to .env.local as ENCRYPTION_KEY):")
        print(f"\n{encryption_key}\n")
        print("⚠️  WARNING: If you lose this key, you cannot decrypt your credentials!")
    else:
        encryption_key = getpass.getpass("\nEnter your encryption key: ").strip()
    
    # Validate key
    try:
        Fernet(encryption_key.encode())
    except Exception:
        print("❌ Invalid encryption key!")
        sys.exit(1)
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("What would you like to do?")
        print("1. Encrypt Alpaca credentials")
        print("2. Encrypt custom API key")
        print("3. Decrypt and verify credentials")
        print("4. Generate new encryption key")
        print("5. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            print("\n🔐 Encrypting Alpaca Credentials")
            api_key = getpass.getpass("Enter Alpaca API Key: ").strip()
            api_secret = getpass.getpass("Enter Alpaca Secret Key: ").strip()
            
            encrypted_key = encrypt_value(api_key, encryption_key)
            encrypted_secret = encrypt_value(api_secret, encryption_key)
            
            print("\n✅ Encrypted credentials (add these to .env.local):")
            print(f"\nAPI_KEY_ENCRYPTED={encrypted_key}")
            print(f"API_SECRET_ENCRYPTED={encrypted_secret}")
            
        elif choice == "2":
            print("\n🔐 Encrypting Custom API Key")
            key_name = input("Enter key name (e.g., GOOGLE_API_KEY): ").strip()
            key_value = getpass.getpass(f"Enter {key_name} value: ").strip()
            
            encrypted = encrypt_value(key_value, encryption_key)
            
            print(f"\n✅ Encrypted {key_name}:")
            print(f"\n{key_name}_ENCRYPTED={encrypted}")
            
        elif choice == "3":
            print("\n🔓 Decrypting Credentials for Verification")
            encrypted = getpass.getpass("Enter encrypted value: ").strip()
            
            try:
                decrypted = decrypt_value(encrypted, encryption_key)
                # Show only first and last 3 characters for security
                if len(decrypted) > 6:
                    masked = f"{decrypted[:3]}...{decrypted[-3:]}"
                else:
                    masked = "*" * len(decrypted)
                print(f"\n✅ Decryption successful!")
                print(f"Value (masked): {masked}")
                print(f"Length: {len(decrypted)} characters")
            except Exception as e:
                print(f"❌ Decryption failed: {e}")
        
        elif choice == "4":
            new_key = generate_encryption_key()
            print(f"\n✅ New encryption key:")
            print(f"\n{new_key}\n")
            print("⚠️  Remember to re-encrypt all values with this new key!")
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")
    
    # Final reminder
    print("\n" + "="*60)
    print("⚠️  SECURITY REMINDERS:")
    print("1. Never commit .env.local to git")
    print("2. Add .env.local to .gitignore")
    print("3. Store encryption key separately from encrypted values")
    print("4. Rotate keys every 90 days")
    print("5. Use different keys for dev/staging/production")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

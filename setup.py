#!/usr/bin/env python3
"""
Setup script for Omni Alpha project
Creates virtual environment and installs dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Setup the Omni Alpha project"""
    print("🚀 Setting up Omni Alpha project...")
    print("=" * 50)
    
    # Check if Python is available
    if not run_command("python --version", "Checking Python version"):
        print("❌ Python is not available. Please install Python 3.8+")
        sys.exit(1)
    
    # Create virtual environment
    if not run_command("python -m venv .venv", "Creating virtual environment"):
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        print("❌ Failed to upgrade pip")
        sys.exit(1)
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("2. Run the application:")
    print("   uvicorn src.app:app --host 127.0.0.1 --port 8000")
    print("3. Run tests:")
    print("   python check_step4_endpoints.py")
    print("   python check_step7_webhook.py")
    print("   python check_step8_advice.py")

if __name__ == "__main__":
    main()

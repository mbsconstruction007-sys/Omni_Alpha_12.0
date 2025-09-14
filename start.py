#!/usr/bin/env python3
"""
Start script for Omni Alpha project
Starts the FastAPI server with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Omni Alpha application"""
    print("🚀 Starting Omni Alpha Application...")
    print("=" * 50)
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup.py first.")
        sys.exit(1)
    
    # Check if src/app.py exists
    app_path = Path("src/app.py")
    if not app_path.exists():
        print("❌ Application file not found. Please check your project structure.")
        sys.exit(1)
    
    # Check if static directory exists
    static_path = Path("static")
    if not static_path.exists():
        print("❌ Static directory not found. Please check your project structure.")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("\n🌐 Starting server...")
    print("📡 API will be available at: http://127.0.0.1:8000")
    print("🎨 Dashboard will be available at: http://127.0.0.1:8000")
    print("📚 API docs will be available at: http://127.0.0.1:8000/docs")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.app:app", 
            "--host", "127.0.0.1", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

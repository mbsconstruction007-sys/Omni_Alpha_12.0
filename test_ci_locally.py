#!/usr/bin/env python3
"""
Local CI Test Script
Run this to test the same checks that CI runs
"""

import sys
import subprocess
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔍 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all CI tests locally"""
    print("🚀 Running Local CI Tests")
    print("=" * 50)
    
    tests = [
        ("python -c \"import src.app; print('App imports successfully')\"", "Import Test"),
        ("python -c \"from src.app import app; print('FastAPI app created successfully')\"", "FastAPI App Test"),
        ("python -c \"import requests; print('Requests module available')\"", "Requests Module Test"),
        ("python ci_health_check.py", "Health Check Test"),
        ("python -c \"from src.app import app; routes = [route.path for route in app.routes]; expected_routes = ['/', '/api', '/health', '/steps', '/analysis/start', '/webhook', '/advice']; [print(f'Route {route} found') if any(route in r for r in routes) else print(f'Route {route} missing') for route in expected_routes]; print('Route validation completed')\"", "Route Validation Test"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if run_command(command, description):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CI should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

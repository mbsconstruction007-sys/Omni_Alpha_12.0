#!/usr/bin/env python3
"""
Check Step 4 Endpoints Script
Tests the core API endpoints for the Omni Alpha project
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
BASE_URL = "http://127.0.0.1:8000"
BOT_BASE_URL = "http://127.0.0.1:8000"

def make_request(method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[Any, Any]:
    """Make HTTP request and return response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"error": str(e)}

def test_root_endpoint():
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    result = make_request("GET", "/")
    if "error" not in result:
        print(f"✅ Root endpoint: {result}")
        return True
    else:
        print(f"❌ Root endpoint failed: {result}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    result = make_request("GET", "/health")
    if "error" not in result:
        print(f"✅ Health check: {result}")
        return True
    else:
        print(f"❌ Health check failed: {result}")
        return False

def test_get_steps():
    """Test getting all analysis steps"""
    print("🔍 Testing get all steps...")
    result = make_request("GET", "/steps")
    if "error" not in result and "steps" in result:
        steps_count = len(result["steps"])
        print(f"✅ Get steps: Found {steps_count} steps")
        return True
    else:
        print(f"❌ Get steps failed: {result}")
        return False

def test_get_specific_step():
    """Test getting a specific step"""
    print("🔍 Testing get specific step...")
    result = make_request("GET", "/steps/1")
    if "error" not in result and "step_id" in result:
        print(f"✅ Get specific step: {result}")
        return True
    else:
        print(f"❌ Get specific step failed: {result}")
        return False

def test_start_analysis():
    """Test starting an analysis"""
    print("🔍 Testing start analysis...")
    data = {
        "analysis_type": "test_analysis",
        "parameters": {"test_param": "test_value"}
    }
    result = make_request("POST", "/analysis/start", data)
    if "error" not in result and "analysis" in result:
        print(f"✅ Start analysis: {result}")
        return True
    else:
        print(f"❌ Start analysis failed: {result}")
        return False

def test_complete_step():
    """Test completing a step"""
    print("🔍 Testing complete step...")
    data = {"test_data": "step_completed"}
    result = make_request("POST", "/steps/1/complete", data)
    if "error" not in result and "step" in result:
        print(f"✅ Complete step: {result}")
        return True
    else:
        print(f"❌ Complete step failed: {result}")
        return False

def test_webhook():
    """Test webhook endpoint"""
    print("🔍 Testing webhook endpoint...")
    data = {
        "event_type": "test_event",
        "data": {"test": "webhook_data"},
        "timestamp": "2024-01-01T00:00:00Z"
    }
    result = make_request("POST", "/webhook", data)
    if "error" not in result and "event_type" in result:
        print(f"✅ Webhook: {result}")
        return True
    else:
        print(f"❌ Webhook failed: {result}")
        return False

def test_advice():
    """Test advice endpoint"""
    print("🔍 Testing advice endpoint...")
    result = make_request("GET", "/advice")
    if "error" not in result and "recommendations" in result:
        print(f"✅ Advice: {result}")
        return True
    else:
        print(f"❌ Advice failed: {result}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Step 4 Endpoints Check...")
    print(f"📡 Base URL: {BASE_URL}")
    print("=" * 50)
    
    tests = [
        test_root_endpoint,
        test_health_check,
        test_get_steps,
        test_get_specific_step,
        test_start_analysis,
        test_complete_step,
        test_webhook,
        test_advice
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
        print("-" * 30)
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

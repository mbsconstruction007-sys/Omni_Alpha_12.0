#!/usr/bin/env python3
"""
Check Step 7 Webhook Script
Tests webhook functionality and bot integration for the Omni Alpha project
"""

import requests
import json
import sys
import time
from typing import Dict, Any, List

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

def test_webhook_basic():
    """Test basic webhook functionality"""
    print("🔍 Testing basic webhook...")
    
    payload = {
        "event_type": "analysis_started",
        "data": {
            "analysis_id": "test_123",
            "user_id": "user_456",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    result = make_request("POST", "/webhook", payload)
    if "error" not in result:
        print(f"✅ Basic webhook: {result}")
        return True
    else:
        print(f"❌ Basic webhook failed: {result}")
        return False

def test_webhook_step_completion():
    """Test webhook for step completion events"""
    print("🔍 Testing step completion webhook...")
    
    payload = {
        "event_type": "step_completed",
        "data": {
            "step_id": 1,
            "step_title": "Initial Analysis",
            "completion_time": "2024-01-01T00:05:00Z",
            "results": {"status": "success", "data_points": 42}
        },
        "timestamp": "2024-01-01T00:05:00Z"
    }
    
    result = make_request("POST", "/webhook", payload)
    if "error" not in result:
        print(f"✅ Step completion webhook: {result}")
        return True
    else:
        print(f"❌ Step completion webhook failed: {result}")
        return False

def test_webhook_analysis_complete():
    """Test webhook for analysis completion"""
    print("🔍 Testing analysis completion webhook...")
    
    payload = {
        "event_type": "analysis_completed",
        "data": {
            "analysis_id": "test_123",
            "total_steps": 24,
            "completed_steps": 24,
            "final_results": {
                "success_rate": 95.5,
                "recommendations": ["Continue monitoring", "Review quarterly"]
            }
        },
        "timestamp": "2024-01-01T01:00:00Z"
    }
    
    result = make_request("POST", "/webhook", payload)
    if "error" not in result:
        print(f"✅ Analysis completion webhook: {result}")
        return True
    else:
        print(f"❌ Analysis completion webhook failed: {result}")
        return False

def test_webhook_error_handling():
    """Test webhook error handling"""
    print("🔍 Testing webhook error handling...")
    
    # Test with invalid payload
    invalid_payload = {
        "event_type": "invalid_event",
        "data": "invalid_data_format"
    }
    
    result = make_request("POST", "/webhook", invalid_payload)
    if "error" not in result:
        print(f"✅ Error handling webhook: {result}")
        return True
    else:
        print(f"❌ Error handling webhook failed: {result}")
        return False

def test_webhook_bot_integration():
    """Test webhook integration with bot functionality"""
    print("🔍 Testing bot integration webhook...")
    
    payload = {
        "event_type": "bot_command",
        "data": {
            "command": "get_analysis_status",
            "user_id": "bot_user_123",
            "parameters": {"analysis_id": "test_123"}
        },
        "timestamp": "2024-01-01T00:10:00Z"
    }
    
    result = make_request("POST", "/webhook", payload)
    if "error" not in result:
        print(f"✅ Bot integration webhook: {result}")
        return True
    else:
        print(f"❌ Bot integration webhook failed: {result}")
        return False

def test_webhook_sequence():
    """Test a sequence of webhook events"""
    print("🔍 Testing webhook sequence...")
    
    events = [
        {
            "event_type": "analysis_started",
            "data": {"analysis_id": "seq_test_123", "user_id": "user_789"}
        },
        {
            "event_type": "step_started",
            "data": {"step_id": 1, "analysis_id": "seq_test_123"}
        },
        {
            "event_type": "step_completed",
            "data": {"step_id": 1, "analysis_id": "seq_test_123", "duration": 30}
        },
        {
            "event_type": "analysis_paused",
            "data": {"analysis_id": "seq_test_123", "reason": "user_request"}
        }
    ]
    
    success_count = 0
    for i, event in enumerate(events):
        print(f"  📤 Sending event {i+1}: {event['event_type']}")
        result = make_request("POST", "/webhook", event)
        if "error" not in result:
            success_count += 1
            print(f"  ✅ Event {i+1} successful")
        else:
            print(f"  ❌ Event {i+1} failed: {result}")
        time.sleep(0.1)  # Small delay between requests
    
    if success_count == len(events):
        print(f"✅ Webhook sequence: All {len(events)} events processed")
        return True
    else:
        print(f"❌ Webhook sequence: {success_count}/{len(events)} events successful")
        return False

def test_webhook_performance():
    """Test webhook performance with multiple rapid requests"""
    print("🔍 Testing webhook performance...")
    
    payload = {
        "event_type": "performance_test",
        "data": {"test_id": "perf_123", "timestamp": "2024-01-01T00:00:00Z"}
    }
    
    start_time = time.time()
    success_count = 0
    total_requests = 10
    
    for i in range(total_requests):
        result = make_request("POST", "/webhook", payload)
        if "error" not in result:
            success_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    avg_response_time = duration / total_requests
    
    print(f"✅ Performance test: {success_count}/{total_requests} successful")
    print(f"📊 Average response time: {avg_response_time:.3f}s")
    
    if success_count == total_requests and avg_response_time < 1.0:
        return True
    else:
        return False

def main():
    """Run all webhook tests"""
    print("🚀 Starting Step 7 Webhook Check...")
    print(f"📡 Base URL: {BASE_URL}")
    print("=" * 50)
    
    tests = [
        test_webhook_basic,
        test_webhook_step_completion,
        test_webhook_analysis_complete,
        test_webhook_error_handling,
        test_webhook_bot_integration,
        test_webhook_sequence,
        test_webhook_performance
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
        print("🎉 All webhook tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some webhook tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

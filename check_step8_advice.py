#!/usr/bin/env python3
"""
Check Step 8 Advice Script
Tests advice and recommendation functionality for the Omni Alpha project
"""

import requests
import json
import sys
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

def test_basic_advice():
    """Test basic advice endpoint"""
    print("🔍 Testing basic advice endpoint...")
    
    result = make_request("GET", "/advice")
    if "error" not in result:
        required_fields = ["total_steps", "completed_steps", "progress_percentage", "recommendations"]
        if all(field in result for field in required_fields):
            print(f"✅ Basic advice: {result}")
            return True
        else:
            print(f"❌ Basic advice missing required fields: {result}")
            return False
    else:
        print(f"❌ Basic advice failed: {result}")
        return False

def test_advice_with_completed_steps():
    """Test advice after completing some steps"""
    print("🔍 Testing advice with completed steps...")
    
    # First, complete a few steps
    for step_id in [1, 2, 3]:
        data = {"completion_data": f"Step {step_id} completed", "timestamp": "2024-01-01T00:00:00Z"}
        make_request("POST", f"/steps/{step_id}/complete", data)
    
    # Then get advice
    result = make_request("GET", "/advice")
    if "error" not in result:
        if result["completed_steps"] >= 3:
            print(f"✅ Advice with completed steps: {result}")
            return True
        else:
            print(f"❌ Advice doesn't reflect completed steps: {result}")
            return False
    else:
        print(f"❌ Advice with completed steps failed: {result}")
        return False

def test_advice_progress_calculation():
    """Test advice progress calculation"""
    print("🔍 Testing advice progress calculation...")
    
    result = make_request("GET", "/advice")
    if "error" not in result:
        expected_progress = (result["completed_steps"] / result["total_steps"]) * 100
        if abs(result["progress_percentage"] - expected_progress) < 0.1:
            print(f"✅ Progress calculation: {result['progress_percentage']:.1f}%")
            return True
        else:
            print(f"❌ Progress calculation incorrect: {result}")
            return False
    else:
        print(f"❌ Progress calculation failed: {result}")
        return False

def test_advice_recommendations():
    """Test advice recommendations"""
    print("🔍 Testing advice recommendations...")
    
    result = make_request("GET", "/advice")
    if "error" not in result:
        if "recommendations" in result and isinstance(result["recommendations"], list):
            if len(result["recommendations"]) > 0:
                print(f"✅ Recommendations: {len(result['recommendations'])} items")
                for i, rec in enumerate(result["recommendations"]):
                    print(f"  📝 {i+1}. {rec}")
                return True
            else:
                print(f"❌ No recommendations provided: {result}")
                return False
        else:
            print(f"❌ Invalid recommendations format: {result}")
            return False
    else:
        print(f"❌ Recommendations test failed: {result}")
        return False

def test_advice_after_full_analysis():
    """Test advice after completing all steps"""
    print("🔍 Testing advice after full analysis...")
    
    # Complete all remaining steps
    for step_id in range(4, 25):
        data = {"completion_data": f"Step {step_id} completed", "timestamp": "2024-01-01T00:00:00Z"}
        make_request("POST", f"/steps/{step_id}/complete", data)
    
    # Get advice
    result = make_request("GET", "/advice")
    if "error" not in result:
        if result["completed_steps"] == 24 and result["progress_percentage"] == 100.0:
            print(f"✅ Full analysis advice: {result}")
            return True
        else:
            print(f"❌ Full analysis advice incorrect: {result}")
            return False
    else:
        print(f"❌ Full analysis advice failed: {result}")
        return False

def test_advice_consistency():
    """Test advice consistency across multiple calls"""
    print("🔍 Testing advice consistency...")
    
    results = []
    for i in range(3):
        result = make_request("GET", "/advice")
        if "error" not in result:
            results.append(result)
        time.sleep(0.1)
    
    if len(results) == 3:
        # Check if all results are consistent
        first_result = results[0]
        consistent = all(
            result["total_steps"] == first_result["total_steps"] and
            result["completed_steps"] == first_result["completed_steps"] and
            result["progress_percentage"] == first_result["progress_percentage"]
            for result in results
        )
        
        if consistent:
            print(f"✅ Advice consistency: All {len(results)} calls consistent")
            return True
        else:
            print(f"❌ Advice consistency: Results vary between calls")
            return False
    else:
        print(f"❌ Advice consistency: Only {len(results)}/3 calls successful")
        return False

def test_advice_with_analysis_start():
    """Test advice after starting a new analysis"""
    print("🔍 Testing advice after analysis start...")
    
    # Start a new analysis
    analysis_data = {
        "analysis_type": "comprehensive_test",
        "parameters": {"test_mode": True}
    }
    start_result = make_request("POST", "/analysis/start", analysis_data)
    
    if "error" not in start_result:
        # Get advice after starting
        advice_result = make_request("GET", "/advice")
        if "error" not in advice_result:
            if advice_result["completed_steps"] == 0:
                print(f"✅ Advice after analysis start: {advice_result}")
                return True
            else:
                print(f"❌ Advice should show 0 completed steps: {advice_result}")
                return False
        else:
            print(f"❌ Advice after analysis start failed: {advice_result}")
            return False
    else:
        print(f"❌ Analysis start failed: {start_result}")
        return False

def test_advice_error_handling():
    """Test advice error handling"""
    print("🔍 Testing advice error handling...")
    
    # This test ensures the advice endpoint handles edge cases gracefully
    result = make_request("GET", "/advice")
    if "error" not in result:
        # Check if all required fields are present and valid
        if (isinstance(result["total_steps"], int) and 
            isinstance(result["completed_steps"], int) and 
            isinstance(result["progress_percentage"], (int, float)) and 
            isinstance(result["recommendations"], list)):
            print(f"✅ Error handling: Valid response format")
            return True
        else:
            print(f"❌ Error handling: Invalid response format: {result}")
            return False
    else:
        print(f"❌ Error handling: Request failed: {result}")
        return False

def main():
    """Run all advice tests"""
    print("🚀 Starting Step 8 Advice Check...")
    print(f"📡 Base URL: {BASE_URL}")
    print("=" * 50)
    
    tests = [
        test_basic_advice,
        test_advice_with_completed_steps,
        test_advice_progress_calculation,
        test_advice_recommendations,
        test_advice_after_full_analysis,
        test_advice_consistency,
        test_advice_with_analysis_start,
        test_advice_error_handling
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
        print("🎉 All advice tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some advice tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

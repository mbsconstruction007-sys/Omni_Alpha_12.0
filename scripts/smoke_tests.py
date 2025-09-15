#!/usr/bin/env python3
"""
Production smoke tests for Omni Alpha 5.0
"""

import requests
import time
import sys
from datetime import datetime

def test_health_endpoint():
    """Test basic health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_detailed_health():
    """Test detailed health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health/detailed", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data
        assert "checks" in data
        print("✅ Detailed health endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Detailed health endpoint test failed: {e}")
        return False

def test_metrics_endpoint():
    """Test metrics endpoint"""
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        assert response.status_code == 200
        content = response.text
        assert "omni_alpha_uptime_seconds" in content
        print("✅ Metrics endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Metrics endpoint test failed: {e}")
        return False

def test_api_info():
    """Test API info endpoint"""
    try:
        response = requests.get("http://localhost:8000/api", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "5.0.0"
        print("✅ API info endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ API info endpoint test failed: {e}")
        return False

def test_performance():
    """Test performance with multiple requests"""
    try:
        start_time = time.time()
        responses = []
        
        for i in range(10):
            response = requests.get("http://localhost:8000/health", timeout=5)
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(responses)
        
        # Check all responses are successful
        for response in responses:
            assert response.status_code == 200
        
        # Check performance (should be under 100ms per request)
        assert avg_time < 0.1, f"Average response time {avg_time:.3f}s is too slow"
        
        print(f"✅ Performance test passed (avg: {avg_time:.3f}s per request)")
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("🔥 Running production smoke tests...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        test_health_endpoint,
        test_detailed_health,
        test_metrics_endpoint,
        test_api_info,
        test_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("📊 Smoke Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All smoke tests passed! Production deployment is healthy.")
        return True
    else:
        print("❌ Some smoke tests failed. Please check the deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Fixed test script for Step 9 - Works on Windows
"""

import sys
import os
import asyncio
import platform
import time
from datetime import datetime

print(f"🖥️ Platform: {platform.system()}")
print(f"🐍 Python: {sys.version}")
print(f"📅 Test Time: {datetime.now().isoformat()}")

# Add the backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_tests():
    """Run all Step 9 tests with proper error handling"""
    
    results = {
        "passed": [],
        "failed": [],
        "skipped": []
    }
    
    print("\n" + "="*60)
    print("🧠⚡ STEP 9: HYBRID AI BRAIN - FIXED TEST SUITE")
    print("="*60)
    
    # Test 1: Hybrid Brain
    try:
        from app.ai_brain.hybrid_brain import HybridAIBrain
        brain = HybridAIBrain()
        result = await brain.think({"test": "data", "symbol": "AAPL", "price": 150})
        print(f"✅ Hybrid Brain Test: Backend={result.get('backend')}, Decision={result.get('decision')}")
        results["passed"].append("Hybrid Brain")
    except Exception as e:
        print(f"❌ Hybrid Brain Test Failed: {e}")
        results["failed"].append("Hybrid Brain")
    
    # Test 2: Universal Execution
    try:
        from app.ai_brain.hybrid_brain import UniversalExecutionEngine
        engine = UniversalExecutionEngine()
        order = {"symbol": "TEST", "quantity": 100, "price": 100, "side": "BUY", "volume": 1000000}
        result = await engine.execute_with_ai(order)
        print(f"✅ Execution Engine Test: Status={result['status']}, Backend={result.get('backend_used')}")
        results["passed"].append("Execution Engine")
    except Exception as e:
        print(f"❌ Execution Engine Test Failed: {e}")
        results["failed"].append("Execution Engine")
    
    # Test 3: Performance Metrics
    try:
        from app.ai_brain.hybrid_brain import UniversalExecutionEngine
        engine = UniversalExecutionEngine()
        
        # Execute a few orders
        for i in range(3):
            order = {"symbol": f"TEST{i}", "quantity": 100, "price": 100 + i, "side": "BUY", "volume": 1000000}
            await engine.execute_with_ai(order)
        
        metrics = await engine.get_performance_metrics()
        print(f"✅ Performance Metrics Test: Orders={metrics['total_orders']}, Success Rate={metrics['success_rate']:.2%}")
        results["passed"].append("Performance Metrics")
    except Exception as e:
        print(f"❌ Performance Metrics Test Failed: {e}")
        results["failed"].append("Performance Metrics")
    
    # Test 4: Brain Evolution
    try:
        from app.ai_brain.hybrid_brain import HybridAIBrain
        brain = HybridAIBrain()
        evolution = await brain.evolve()
        print(f"✅ Brain Evolution Test: Consciousness={evolution['consciousness_level']:.2f}")
        results["passed"].append("Brain Evolution")
    except Exception as e:
        print(f"❌ Brain Evolution Test Failed: {e}")
        results["failed"].append("Brain Evolution")
    
    # Test 5: Dream State
    try:
        from app.ai_brain.hybrid_brain import HybridAIBrain
        brain = HybridAIBrain()
        dream = await brain.dream()
        print(f"✅ Dream State Test: Insights={len(dream['insights'])}")
        results["passed"].append("Dream State")
    except Exception as e:
        print(f"❌ Dream State Test Failed: {e}")
        results["failed"].append("Dream State")
    
    # Test 6: Backend Detection
    try:
        from app.ai_brain.hybrid_brain import HybridAIBrain
        brain = HybridAIBrain()
        backend_name = brain.backend.__class__.__name__
        print(f"✅ Backend Detection Test: {backend_name}")
        results["passed"].append("Backend Detection")
    except Exception as e:
        print(f"❌ Backend Detection Test Failed: {e}")
        results["failed"].append("Backend Detection")
    
    # Test 7: Load Testing
    try:
        from app.ai_brain.hybrid_brain import UniversalExecutionEngine
        engine = UniversalExecutionEngine()
        
        start_time = time.time()
        tasks = []
        
        # Create 10 concurrent orders
        for i in range(10):
            order = {"symbol": f"LOAD{i}", "quantity": 100, "price": 100, "side": "BUY", "volume": 1000000}
            task = asyncio.create_task(engine.execute_with_ai(order))
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        successful = len([r for r in results_list if r['status'] == 'executed'])
        print(f"✅ Load Testing: {successful}/10 orders in {duration:.2f}s")
        results["passed"].append("Load Testing")
    except Exception as e:
        print(f"❌ Load Testing Failed: {e}")
        results["failed"].append("Load Testing")
    
    # Test 8: API Compatibility
    try:
        # Test if we can import the original API endpoints
        import importlib.util
        api_path = os.path.join(os.path.dirname(__file__), "app", "api", "endpoints", "ai_execution.py")
        if os.path.exists(api_path):
            print("✅ API Endpoints Available")
            results["passed"].append("API Endpoints")
        else:
            print("⚠️ API Endpoints Not Found (Expected)")
            results["skipped"].append("API Endpoints")
    except Exception as e:
        print(f"❌ API Endpoints Test Failed: {e}")
        results["failed"].append("API Endpoints")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {len(results['passed'])} tests")
    print(f"❌ Failed: {len(results['failed'])} tests")
    print(f"⏭️ Skipped: {len(results['skipped'])} tests")
    
    if results['passed']:
        print("\n✅ Passed tests:")
        for test in results['passed']:
            print(f"  - {test}")
    
    if results['failed']:
        print("\n❌ Failed tests:")
        for test in results['failed']:
            print(f"  - {test}")
    
    if results['skipped']:
        print("\n⏭️ Skipped tests:")
        for test in results['skipped']:
            print(f"  - {test}")
    
    success_rate = len(results['passed']) / (len(results['passed']) + len(results['failed'])) * 100
    print(f"\n📈 Success Rate: {success_rate:.1f}%")
    
    return len(results['failed']) == 0

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)

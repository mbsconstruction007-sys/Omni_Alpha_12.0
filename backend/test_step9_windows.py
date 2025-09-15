"""
Windows-Compatible Step 9 Test Script
No PyTorch dependencies - Pure NumPy implementation
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
    """Run all Step 9 tests with Windows compatibility"""
    
    results = {
        "passed": [],
        "failed": [],
        "skipped": []
    }
    
    print("\n" + "="*60)
    print("🧠⚡ STEP 9: WINDOWS AI BRAIN - COMPATIBILITY TEST SUITE")
    print("="*60)
    
    # Test 1: Windows AI Brain
    try:
        from app.ai_brain.windows_brain import WindowsAIBrain
        brain = WindowsAIBrain()
        result = await brain.think({"test": "data", "symbol": "AAPL", "price": 150, "volume": 2000000})
        print(f"✅ Windows AI Brain Test: Backend={result.get('backend')}, Decision={result.get('decision')}")
        results["passed"].append("Windows AI Brain")
    except Exception as e:
        print(f"❌ Windows AI Brain Test Failed: {e}")
        results["failed"].append("Windows AI Brain")
    
    # Test 2: Windows Execution Engine
    try:
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        engine = WindowsExecutionEngine()
        order = {"symbol": "TEST", "quantity": 100, "price": 100, "side": "BUY", "volume": 1000000}
        result = await engine.execute_with_ai(order)
        print(f"✅ Windows Execution Engine Test: Status={result['status']}, Backend={result.get('backend_used')}")
        results["passed"].append("Windows Execution Engine")
    except Exception as e:
        print(f"❌ Windows Execution Engine Test Failed: {e}")
        results["failed"].append("Windows Execution Engine")
    
    # Test 3: Performance Metrics
    try:
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        engine = WindowsExecutionEngine()
        
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
        from app.ai_brain.windows_brain import WindowsAIBrain
        brain = WindowsAIBrain()
        evolution = await brain.evolve()
        print(f"✅ Brain Evolution Test: Consciousness={evolution['consciousness_level']:.2f}")
        results["passed"].append("Brain Evolution")
    except Exception as e:
        print(f"❌ Brain Evolution Test Failed: {e}")
        results["failed"].append("Brain Evolution")
    
    # Test 5: Dream State
    try:
        from app.ai_brain.windows_brain import WindowsAIBrain
        brain = WindowsAIBrain()
        dream = await brain.dream()
        print(f"✅ Dream State Test: Insights={len(dream['insights'])}")
        results["passed"].append("Dream State")
    except Exception as e:
        print(f"❌ Dream State Test Failed: {e}")
        results["failed"].append("Dream State")
    
    # Test 6: Market Analysis
    try:
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        engine = WindowsExecutionEngine()
        analysis = await engine.get_market_analysis("AAPL")
        print(f"✅ Market Analysis Test: Trend={analysis['trend']}, Recommendation={analysis['recommendation']}")
        results["passed"].append("Market Analysis")
    except Exception as e:
        print(f"❌ Market Analysis Test Failed: {e}")
        results["failed"].append("Market Analysis")
    
    # Test 7: Load Testing
    try:
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        engine = WindowsExecutionEngine()
        
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
    
    # Test 8: Neural Network Operations
    try:
        from app.ai_brain.windows_brain import WindowsAIBrain
        brain = WindowsAIBrain()
        
        # Test neural network operations
        test_input = {"symbol": "NN_TEST", "price": 200, "volume": 3000000, "quantity": 500}
        result = await brain.think(test_input)
        
        # Verify neural network output
        if "probabilities" in result and len(result["probabilities"]) == 3:
            print(f"✅ Neural Network Test: Probabilities calculated correctly")
            results["passed"].append("Neural Network Operations")
        else:
            print(f"❌ Neural Network Test: Invalid output format")
            results["failed"].append("Neural Network Operations")
    except Exception as e:
        print(f"❌ Neural Network Test Failed: {e}")
        results["failed"].append("Neural Network Operations")
    
    # Test 9: Consciousness System
    try:
        from app.ai_brain.windows_brain import WindowsAIBrain
        brain = WindowsAIBrain()
        
        # Test consciousness evolution
        initial_level = brain.consciousness_level
        await brain.evolve()
        final_level = brain.consciousness_level
        
        if final_level > initial_level:
            print(f"✅ Consciousness Test: Level increased from {initial_level:.2f} to {final_level:.2f}")
            results["passed"].append("Consciousness System")
        else:
            print(f"❌ Consciousness Test: Level did not increase")
            results["failed"].append("Consciousness System")
    except Exception as e:
        print(f"❌ Consciousness Test Failed: {e}")
        results["failed"].append("Consciousness System")
    
    # Test 10: Platform Compatibility
    try:
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        engine = WindowsExecutionEngine()
        metrics = await engine.get_performance_metrics()
        
        if metrics.get("platform") == platform.system():
            print(f"✅ Platform Compatibility Test: Correctly detected {metrics['platform']}")
            results["passed"].append("Platform Compatibility")
        else:
            print(f"❌ Platform Compatibility Test: Platform mismatch")
            results["failed"].append("Platform Compatibility")
    except Exception as e:
        print(f"❌ Platform Compatibility Test Failed: {e}")
        results["failed"].append("Platform Compatibility")
    
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
    
    # Platform info
    print(f"\n🖥️ Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📦 Dependencies: NumPy only (no PyTorch required)")
    
    return len(results['failed']) == 0

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")
    if success:
        print("🎉 Windows compatibility achieved! Step 9 is fully operational.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    sys.exit(0 if success else 1)

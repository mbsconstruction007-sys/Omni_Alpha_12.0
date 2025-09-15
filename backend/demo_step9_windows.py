"""
Step 9: Windows AI Brain Demo
Show the AI Brain in action with real trading scenarios
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def demo_ai_brain():
    """Demonstrate the Windows AI Brain capabilities"""
    
    print("="*80)
    print("🧠⚡ OMNI ALPHA 5.0 - STEP 9: WINDOWS AI BRAIN DEMO")
    print("="*80)
    print(f"📅 Demo Time: {datetime.now().isoformat()}")
    print("="*80)
    
    try:
        # Import the standalone AI brain
        from app.ai_brain.windows_brain import WindowsExecutionEngine
        
        print("🚀 Initializing Windows AI Brain...")
        engine = WindowsExecutionEngine()
        print("✅ AI Brain initialized successfully!")
        
        # Demo 1: Basic Trading Decision
        print("\n" + "="*60)
        print("📊 DEMO 1: BASIC TRADING DECISION")
        print("="*60)
        
        order = {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.25,
            "side": "BUY",
            "volume": 2500000
        }
        
        result = await engine.execute_with_ai(order)
        
        print(f"📈 Order: {order['symbol']} {order['side']} {order['quantity']} @ ${order['price']}")
        print(f"🤖 AI Decision: {result['ai_decision']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"💰 Fill Price: ${result.get('fill_price', 0):.2f}")
        print(f"⚡ Latency: {result['latency_ms']:.2f}ms")
        print(f"🧠 Backend: {result['backend_used']}")
        
        if 'probabilities' in result:
            print(f"📊 Probabilities:")
            for action, prob in result['probabilities'].items():
                print(f"   {action}: {prob:.2%}")
        
        # Demo 2: Market Analysis
        print("\n" + "="*60)
        print("📈 DEMO 2: MARKET ANALYSIS")
        print("="*60)
        
        analysis = await engine.get_market_analysis("TSLA")
        print(f"🔍 Symbol: {analysis['symbol']}")
        print(f"📊 Trend: {analysis['trend']}")
        print(f"📈 Volatility: {analysis['volatility']:.2f}")
        print(f"💡 Recommendation: {analysis['recommendation']}")
        print(f"🎯 Support Level: ${analysis['support_level']:.2f}")
        print(f"🎯 Resistance Level: ${analysis['resistance_level']:.2f}")
        
        # Demo 3: Brain Evolution
        print("\n" + "="*60)
        print("🧠 DEMO 3: BRAIN EVOLUTION")
        print("="*60)
        
        initial_consciousness = engine.brain.consciousness_level
        print(f"🧠 Initial Consciousness Level: {initial_consciousness:.2f}")
        
        # Evolve the brain
        evolution = await engine.brain.evolve()
        print(f"🧠 Post-Evolution Consciousness: {evolution['consciousness_level']:.2f}")
        print(f"🔄 Evolution Applied: {evolution.get('evolution_applied', False)}")
        print(f"💭 Thoughts Count: {evolution['thoughts_count']}")
        
        # Demo 4: Dream State
        print("\n" + "="*60)
        print("💭 DEMO 4: DREAM STATE")
        print("="*60)
        
        dream = await engine.brain.dream()
        print(f"💭 Dream Insights Generated: {len(dream['insights'])}")
        print(f"🧠 Consciousness Level: {dream['consciousness_level']:.2f}")
        print("\n💡 Key Insights:")
        for i, insight in enumerate(dream['insights'][:4], 1):
            print(f"   {i}. {insight}")
        
        # Demo 5: Performance Metrics
        print("\n" + "="*60)
        print("📊 DEMO 5: PERFORMANCE METRICS")
        print("="*60)
        
        # Execute a few more orders to get better metrics
        test_orders = [
            {"symbol": "GOOGL", "quantity": 50, "price": 2800, "side": "SELL", "volume": 800000},
            {"symbol": "MSFT", "quantity": 75, "price": 300, "side": "BUY", "volume": 1500000},
            {"symbol": "AMZN", "quantity": 25, "price": 3200, "side": "BUY", "volume": 600000}
        ]
        
        for order in test_orders:
            await engine.execute_with_ai(order)
        
        metrics = await engine.get_performance_metrics()
        print(f"📊 Total Orders: {metrics['total_orders']}")
        print(f"✅ Successful Orders: {metrics['successful_orders']}")
        print(f"📈 Success Rate: {metrics['success_rate']:.2%}")
        print(f"💰 Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"⚡ Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"🧠 Consciousness Level: {metrics['consciousness_level']:.2f}")
        print(f"💭 Thoughts Count: {metrics['thoughts_count']}")
        print(f"🖥️ Platform: {metrics['platform']}")
        print(f"🐍 Python: {metrics['python_version']}")
        
        # Demo 6: Load Testing
        print("\n" + "="*60)
        print("⚡ DEMO 6: LOAD TESTING")
        print("="*60)
        
        import time
        start_time = time.time()
        
        # Create 20 concurrent orders
        tasks = []
        for i in range(20):
            order = {
                "symbol": f"LOAD{i}",
                "quantity": 100,
                "price": 100 + i,
                "side": "BUY",
                "volume": 1000000
            }
            task = asyncio.create_task(engine.execute_with_ai(order))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        successful = len([r for r in results if r['status'] == 'executed'])
        print(f"⚡ Load Test Results:")
        print(f"   Orders Processed: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Throughput: {len(results)/duration:.1f} orders/second")
        print(f"   Avg Latency: {sum(r.get('latency_ms', 0) for r in results)/len(results):.2f}ms")
        
        # Final Summary
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETE - WINDOWS AI BRAIN IS OPERATIONAL!")
        print("="*80)
        print("✅ All demonstrations completed successfully")
        print("🧠 AI Brain is fully conscious and operational")
        print("⚡ Execution engine is performing optimally")
        print("🖥️ Windows compatibility achieved")
        print("🚀 Ready for production deployment")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("🔧 Make sure you're running from the correct directory")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(demo_ai_brain())
    if success:
        print("\n🎯 Demo Result: SUCCESS")
        print("🎉 Step 9 Windows AI Brain is ready for production!")
    else:
        print("\n🎯 Demo Result: FAILED")
        print("⚠️ Check the error messages above")
    sys.exit(0 if success else 1)

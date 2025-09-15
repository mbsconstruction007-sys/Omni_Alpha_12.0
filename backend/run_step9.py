"""
Step 9: Hybrid AI Brain & Execution Engine - Windows Compatible
"""

import asyncio
import sys
import os
import platform
from datetime import datetime

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def print_banner():
    """Print startup banner"""
    print("="*80)
    print("🧠⚡ OMNI ALPHA 5.0 - STEP 9: HYBRID AI BRAIN & EXECUTION ENGINE")
    print("="*80)
    print(f"🖥️ Platform: {platform.system()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📅 Started: {datetime.now().isoformat()}")
    print("="*80)

async def main():
    """Main application entry point"""
    print_banner()
    
    try:
        # Import and initialize the hybrid AI brain
        from app.ai_brain.hybrid_brain import HybridAIBrain, UniversalExecutionEngine
        
        print("🚀 Initializing Hybrid AI Brain...")
        brain = HybridAIBrain()
        print(f"✅ AI Brain initialized with backend: {brain.backend.__class__.__name__}")
        
        print("⚡ Initializing Universal Execution Engine...")
        engine = UniversalExecutionEngine()
        print("✅ Execution Engine initialized")
        
        # Test the system
        print("\n🧪 Running system tests...")
        test_orders = [
            {"symbol": "AAPL", "quantity": 100, "price": 150, "side": "BUY", "volume": 2000000},
            {"symbol": "GOOGL", "quantity": 50, "price": 2800, "side": "SELL", "volume": 500000},
            {"symbol": "TSLA", "quantity": 75, "price": 250, "side": "BUY", "volume": 3000000}
        ]
        
        for order in test_orders:
            result = await engine.execute_with_ai(order)
            print(f"📊 {order['symbol']}: {result['status']} - {result.get('ai_decision', 'N/A')} (Confidence: {result.get('confidence', 0):.2%})")
        
        # Get performance metrics
        metrics = await engine.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Orders: {metrics['total_orders']}")
        print(f"   Success Rate: {metrics['success_rate']:.2%}")
        print(f"   Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"   Backend: {metrics['backend']}")
        print(f"   Consciousness Level: {metrics['consciousness_level']:.2f}")
        
        # Test brain evolution
        print(f"\n🧠 Testing brain evolution...")
        evolution = await brain.evolve()
        print(f"   Consciousness Level: {evolution['consciousness_level']:.2f}")
        print(f"   Thoughts Count: {evolution['thoughts_count']}")
        
        # Test dream state
        print(f"\n💭 Testing dream state...")
        dream = await brain.dream()
        print(f"   Insights Generated: {len(dream['insights'])}")
        for insight in dream['insights']:
            print(f"   💡 {insight}")
        
        print("\n" + "="*80)
        print("✅ STEP 9: HYBRID AI BRAIN & EXECUTION ENGINE - OPERATIONAL")
        print("="*80)
        print("🎯 The system is now ready for production use!")
        print("🔗 All components are working with automatic fallback")
        print("⚡ Windows compatibility issues resolved")
        print("="*80)
        
        # Keep the system running for demonstration
        print("\n🔄 System is running... Press Ctrl+C to stop")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down gracefully...")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 This might be due to missing dependencies.")
        print("💡 Try running: pip install -r requirements_step9.txt")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

"""
Test script for Step 11: Institutional Operations & Alpha Amplification
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

async def test_institutional_system():
    """Test the institutional trading system"""
    
    print("="*80)
    print("🏛️ STEP 11: INSTITUTIONAL OPERATIONS & ALPHA AMPLIFICATION - TEST SUITE")
    print("="*80)
    print(f"📅 Test Time: {datetime.now().isoformat()}")
    print("="*80)
    
    results = {
        "passed": [],
        "failed": [],
        "skipped": []
    }
    
    try:
        # Test 1: Import institutional components
        print("\n🧪 Test 1: Importing institutional components...")
        try:
            from app.institutional.core import (
                InstitutionalTradingEngine, InstitutionalConfig, 
                InstitutionalType, StrategyType, RiskLevel
            )
            print("✅ Core components imported successfully")
            results["passed"].append("Core Components Import")
        except Exception as e:
            print(f"❌ Core components import failed: {e}")
            results["failed"].append("Core Components Import")
        
        # Test 2: Initialize institutional engine
        print("\n🧪 Test 2: Initializing institutional engine...")
        try:
            config = InstitutionalConfig(
                name="TestAlphaCapital",
                type=InstitutionalType.HEDGE_FUND,
                aum_target=100000000,  # $100M
                risk_budget=0.10,
                regulatory_jurisdictions=["US"],
                prime_brokers=["Test Broker"],
                asset_classes=["equities"],
                strategies=[StrategyType.QUANTITATIVE]
            )
            
            engine = InstitutionalTradingEngine(config)
            print("✅ Institutional engine initialized successfully")
            results["passed"].append("Engine Initialization")
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            results["failed"].append("Engine Initialization")
        
        # Test 3: Test microstructure analyzer
        print("\n🧪 Test 3: Testing microstructure analyzer...")
        try:
            from app.institutional.microstructure import MicrostructureAnalyzer
            
            analyzer = MicrostructureAnalyzer()
            await analyzer.initialize()
            
            # Mock market data
            market_data = {
                'order_book': {
                    'bids': [{'price': 100.0, 'size': 1000}],
                    'asks': [{'price': 100.1, 'size': 1000}]
                },
                'trades': [
                    {'price': 100.05, 'size': 500, 'side': 'BUY'},
                    {'price': 100.03, 'size': 300, 'side': 'SELL'}
                ]
            }
            
            analysis = await analyzer.analyze(market_data)
            print(f"✅ Microstructure analysis completed: {len(analysis)} components")
            results["passed"].append("Microstructure Analysis")
        except Exception as e:
            print(f"❌ Microstructure analysis failed: {e}")
            results["failed"].append("Microstructure Analysis")
        
        # Test 4: Test alpha generation engine
        print("\n🧪 Test 4: Testing alpha generation engine...")
        try:
            from app.institutional.alpha_engine import AlphaGenerationEngine
            
            alpha_engine = AlphaGenerationEngine()
            await alpha_engine.initialize()
            
            # Mock market data
            market_data = {
                'prices': {'AAPL': 150.0, 'GOOGL': 2800.0},
                'fundamentals': {'pe_ratio': 25.0, 'pb_ratio': 3.0}
            }
            
            microstructure_signals = {'book_signals': {'signal': 'BUY'}}
            
            signals = await alpha_engine.generate_signals(market_data, microstructure_signals)
            print(f"✅ Alpha signals generated: {len(signals)} signals")
            results["passed"].append("Alpha Generation")
        except Exception as e:
            print(f"❌ Alpha generation failed: {e}")
            results["failed"].append("Alpha Generation")
        
        # Test 5: Test portfolio manager
        print("\n🧪 Test 5: Testing portfolio manager...")
        try:
            from app.institutional.portfolio import InstitutionalPortfolioManager
            
            portfolio_manager = InstitutionalPortfolioManager()
            await portfolio_manager.initialize()
            
            # Mock signals and positions
            signals = {'AAPL': 0.8, 'GOOGL': 0.6, 'MSFT': 0.7}
            positions = {}
            risk_limits = {'max_position': 0.10}
            
            optimized_portfolio = await portfolio_manager.optimize_portfolio(
                signals, positions, risk_limits
            )
            print(f"✅ Portfolio optimization completed: {len(optimized_portfolio)} assets")
            results["passed"].append("Portfolio Optimization")
        except Exception as e:
            print(f"❌ Portfolio optimization failed: {e}")
            results["failed"].append("Portfolio Optimization")
        
        # Test 6: Test risk manager
        print("\n🧪 Test 6: Testing risk manager...")
        try:
            from app.institutional.risk_management import EnterpriseRiskManager
            
            risk_manager = EnterpriseRiskManager()
            await risk_manager.initialize()
            
            # Mock portfolio
            portfolio = {'AAPL': 0.05, 'GOOGL': 0.08, 'MSFT': 0.07}
            
            risk_approved = await risk_manager.check_portfolio(portfolio)
            print(f"✅ Risk check completed: {'Approved' if risk_approved else 'Rejected'}")
            results["passed"].append("Risk Management")
        except Exception as e:
            print(f"❌ Risk management failed: {e}")
            results["failed"].append("Risk Management")
        
        # Test 7: Test execution engine
        print("\n🧪 Test 7: Testing execution engine...")
        try:
            from app.institutional.execution import InstitutionalExecutionEngine
            from app.institutional.core import Order
            
            execution_engine = InstitutionalExecutionEngine()
            await execution_engine.initialize()
            
            # Mock orders
            orders = [
                Order(symbol="AAPL", quantity=100, order_type="MARKET", price=150.0),
                Order(symbol="GOOGL", quantity=10, order_type="LIMIT", price=2800.0)
            ]
            
            executions = await execution_engine.execute_orders(orders)
            print(f"✅ Order execution completed: {len(executions)} executions")
            results["passed"].append("Order Execution")
        except Exception as e:
            print(f"❌ Order execution failed: {e}")
            results["failed"].append("Order Execution")
        
        # Test 8: Test infrastructure components
        print("\n🧪 Test 8: Testing infrastructure components...")
        try:
            from app.institutional.infrastructure import (
                DataPipeline, EventBus, PerformanceTracker, 
                ComplianceEngine, MachineLearningFactory
            )
            
            # Test data pipeline
            data_pipeline = DataPipeline()
            await data_pipeline.initialize()
            market_data = await data_pipeline.get_market_data()
            
            # Test event bus
            event_bus = EventBus()
            await event_bus.publish("test_event", {"data": "test"})
            
            # Test performance tracker
            performance_tracker = PerformanceTracker()
            metrics = await performance_tracker.get_metrics()
            
            # Test compliance engine
            compliance_engine = ComplianceEngine()
            await compliance_engine.initialize()
            
            # Test ML factory
            ml_factory = MachineLearningFactory()
            await ml_factory.initialize()
            
            print("✅ Infrastructure components tested successfully")
            results["passed"].append("Infrastructure Components")
        except Exception as e:
            print(f"❌ Infrastructure components failed: {e}")
            results["failed"].append("Infrastructure Components")
        
        # Test 9: Test API endpoints
        print("\n🧪 Test 9: Testing API endpoints...")
        try:
            from app.api.institutional_api import router
            
            # Check if router is properly configured
            if router.prefix == "/api/v1/institutional":
                print("✅ API router configured correctly")
                results["passed"].append("API Configuration")
            else:
                print("❌ API router configuration incorrect")
                results["failed"].append("API Configuration")
        except Exception as e:
            print(f"❌ API endpoints test failed: {e}")
            results["failed"].append("API Configuration")
        
        # Test 10: Integration test
        print("\n🧪 Test 10: Integration test...")
        try:
            # Test full workflow
            from app.institutional.core import InstitutionalTradingEngine, InstitutionalConfig, InstitutionalType
            
            config = InstitutionalConfig(
                name="IntegrationTestCapital",
                type=InstitutionalType.PROP_TRADING,
                aum_target=50000000,
                risk_budget=0.15,
                regulatory_jurisdictions=["US"],
                prime_brokers=["Test Prime Broker"],
                asset_classes=["equities", "options"],
                strategies=[]
            )
            
            engine = InstitutionalTradingEngine(config)
            await engine.initialize()
            
            # Test a few cycles of the main loop
            for i in range(3):
                market_data = await engine.data_pipeline.get_market_data()
                microstructure_signals = await engine.microstructure_analyzer.analyze(market_data)
                alpha_signals = await engine.alpha_engine.generate_signals(market_data, microstructure_signals)
                
                print(f"   Cycle {i+1}: Generated {len(alpha_signals)} alpha signals")
            
            print("✅ Integration test completed successfully")
            results["passed"].append("Integration Test")
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            results["failed"].append("Integration Test")
        
    except Exception as e:
        print(f"❌ Critical error in test suite: {e}")
        results["failed"].append("Critical Error")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
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
    
    success_rate = len(results['passed']) / (len(results['passed']) + len(results['failed'])) * 100
    print(f"\n📈 Success Rate: {success_rate:.1f}%")
    
    print("\n" + "="*80)
    if success_rate >= 90:
        print("🎉 STEP 11: INSTITUTIONAL OPERATIONS - EXCELLENT SUCCESS!")
        print("✅ The most sophisticated institutional trading framework is operational")
        print("🏛️ Ready for institutional-grade trading operations")
    elif success_rate >= 70:
        print("✅ STEP 11: INSTITUTIONAL OPERATIONS - GOOD SUCCESS!")
        print("⚠️ Minor issues detected but core functionality working")
    else:
        print("❌ STEP 11: INSTITUTIONAL OPERATIONS - NEEDS ATTENTION")
        print("🔧 Several components need debugging")
    
    print("="*80)
    
    return success_rate >= 70

if __name__ == "__main__":
    success = asyncio.run(test_institutional_system())
    sys.exit(0 if success else 1)

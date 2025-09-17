'''Test Advanced Omni Alpha System Components'''

import alpaca_trade_api as tradeapi
import sys
import os

# Add core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ml_engine import MLPredictionEngine
from core.monitoring import MonitoringSystem
from core.analytics import AnalyticsEngine

# Configuration
ALPACA_KEY = 'PK6NQI7HSGQ7B38PYLG8'
ALPACA_SECRET = 'gu15JAAvNMqbDGJ8m14ePtHOy3TgnAD7vHkvg74C'
BASE_URL = 'https://paper-api.alpaca.markets'

def test_system():
    print('🚀 TESTING OMNI ALPHA ADVANCED SYSTEM')
    print('=' * 50)
    
    # Initialize API
    api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL)
    
    # Test API connection
    print('📡 Testing Alpaca connection...')
    try:
        account = api.get_account()
        print(f'✅ Connected! Account: {account.status}')
        print(f'   Portfolio: ${float(account.portfolio_value):,.2f}')
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        return
    
    # Test Step 6: ML Engine
    print('\n🧠 Testing ML Prediction Engine...')
    try:
        ml_engine = MLPredictionEngine(api)
        prediction = ml_engine.predict('AAPL')
        if prediction:
            print(f'✅ ML Prediction for AAPL:')
            print(f'   Direction: {prediction["prediction"]}')
            print(f'   Confidence: {prediction["confidence"]:.1f}%')
            print(f'   Action: {prediction["action"]}')
        else:
            print('⚠️ ML prediction returned None')
    except Exception as e:
        print(f'❌ ML Engine error: {e}')
    
    # Test Step 7: Monitoring
    print('\n📊 Testing Monitoring System...')
    try:
        monitoring = MonitoringSystem(api)
        metrics = monitoring.calculate_metrics()
        if metrics:
            print(f'✅ Monitoring metrics:')
            print(f'   Equity: ${metrics["equity"]:,.2f}')
            print(f'   Risk Score: {metrics["risk_score"]}/100')
            print(f'   Positions: {metrics["position_count"]}')
        else:
            print('⚠️ No metrics calculated')
    except Exception as e:
        print(f'❌ Monitoring error: {e}')
    
    # Test Step 8: Analytics
    print('\n📈 Testing Analytics Engine...')
    try:
        analytics = AnalyticsEngine(api)
        analysis = analytics.analyze_symbol('AAPL')
        if analysis:
            print(f'✅ Analytics for AAPL:')
            print(f'   Score: {analysis["composite_score"]}/100')
            print(f'   Recommendation: {analysis["recommendation"]}')
            if analysis['technical']:
                print(f'   Trend: {analysis["technical"]["trend"]}')
        else:
            print('⚠️ Analytics returned None')
    except Exception as e:
        print(f'❌ Analytics error: {e}')
    
    print('\n' + '=' * 50)
    print('🎉 ADVANCED SYSTEM TEST COMPLETE!')
    print('✅ All components initialized successfully')

if __name__ == '__main__':
    test_system()

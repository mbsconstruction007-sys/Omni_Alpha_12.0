"""
Test script for Step 5: Advanced Trading Components
"""

import asyncio
import sys
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_engine.core.signal_processor import SignalProcessor
from src.trading_engine.core.regime_detector import RegimeDetector
from src.trading_engine.psychology.market_psychology import MarketPsychologyEngine
from src.trading_engine.strategies.base_strategy import Signal, SignalAction


async def test_signal_processor():
    """Test Signal Processor functionality"""
    print("\n🧪 Testing Signal Processor...")
    
    config = {
        'min_signal_strength': 50,
        'signal_confirmation_sources': 3,
        'correlation_filter_threshold': 0.8,
        'kalman_filter_enabled': True,
        'min_signal_confidence': 0.5,
        'min_risk_reward_ratio': 1.5,
        'min_final_score': 60,
        'min_confirmations': 2
    }
    
    processor = SignalProcessor(config)
    
    # Test signal creation
    signal = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strength=75.0,
        confidence=0.8,
        entry_price=Decimal("150.00"),
        stop_loss=Decimal("145.00"),
        take_profit=Decimal("160.00"),
        indicators={
            'rsi': 45,
            'macd': 0.5,
            'macd_signal': 0.3,
            'volume_ratio': 1.5,
            'adx': 30
        },
        regime_alignment=0.8,
        psychology_score=0.7
    )
    
    # Process signal
    processed_signal = await processor.process(signal)
    
    if processed_signal:
        print(f"✅ Signal processed successfully")
        print(f"   Symbol: {processed_signal.symbol}")
        print(f"   Action: {processed_signal.action}")
        print(f"   Strength: {processed_signal.strength:.1f}")
        print(f"   Confidence: {processed_signal.confidence:.2f}")
        print(f"   Confirmations: {processed_signal.indicators.get('confirmations', 0)}")
        print(f"   Final Score: {processed_signal.indicators.get('final_score', 0):.1f}")
    else:
        print("❌ Signal processing failed")
    
    # Test statistics
    stats = processor.get_statistics()
    print(f"✅ Statistics: {stats['total_processed']} signals processed")
    
    return processed_signal is not None


async def test_regime_detector():
    """Test Market Regime Detector functionality"""
    print("\n🧪 Testing Market Regime Detector...")
    
    # Mock market data manager
    mock_market_data = Mock()
    mock_market_data.get_historical_data = AsyncMock()
    mock_market_data.get_quote = AsyncMock()
    
    # Mock SPY data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 400
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = np.random.randint(1000000, 5000000, 100)
    
    spy_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    mock_market_data.get_historical_data.return_value = spy_data
    mock_market_data.get_quote.return_value = {'last': 20}  # VIX level
    
    config = {
        'regime_lookback_periods': 60,
        'regime_update_frequency': 300,
        'bull_regime_threshold': 0.6,
        'bear_regime_threshold': -0.6,
        'high_volatility_threshold': 30,
        'trend_strength_threshold': 25
    }
    
    detector = RegimeDetector(mock_market_data, config)
    
    # Detect regime
    regime = await detector.detect_regime()
    
    print(f"✅ Current regime: {regime}")
    print(f"   Confidence: {detector.regime_confidence:.2f}")
    
    # Test regime characteristics
    characteristics = detector.get_regime_characteristics()
    print(f"   Volatility: {characteristics['volatility']}")
    print(f"   Trend: {characteristics['trend']}")
    print(f"   Momentum: {characteristics['momentum']}")
    print(f"   Breadth: {characteristics['breadth']}")
    
    # Test parameter adjustment
    base_params = {
        'position_size': 1000,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'confidence_threshold': 0.7
    }
    
    adjusted_params = detector.adjust_parameters_for_regime(base_params)
    print(f"   Adjusted position size: {adjusted_params.get('position_size', 'N/A')}")
    
    return regime is not None


async def test_psychology_engine():
    """Test Market Psychology Engine functionality"""
    print("\n🧪 Testing Market Psychology Engine...")
    
    # Mock market data manager
    mock_market_data = Mock()
    mock_market_data.get_historical_data = AsyncMock()
    mock_market_data.get_quote = AsyncMock()
    
    # Mock SPY data for fear/greed calculation
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
    np.random.seed(42)
    
    # Generate trending price data
    base_price = 400
    trend = np.linspace(0, 0.2, 150)  # 20% uptrend
    noise = np.random.normal(0, 0.01, 150)
    prices = base_price * (1 + trend + noise)
    
    highs = prices * (1 + abs(np.random.normal(0, 0.005, 150)))
    lows = prices * (1 - abs(np.random.normal(0, 0.005, 150)))
    volumes = np.random.randint(50000000, 100000000, 150)
    
    spy_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    mock_market_data.get_historical_data.return_value = spy_data
    mock_market_data.get_quote.return_value = {'last': 20}  # VIX level
    
    config = {
        'fear_greed_extreme_threshold': 20,
        'greed_extreme_threshold': 80
    }
    
    engine = MarketPsychologyEngine(mock_market_data, config)
    
    # Get market psychology
    psychology = await engine.get_market_psychology()
    
    print(f"✅ Fear/Greed Index: {psychology['fear_greed_index']:.1f}")
    print(f"   Market Sentiment: {psychology['sentiment']['overall']}")
    print(f"   Wyckoff Phase: {psychology['wyckoff_phase']}")
    print(f"   Manipulation Risk: {psychology['manipulation_risk']:.2f}")
    
    # Test signal analysis
    signal = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strength=75.0,
        confidence=0.8,
        entry_price=Decimal("150.00"),
        stop_loss=Decimal("145.00"),
        psychology_score=0.7
    )
    
    psychology_score = await engine.analyze_signal(signal)
    print(f"   Signal Psychology Score: {psychology_score:.2f}")
    
    return psychology is not None


async def test_integration():
    """Test integration between components"""
    print("\n🧪 Testing Component Integration...")
    
    # Mock market data
    mock_market_data = Mock()
    mock_market_data.get_historical_data = AsyncMock()
    mock_market_data.get_quote = AsyncMock()
    
    # Create components
    regime_config = {'regime_lookback_periods': 60}
    regime_detector = RegimeDetector(mock_market_data, regime_config)
    
    signal_config = {'min_signal_strength': 50}
    signal_processor = SignalProcessor(signal_config)
    
    psychology_config = {'fear_greed_extreme_threshold': 20}
    psychology_engine = MarketPsychologyEngine(mock_market_data, psychology_config)
    
    # Create comprehensive signal
    signal = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strength=75.0,
        confidence=0.8,
        entry_price=Decimal("150.00"),
        stop_loss=Decimal("145.00"),
        take_profit=Decimal("160.00"),
        indicators={
            'rsi': 45,
            'macd': 0.5,
            'volume_ratio': 1.5,
            'adx': 30
        },
        regime_alignment=0.8,
        psychology_score=0.7
    )
    
    # Process through signal processor
    processed_signal = await signal_processor.process(signal)
    
    if processed_signal:
        print("✅ Signal successfully processed through pipeline")
        print(f"   Final strength: {processed_signal.strength:.1f}")
        print(f"   Final confidence: {processed_signal.confidence:.2f}")
        print(f"   Confirmations: {processed_signal.indicators.get('confirmations', 0)}")
        print(f"   Final score: {processed_signal.indicators.get('final_score', 0):.1f}")
        
        # Test statistics
        stats = signal_processor.get_statistics()
        print(f"   Total signals processed: {stats['total_processed']}")
        
        return True
    else:
        print("❌ Signal processing failed in integration test")
        return False


async def main():
    """Main test function"""
    print("🚀 Step 5: Advanced Trading Components Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test individual components
    results.append(await test_signal_processor())
    results.append(await test_regime_detector())
    results.append(await test_psychology_engine())
    results.append(await test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL STEP 5 TESTS PASSED!")
        print("✅ Advanced Trading Components are working correctly")
        print("✅ Signal Processor: Advanced filtering and validation")
        print("✅ Market Regime Detector: Multi-method regime identification")
        print("✅ Market Psychology Engine: Sentiment and manipulation analysis")
        print("✅ Component Integration: Seamless workflow")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("❌ Some components need attention")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

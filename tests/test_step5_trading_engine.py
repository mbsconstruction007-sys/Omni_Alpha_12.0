"""
Test suite for Step 5: Advanced Trading Components
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock

# Import Step 5 components
from src.trading_engine.core.signal_processor import SignalProcessor
from src.trading_engine.core.regime_detector import RegimeDetector
from src.trading_engine.psychology.market_psychology import MarketPsychologyEngine
from src.trading_engine.strategies.base_strategy import Signal, SignalAction, BaseStrategy


class TestSignalProcessor:
    """Test Signal Processor functionality"""
    
    @pytest.fixture
    def signal_processor(self):
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
        return SignalProcessor(config)
    
    @pytest.fixture
    def sample_signal(self):
        return Signal(
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
    
    @pytest.mark.asyncio
    async def test_signal_processing_pipeline(self, signal_processor, sample_signal):
        """Test complete signal processing pipeline"""
        processed_signal = await signal_processor.process(sample_signal)
        
        assert processed_signal is not None
        assert processed_signal.symbol == "AAPL"
        assert processed_signal.action == SignalAction.BUY
        assert processed_signal.strength >= 0
        assert processed_signal.confidence >= 0
        assert 'confirmations' in processed_signal.indicators
        assert 'final_score' in processed_signal.indicators
    
    @pytest.mark.asyncio
    async def test_signal_validation(self, signal_processor):
        """Test signal validation logic"""
        # Test weak signal
        weak_signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=30.0,  # Below threshold
            confidence=0.8,
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00")
        )
        
        result = await signal_processor.process(weak_signal)
        assert result is None
        
        # Test low confidence signal
        low_confidence_signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=75.0,
            confidence=0.3,  # Below threshold
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00")
        )
        
        result = await signal_processor.process(low_confidence_signal)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_risk_reward_validation(self, signal_processor):
        """Test risk/reward ratio validation"""
        # Test poor risk/reward ratio
        poor_rr_signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=75.0,
            confidence=0.8,
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("152.00")  # Poor R/R ratio
        )
        
        result = await signal_processor.process(poor_rr_signal)
        assert result is None
    
    def test_kalman_filter_creation(self, signal_processor):
        """Test Kalman filter creation"""
        kf = signal_processor._create_kalman_filter()
        
        assert 'x' in kf
        assert 'P' in kf
        assert 'F' in kf
        assert 'H' in kf
        assert 'R' in kf
        assert 'Q' in kf
        
        # Test filter application
        filtered_value = signal_processor._apply_kalman(kf, 75.0, 0.01, 0.1)
        assert isinstance(filtered_value, float)
    
    def test_correlation_calculation(self, signal_processor):
        """Test indicator correlation calculation"""
        indicators1 = {'rsi': 50, 'macd': 0.5, 'adx': 25}
        indicators2 = {'rsi': 55, 'macd': 0.6, 'adx': 30}
        
        correlation = signal_processor._calculate_indicator_correlation(indicators1, indicators2)
        assert 0 <= correlation <= 1
    
    def test_quality_score_calculation(self, signal_processor, sample_signal):
        """Test quality score calculation"""
        quality_score = signal_processor._calculate_quality_score(sample_signal)
        assert 0 <= quality_score <= 1
    
    def test_statistics(self, signal_processor):
        """Test statistics generation"""
        stats = signal_processor.get_statistics()
        
        assert 'total_processed' in stats
        assert 'symbols_tracked' in stats
        assert 'quality_metrics' in stats
        assert 'average_strength' in stats
        assert 'average_confidence' in stats


class TestRegimeDetector:
    """Test Market Regime Detector functionality"""
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data manager"""
        mock = Mock()
        mock.get_historical_data = AsyncMock()
        mock.get_quote = AsyncMock()
        return mock
    
    @pytest.fixture
    def regime_detector(self, mock_market_data):
        config = {
            'regime_lookback_periods': 60,
            'regime_update_frequency': 300,
            'bull_regime_threshold': 0.6,
            'bear_regime_threshold': -0.6,
            'high_volatility_threshold': 30,
            'trend_strength_threshold': 25
        }
        return RegimeDetector(mock_market_data, config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Add some noise for high/low
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        volumes = np.random.randint(1000000, 5000000, 100)
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    @pytest.mark.asyncio
    async def test_regime_detection(self, regime_detector, mock_market_data, sample_market_data):
        """Test regime detection functionality"""
        # Mock market data responses
        mock_market_data.get_historical_data.return_value = sample_market_data
        mock_market_data.get_quote.return_value = {'last': 20}  # VIX level
        
        regime = await regime_detector.detect_regime()
        
        assert regime in ['bull', 'bear', 'neutral', 'volatile']
        assert regime_detector.current_regime == regime
    
    @pytest.mark.asyncio
    async def test_trend_regime_detection(self, regime_detector, sample_market_data):
        """Test trend-based regime detection"""
        trend_result = await regime_detector._detect_trend_regime(sample_market_data)
        
        assert 'regime' in trend_result
        assert 'confidence' in trend_result
        assert 'adx' in trend_result
        assert 'slope' in trend_result
        assert trend_result['regime'] in ['bull', 'bear', 'neutral']
    
    @pytest.mark.asyncio
    async def test_volatility_regime_detection(self, regime_detector, sample_market_data):
        """Test volatility-based regime detection"""
        vol_result = await regime_detector._detect_volatility_regime(sample_market_data)
        
        assert 'regime' in vol_result
        assert 'confidence' in vol_result
        assert 'hvol' in vol_result
        assert 'atr_pct' in vol_result
        assert vol_result['regime'] in ['bull', 'bear', 'neutral', 'volatile']
    
    @pytest.mark.asyncio
    async def test_hmm_regime_detection(self, regime_detector, sample_market_data):
        """Test HMM-based regime detection"""
        hmm_result = await regime_detector._detect_hmm_regime(sample_market_data)
        
        assert 'regime' in hmm_result
        assert 'confidence' in hmm_result
        assert 'probabilities' in hmm_result
        assert hmm_result['regime'] in ['bull', 'bear', 'neutral', 'volatile']
    
    @pytest.mark.asyncio
    async def test_breadth_regime_detection(self, regime_detector, mock_market_data):
        """Test market breadth regime detection"""
        # Mock sector data
        mock_sector_data = pd.DataFrame({
            'close': [100, 101]  # Slight increase
        })
        mock_market_data.get_historical_data.return_value = mock_sector_data
        
        breadth_result = await regime_detector._detect_breadth_regime()
        
        assert 'regime' in breadth_result
        assert 'confidence' in breadth_result
        assert 'advancing' in breadth_result
        assert 'declining' in breadth_result
        assert 'breadth_ratio' in breadth_result
    
    @pytest.mark.asyncio
    async def test_momentum_regime_detection(self, regime_detector, sample_market_data):
        """Test momentum-based regime detection"""
        momentum_result = await regime_detector._detect_momentum_regime(sample_market_data)
        
        assert 'regime' in momentum_result
        assert 'confidence' in momentum_result
        assert 'rsi' in momentum_result
        assert 'macd_histogram' in momentum_result
        assert 'momentum_score' in momentum_result
    
    def test_regime_combination(self, regime_detector):
        """Test regime signal combination"""
        signals = {
            'trend': {'regime': 'bull', 'confidence': 0.8},
            'volatility': {'regime': 'neutral', 'confidence': 0.6},
            'hmm': {'regime': 'bull', 'confidence': 0.7},
            'breadth': {'regime': 'bull', 'confidence': 0.9},
            'momentum': {'regime': 'bull', 'confidence': 0.8}
        }
        
        final_regime = regime_detector._combine_regime_signals(signals)
        assert final_regime in ['bull', 'bear', 'neutral', 'volatile']
    
    def test_parameter_adjustment(self, regime_detector):
        """Test parameter adjustment for different regimes"""
        base_params = {
            'position_size': 1000,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'confidence_threshold': 0.7
        }
        
        # Test bull market adjustments
        regime_detector.current_regime = 'bull'
        adjusted = regime_detector.adjust_parameters_for_regime(base_params)
        
        assert adjusted['position_size'] > base_params['position_size']
        assert adjusted['stop_loss'] > base_params['stop_loss']
        assert adjusted['take_profit'] > base_params['take_profit']
    
    def test_regime_characteristics(self, regime_detector):
        """Test regime characteristics retrieval"""
        characteristics = regime_detector.get_regime_characteristics()
        
        assert 'volatility' in characteristics
        assert 'trend' in characteristics
        assert 'momentum' in characteristics
        assert 'breadth' in characteristics


class TestMarketPsychologyEngine:
    """Test Market Psychology Engine functionality"""
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data manager"""
        mock = Mock()
        mock.get_historical_data = AsyncMock()
        mock.get_quote = AsyncMock()
        return mock
    
    @pytest.fixture
    def psychology_engine(self, mock_market_data):
        config = {
            'fear_greed_extreme_threshold': 20,
            'greed_extreme_threshold': 80
        }
        return MarketPsychologyEngine(mock_market_data, config)
    
    @pytest.fixture
    def sample_spy_data(self):
        """Sample SPY data for testing"""
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
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    @pytest.mark.asyncio
    async def test_fear_greed_calculation(self, psychology_engine, mock_market_data, sample_spy_data):
        """Test fear/greed index calculation"""
        # Mock market data responses
        mock_market_data.get_historical_data.return_value = sample_spy_data
        mock_market_data.get_quote.return_value = {'last': 20}  # VIX level
        
        fear_greed = await psychology_engine._calculate_fear_greed_index()
        
        assert 0 <= fear_greed <= 100
        assert psychology_engine.fear_greed_index == fear_greed
        assert psychology_engine.market_sentiment in ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
    
    @pytest.mark.asyncio
    async def test_market_psychology_analysis(self, psychology_engine, mock_market_data, sample_spy_data):
        """Test comprehensive market psychology analysis"""
        mock_market_data.get_historical_data.return_value = sample_spy_data
        mock_market_data.get_quote.return_value = {'last': 20}
        
        psychology = await psychology_engine.get_market_psychology()
        
        assert 'fear_greed_index' in psychology
        assert 'sentiment' in psychology
        assert 'crowd_behavior' in psychology
        assert 'smart_money' in psychology
        assert 'wyckoff_phase' in psychology
        assert 'manipulation_risk' in psychology
    
    @pytest.mark.asyncio
    async def test_wyckoff_phase_identification(self, psychology_engine, mock_market_data, sample_spy_data):
        """Test Wyckoff phase identification"""
        mock_market_data.get_historical_data.return_value = sample_spy_data
        
        wyckoff_phase = await psychology_engine._identify_wyckoff_phase()
        
        assert wyckoff_phase in ['accumulation', 'markup', 'distribution', 'markdown', 'unknown']
        assert psychology_engine.wyckoff_phase == wyckoff_phase
    
    @pytest.mark.asyncio
    async def test_manipulation_detection(self, psychology_engine):
        """Test manipulation pattern detection"""
        manipulation = await psychology_engine.detect_manipulation()
        
        # Should return None or dict with manipulation patterns
        assert manipulation is None or isinstance(manipulation, dict)
    
    @pytest.mark.asyncio
    async def test_signal_psychology_analysis(self, psychology_engine):
        """Test signal analysis from psychology perspective"""
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=75.0,
            confidence=0.8,
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00")
        )
        
        # Set some psychology state
        psychology_engine.market_sentiment = "extreme_fear"
        psychology_engine.wyckoff_phase = "accumulation"
        psychology_engine.manipulation_detected = False
        
        psychology_score = await psychology_engine.analyze_signal(signal)
        
        assert 0 <= psychology_score <= 1
    
    def test_wyckoff_pattern_detection(self, psychology_engine):
        """Test Wyckoff pattern detection methods"""
        # Test accumulation patterns
        close = np.array([100, 101, 99, 102, 100, 101, 99, 102, 100, 101] * 10)
        volume = np.array([1000000] * 100)
        high = close * 1.01
        low = close * 0.99
        
        is_accumulation = psychology_engine._check_accumulation_patterns(close, volume, high, low)
        assert isinstance(is_accumulation, bool)
        
        # Test markup patterns
        is_markup = psychology_engine._check_markup_patterns(close, volume)
        assert isinstance(is_markup, bool)
        
        # Test distribution patterns
        is_distribution = psychology_engine._check_distribution_patterns(close, volume, high, low)
        assert isinstance(is_distribution, bool)
        
        # Test markdown patterns
        is_markdown = psychology_engine._check_markdown_patterns(close, volume)
        assert isinstance(is_markdown, bool)


class TestIntegration:
    """Integration tests for Step 5 components"""
    
    @pytest.mark.asyncio
    async def test_signal_processor_with_regime_detector(self):
        """Test signal processor integration with regime detector"""
        # Mock market data
        mock_market_data = Mock()
        mock_market_data.get_historical_data = AsyncMock()
        mock_market_data.get_quote = AsyncMock()
        
        # Create components
        regime_config = {'regime_lookback_periods': 60}
        regime_detector = RegimeDetector(mock_market_data, regime_config)
        
        signal_config = {'min_signal_strength': 50}
        signal_processor = SignalProcessor(signal_config)
        
        # Create signal with regime alignment
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=75.0,
            confidence=0.8,
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
            regime_alignment=0.8
        )
        
        # Process signal
        processed_signal = await signal_processor.process(signal)
        
        assert processed_signal is not None
        assert processed_signal.regime_alignment == 0.8
    
    @pytest.mark.asyncio
    async def test_psychology_engine_with_signal_processor(self):
        """Test psychology engine integration with signal processor"""
        # Mock market data
        mock_market_data = Mock()
        mock_market_data.get_historical_data = AsyncMock()
        mock_market_data.get_quote = AsyncMock()
        
        # Create components
        psychology_config = {'fear_greed_extreme_threshold': 20}
        psychology_engine = MarketPsychologyEngine(mock_market_data, psychology_config)
        
        signal_config = {'min_signal_strength': 50}
        signal_processor = SignalProcessor(signal_config)
        
        # Create signal with psychology score
        signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strength=75.0,
            confidence=0.8,
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
            psychology_score=0.7
        )
        
        # Process signal
        processed_signal = await signal_processor.process(signal)
        
        assert processed_signal is not None
        assert processed_signal.psychology_score == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

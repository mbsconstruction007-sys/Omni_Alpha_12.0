"""
Infrastructure Components for Institutional Trading
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================
# INFRASTRUCTURE COMPONENTS
# ============================================

class DataPipeline:
    """Data ingestion and processing pipeline"""
    
    def __init__(self):
        self.data_sources = {}
        self.market_data = {}
        self.data_quality_monitor = DataQualityMonitor()
        
    async def initialize(self):
        """Initialize data connections"""
        await self._connect_data_sources()
        await self.data_quality_monitor.initialize()
        
    async def _connect_data_sources(self):
        """Connect to data sources"""
        self.data_sources = {
            'market_data': MarketDataProvider(),
            'fundamental_data': FundamentalDataProvider(),
            'alternative_data': AlternativeDataProvider(),
            'news_data': NewsDataProvider()
        }
    
    async def get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        # This would connect to real data sources
        market_data = {
            'prices': await self._get_prices(),
            'order_book': await self._get_order_book(),
            'trades': await self._get_trades(),
            'fundamentals': await self._get_fundamentals(),
            'alternative_data': await self._get_alternative_data(),
            'timestamp': datetime.now()
        }
        
        # Monitor data quality
        await self.data_quality_monitor.check_quality(market_data)
        
        return market_data
    
    async def get_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        # Simulate price
        return np.random.uniform(50, 500)
    
    async def _get_prices(self) -> Dict[str, float]:
        """Get current prices"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        return {symbol: np.random.uniform(50, 500) for symbol in symbols}
    
    async def _get_order_book(self) -> Dict[str, Any]:
        """Get order book data"""
        return {
            'bids': [{'price': 100.0, 'size': 1000}],
            'asks': [{'price': 100.1, 'size': 1000}]
        }
    
    async def _get_trades(self) -> List[Dict[str, Any]]:
        """Get recent trades"""
        return [
            {'price': 100.05, 'size': 500, 'side': 'BUY', 'timestamp': datetime.now()},
            {'price': 100.03, 'size': 300, 'side': 'SELL', 'timestamp': datetime.now()}
        ]
    
    async def _get_fundamentals(self) -> Dict[str, Any]:
        """Get fundamental data"""
        return {
            'pe_ratio': 25.0,
            'pb_ratio': 3.0,
            'roe': 0.15,
            'debt_to_equity': 0.3
        }
    
    async def _get_alternative_data(self) -> Dict[str, Any]:
        """Get alternative data"""
        return {
            'satellite_data': {'activity_score': 0.8},
            'social_sentiment': 0.6,
            'web_data': {'search_volume': 1000}
        }

class EventBus:
    """Event-driven architecture support"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = deque(maxlen=10000)
        
    async def publish(self, event_type: str, data: Any):
        """Publish event"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now()
        }
        
        self.event_history.append(event)
        
        # Notify subscribers
        for subscriber in self.subscribers[event_type]:
            try:
                asyncio.create_task(subscriber(data))
            except Exception as e:
                logger.error(f"Event subscriber error: {str(e)}")
    
    def subscribe(self, event_type: str, handler):
        """Subscribe to event"""
        self.subscribers[event_type].append(handler)
    
    def get_event_history(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get event history"""
        if event_type:
            return [event for event in self.event_history if event['type'] == event_type]
        return list(self.event_history)

class PerformanceTracker:
    """Track trading performance"""
    
    def __init__(self):
        self.metrics = {}
        self.pnl_history = deque(maxlen=10000)
        self.returns_history = deque(maxlen=10000)
        
    async def update(self, positions: Dict[str, Any]):
        """Update performance metrics"""
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in positions.values())
        self.pnl_history.append(total_pnl)
        
        # Calculate returns
        if len(self.pnl_history) > 1:
            returns = np.diff(self.pnl_history) / np.array(self.pnl_history[:-1])
            self.returns_history.extend(returns)
        
        # Calculate metrics
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            
            self.metrics = {
                'total_pnl': total_pnl,
                'sharpe_ratio': self._calculate_sharpe(returns_array),
                'sortino_ratio': self._calculate_sortino(returns_array),
                'calmar_ratio': self._calculate_calmar(),
                'max_drawdown': self._calculate_drawdown(),
                'win_rate': self._calculate_win_rate(returns_array),
                'avg_win': self._calculate_avg_win(returns_array),
                'avg_loss': self._calculate_avg_loss(returns_array),
                'volatility': np.std(returns_array) * np.sqrt(252),
                'timestamp': datetime.now()
            }
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio"""
        if len(self.pnl_history) < 2:
            return 0.0
        
        max_drawdown = self._calculate_drawdown()
        if max_drawdown == 0:
            return 0.0
        
        annual_return = (self.pnl_history[-1] / self.pnl_history[0] - 1) * 252 / len(self.pnl_history)
        return annual_return / max_drawdown
    
    def _calculate_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.pnl_history:
            return 0.0
        
        peak = np.maximum.accumulate(self.pnl_history)
        drawdown = (self.pnl_history - peak) / peak
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate"""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
    
    def _calculate_avg_win(self, returns: np.ndarray) -> float:
        """Calculate average winning return"""
        winning_returns = returns[returns > 0]
        return np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
    
    def _calculate_avg_loss(self, returns: np.ndarray) -> float:
        """Calculate average losing return"""
        losing_returns = returns[returns < 0]
        return np.mean(losing_returns) if len(losing_returns) > 0 else 0.0
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

class ComplianceEngine:
    """Regulatory compliance monitoring"""
    
    def __init__(self):
        self.rules = []
        self.violations = []
        
    async def initialize(self):
        """Initialize compliance rules"""
        self.rules = [
            PositionLimitRule(),
            SectorLimitRule(),
            LeverageLimitRule(),
            InsiderTradingRule(),
            MarketManipulationRule()
        ]
        
    async def check_trades(self, portfolio: Dict[str, float]) -> bool:
        """Check trades for compliance"""
        violations = []
        
        for rule in self.rules:
            try:
                violation = await rule.check(portfolio)
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.error(f"Compliance rule error: {str(e)}")
        
        if violations:
            self.violations.extend(violations)
            logger.warning(f"Compliance violations: {violations}")
            return False
        
        return True

class MachineLearningFactory:
    """Factory for ML models"""
    
    def __init__(self):
        self.models = {}
        
    async def initialize(self):
        """Initialize ML models"""
        # This would load actual trained models
        self.models = {
            'price_predictor': MockMLModel('price_predictor'),
            'volatility_forecaster': MockMLModel('volatility_forecaster'),
            'regime_classifier': MockMLModel('regime_classifier')
        }
    
    def get_model(self, model_name: str):
        """Get ML model by name"""
        return self.models.get(model_name)

class FeatureStore:
    """Feature storage and management"""
    
    def __init__(self):
        self.features = {}
        
    async def store_features(self, features: Dict[str, Any]):
        """Store features"""
        self.features.update(features)
    
    async def get_features(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get features by name"""
        return {name: self.features.get(name) for name in feature_names}

# ============================================
# SUPPORTING CLASSES
# ============================================

class DataQualityMonitor:
    """Monitor data quality"""
    
    async def initialize(self):
        """Initialize data quality monitoring"""
        pass
    
    async def check_quality(self, data: Dict[str, Any]):
        """Check data quality"""
        # Mock data quality checks
        pass

class DataProvider(ABC):
    """Base class for data providers"""
    
    @abstractmethod
    async def get_data(self) -> Dict[str, Any]:
        """Get data from provider"""
        pass

class MarketDataProvider(DataProvider):
    """Market data provider"""
    
    async def get_data(self) -> Dict[str, Any]:
        """Get market data"""
        return {'prices': {}, 'volume': {}}

class FundamentalDataProvider(DataProvider):
    """Fundamental data provider"""
    
    async def get_data(self) -> Dict[str, Any]:
        """Get fundamental data"""
        return {'pe_ratio': 25.0, 'pb_ratio': 3.0}

class AlternativeDataProvider(DataProvider):
    """Alternative data provider"""
    
    async def get_data(self) -> Dict[str, Any]:
        """Get alternative data"""
        return {'satellite': {}, 'social': {}}

class NewsDataProvider(DataProvider):
    """News data provider"""
    
    async def get_data(self) -> Dict[str, Any]:
        """Get news data"""
        return {'sentiment': 0.5, 'headlines': []}

class ComplianceRule(ABC):
    """Base class for compliance rules"""
    
    @abstractmethod
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check compliance rule"""
        pass

class PositionLimitRule(ComplianceRule):
    """Position limit compliance rule"""
    
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check position limits"""
        for symbol, weight in portfolio.items():
            if abs(weight) > 0.10:  # Max 10% in any position
                return f"Position limit exceeded for {symbol}: {weight:.2%}"
        return None

class SectorLimitRule(ComplianceRule):
    """Sector limit compliance rule"""
    
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check sector limits"""
        # Mock sector limit check
        return None

class LeverageLimitRule(ComplianceRule):
    """Leverage limit compliance rule"""
    
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check leverage limits"""
        total_exposure = sum(abs(weight) for weight in portfolio.values())
        if total_exposure > 3.0:  # Max 3x leverage
            return f"Leverage limit exceeded: {total_exposure:.2f}"
        return None

class InsiderTradingRule(ComplianceRule):
    """Insider trading compliance rule"""
    
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check for insider trading"""
        # Mock insider trading check
        return None

class MarketManipulationRule(ComplianceRule):
    """Market manipulation compliance rule"""
    
    async def check(self, portfolio: Dict[str, float]) -> Optional[str]:
        """Check for market manipulation"""
        # Mock market manipulation check
        return None

class MockMLModel:
    """Mock ML model for testing"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def predict(self, data: Dict[str, Any]) -> float:
        """Make prediction"""
        return np.random.uniform(-1, 1)

"""
STEP 17: Complete Portfolio Optimization & Multi-Strategy Orchestration
Unified system coordinating all 16 previous trading strategies
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm, multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Advanced optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available, using scipy optimization")
    
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import pickle

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Fix datetime import for classes
from datetime import datetime

# ===================== DATA STRUCTURES =====================

@dataclass
class StrategyConfig:
    """Configuration for each trading strategy"""
    name: str
    step_number: int
    enabled: bool
    min_allocation: float
    max_allocation: float
    risk_budget: float
    priority: int
    dependencies: List[str]
    current_allocation: float = 0.0
    current_performance: float = 0.0
    confidence: float = 1.0

@dataclass
class MarketRegime:
    """Current market regime detection"""
    regime_type: str  # BULL, BEAR, HIGH_VOL, LOW_VOL, RANGING
    confidence: float
    volatility: float
    trend_strength: float
    correlation_regime: str
    detected_at: datetime
    features: Dict[str, float]

@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_value: float
    cash: float
    positions: Dict[str, float]
    strategy_allocations: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_rebalance: datetime
    current_drawdown: float

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float
    risk_contributions: Dict[str, float]
    optimization_method: str

# ===================== PORTFOLIO OPTIMIZER =====================

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization with multiple methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.returns_history = pd.DataFrame()
        self.covariance_matrix = None
        self.expected_returns = None
        
    def optimize(self, method: str = 'HRP') -> OptimizationResult:
        """Main optimization method selector"""
        
        methods = {
            'MARKOWITZ': self.markowitz_optimization,
            'BLACK_LITTERMAN': self.black_litterman_optimization,
            'HRP': self.hierarchical_risk_parity,
            'RISK_PARITY': self.risk_parity,
            'MAX_DIVERSIFICATION': self.maximum_diversification,
            'MIN_VARIANCE': self.minimum_variance,
            'MAX_SHARPE': self.maximum_sharpe
        }
        
        if method in methods:
            return methods[method]()
        else:
            logger.warning(f"Unknown optimization method: {method}")
            return self.equal_weight_optimization()
    
    def prepare_data(self, returns_df: pd.DataFrame):
        """Prepare returns and covariance data"""
        
        self.returns_history = returns_df
        
        # Robust covariance estimation using Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        self.covariance_matrix = pd.DataFrame(
            lw.fit(returns_df).covariance_,
            index=returns_df.columns,
            columns=returns_df.columns
        )
        
        # Expected returns using multiple methods
        self.expected_returns = self.calculate_expected_returns(returns_df)
    
    def calculate_expected_returns(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate expected returns using ensemble methods"""
        
        # Historical mean
        historical_mean = returns_df.mean()
        
        # Exponentially weighted mean (recent data more important)
        ewm_mean = returns_df.ewm(span=60).mean().iloc[-1]
        
        # CAPM-adjusted returns (simplified)
        market_return = 0.12  # Expected market return
        risk_free = 0.065  # Risk-free rate
        betas = returns_df.corrwith(returns_df.mean(axis=1))
        capm_returns = risk_free + betas * (market_return - risk_free)
        
        # Ensemble average
        expected_returns = (historical_mean + ewm_mean + capm_returns) / 3
        
        return expected_returns
    
    # ========== MARKOWITZ OPTIMIZATION ==========
    
    def markowitz_optimization(self, target_return: Optional[float] = None) -> OptimizationResult:
        """Classic mean-variance optimization"""
        
        if not CVXPY_AVAILABLE:
            return self._markowitz_scipy()
        
        n_assets = len(self.expected_returns)
        
        # Optimization variables
        weights = cp.Variable(n_assets)
        
        # Expected portfolio return
        portfolio_return = self.expected_returns.values @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix.values)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0.05,  # Minimum 5% per strategy
            weights <= 0.40   # Maximum 40% per strategy
        ]
        
        if target_return:
            constraints.append(portfolio_return >= target_return)
            # Minimize risk for target return
            objective = cp.Minimize(portfolio_variance)
        else:
            # Maximize Sharpe ratio (using quadratic approximation)
            risk_free = 0.065
            objective = cp.Maximize((portfolio_return - risk_free) / cp.sqrt(portfolio_variance))
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if weights.value is None:
                logger.error("Markowitz optimization failed")
                return self.equal_weight_optimization()
            
            # Create result
            weights_dict = dict(zip(self.expected_returns.index, weights.value))
            
            return self.create_optimization_result(weights_dict, 'MARKOWITZ')
        except Exception as e:
            logger.error(f"CVXPY optimization error: {e}")
            return self._markowitz_scipy()
    
    def _markowitz_scipy(self) -> OptimizationResult:
        """Markowitz using scipy as fallback"""
        
        n_assets = len(self.expected_returns)
        
        def objective(weights):
            return weights @ self.covariance_matrix.values @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0.05, 0.40) for _ in range(n_assets)]
        
        result = minimize(objective, np.ones(n_assets)/n_assets, 
                         method='SLSQP', constraints=constraints, bounds=bounds)
        
        weights_dict = dict(zip(self.expected_returns.index, result.x))
        return self.create_optimization_result(weights_dict, 'MARKOWITZ_SCIPY')
    
    # ========== HIERARCHICAL RISK PARITY (HRP) ==========
    
    def hierarchical_risk_parity(self) -> OptimizationResult:
        """Lopez de Prado's HRP algorithm"""
        
        try:
            # Step 1: Tree clustering
            corr_matrix = self.returns_history.corr()
            dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Step 2: Quasi-diagonalization
            link = self._tree_clustering(dist_matrix)
            sorted_idx = self._get_quasi_diag(link)
            sorted_tickers = corr_matrix.index[sorted_idx].tolist()
            
            # Step 3: Recursive bisection
            weights = self._recursive_bisection(
                self.covariance_matrix.loc[sorted_tickers, sorted_tickers]
            )
            
            weights_dict = dict(zip(sorted_tickers, weights))
            
            return self.create_optimization_result(weights_dict, 'HRP')
        except Exception as e:
            logger.error(f"HRP optimization error: {e}")
            return self.equal_weight_optimization()
    
    def _tree_clustering(self, dist_matrix):
        """Perform hierarchical tree clustering"""
        try:
            from scipy.cluster.hierarchy import linkage
            return linkage(dist_matrix.values, 'single')
        except ImportError:
            logger.warning("Scipy clustering not available")
            return None
    
    def _get_quasi_diag(self, link):
        """Get quasi-diagonal using recursive bisection"""
        try:
            from scipy.cluster.hierarchy import to_tree
            
            if link is None:
                return list(range(len(self.expected_returns)))
            
            root = to_tree(link)
            
            def get_leaves(node):
                if node.is_leaf():
                    return [node.id]
                
                left_leaves = get_leaves(node.get_left())
                right_leaves = get_leaves(node.get_right())
                
                return left_leaves + right_leaves
            
            return get_leaves(root)
        except:
            return list(range(len(self.expected_returns)))
    
    def _recursive_bisection(self, cov, weights=None):
        """Perform recursive bisection for HRP"""
        
        if weights is None:
            weights = np.ones(cov.shape[0]) / cov.shape[0]
        
        if len(cov) == 1:
            return weights
        
        # Split covariance matrix
        n = len(cov) // 2
        cov_1 = cov.iloc[:n, :n]
        cov_2 = cov.iloc[n:, n:]
        
        # Calculate inverse variance weights
        ivp_1 = 1 / np.diag(cov_1)
        ivp_2 = 1 / np.diag(cov_2)
        
        w_1 = ivp_1 / ivp_1.sum()
        w_2 = ivp_2 / ivp_2.sum()
        
        # Calculate cluster variances
        var_1 = w_1 @ cov_1 @ w_1
        var_2 = w_2 @ cov_2 @ w_2
        
        # Allocate between clusters
        alpha = var_2 / (var_1 + var_2)
        
        weights[:n] *= alpha
        weights[n:] *= (1 - alpha)
        
        return weights
    
    # ========== RISK PARITY ==========
    
    def risk_parity(self) -> OptimizationResult:
        """Equal Risk Contribution portfolio"""
        
        n_assets = len(self.expected_returns)
        
        def risk_contribution(weights):
            """Calculate risk contribution of each asset"""
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix.values @ weights)
            marginal_contrib = self.covariance_matrix.values @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            """Minimize difference in risk contributions"""
            contrib = risk_contribution(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Bounds
        bounds = [(0.05, 0.40) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', 
                         constraints=constraints, bounds=bounds)
        
        weights_dict = dict(zip(self.expected_returns.index, result.x))
        
        return self.create_optimization_result(weights_dict, 'RISK_PARITY')
    
    # ========== MAXIMUM SHARPE ==========
    
    def maximum_sharpe(self) -> OptimizationResult:
        """Maximum Sharpe ratio portfolio"""
        
        n_assets = len(self.expected_returns)
        
        def negative_sharpe(weights):
            """Negative Sharpe ratio for minimization"""
            portfolio_return = weights @ self.expected_returns.values
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix.values @ weights)
            risk_free = 0.065
            return -(portfolio_return - risk_free) / portfolio_vol
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0.05, 0.40) for _ in range(n_assets)]
        
        result = minimize(negative_sharpe, np.ones(n_assets)/n_assets,
                         method='SLSQP', constraints=constraints, bounds=bounds)
        
        weights_dict = dict(zip(self.expected_returns.index, result.x))
        
        return self.create_optimization_result(weights_dict, 'MAX_SHARPE')
    
    # ========== MINIMUM VARIANCE ==========
    
    def minimum_variance(self) -> OptimizationResult:
        """Minimum variance portfolio"""
        
        n_assets = len(self.expected_returns)
        
        def objective(weights):
            return weights @ self.covariance_matrix.values @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0.05, 0.40) for _ in range(n_assets)]
        
        result = minimize(objective, np.ones(n_assets)/n_assets,
                         method='SLSQP', constraints=constraints, bounds=bounds)
        
        weights_dict = dict(zip(self.expected_returns.index, result.x))
        
        return self.create_optimization_result(weights_dict, 'MIN_VARIANCE')
    
    # ========== MAXIMUM DIVERSIFICATION ==========
    
    def maximum_diversification(self) -> OptimizationResult:
        """Maximum diversification portfolio"""
        
        n_assets = len(self.expected_returns)
        asset_vols = np.sqrt(np.diag(self.covariance_matrix.values))
        
        def negative_diversification_ratio(weights):
            """Negative diversification ratio for minimization"""
            weighted_avg_vol = weights @ asset_vols
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix.values @ weights)
            return -weighted_avg_vol / portfolio_vol
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0.05, 0.40) for _ in range(n_assets)]
        
        result = minimize(negative_diversification_ratio, np.ones(n_assets)/n_assets,
                         method='SLSQP', constraints=constraints, bounds=bounds)
        
        weights_dict = dict(zip(self.expected_returns.index, result.x))
        
        return self.create_optimization_result(weights_dict, 'MAX_DIVERSIFICATION')
    
    # ========== BLACK-LITTERMAN ==========
    
    def black_litterman_optimization(self, views: Optional[Dict] = None) -> OptimizationResult:
        """Black-Litterman model with investor views"""
        
        # Market capitalization weights (simplified - equal weight as proxy)
        w_market = np.ones(len(self.expected_returns)) / len(self.expected_returns)
        
        # Risk aversion parameter
        delta = 2.5
        
        # Equilibrium returns
        Pi = delta * self.covariance_matrix.values @ w_market
        
        if views:
            # Incorporate views (simplified implementation)
            # P: View matrix, Q: View returns, Omega: Uncertainty
            P = views.get('P', np.eye(len(self.expected_returns)))
            Q = views.get('Q', Pi)
            Omega = views.get('Omega', np.eye(len(Q)) * 0.01)
            
            # Black-Litterman formula
            tau = 0.05
            M = np.linalg.inv(
                np.linalg.inv(tau * self.covariance_matrix.values) + 
                P.T @ np.linalg.inv(Omega) @ P
            )
            
            expected_returns_bl = M @ (
                np.linalg.inv(tau * self.covariance_matrix.values) @ Pi + 
                P.T @ np.linalg.inv(Omega) @ Q
            )
        else:
            expected_returns_bl = Pi
        
        # Use the BL returns in standard optimization
        original_returns = self.expected_returns.copy()
        self.expected_returns = pd.Series(expected_returns_bl, index=self.expected_returns.index)
        
        result = self.markowitz_optimization()
        result.optimization_method = 'BLACK_LITTERMAN'
        
        # Restore original returns
        self.expected_returns = original_returns
        
        return result
    
    # ========== HELPER METHODS ==========
    
    def create_optimization_result(self, weights: Dict[str, float], 
                                  method: str) -> OptimizationResult:
        """Create standardized optimization result"""
        
        weights_array = np.array(list(weights.values()))
        
        # Calculate metrics
        expected_return = weights_array @ self.expected_returns.values
        portfolio_variance = weights_array @ self.covariance_matrix.values @ weights_array
        portfolio_risk = np.sqrt(portfolio_variance)
        
        risk_free = 0.065
        sharpe_ratio = (expected_return - risk_free) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Diversification ratio
        asset_risks = np.sqrt(np.diag(self.covariance_matrix.values))
        weighted_avg_risk = weights_array @ asset_risks
        diversification_ratio = weighted_avg_risk / portfolio_risk if portfolio_risk > 0 else 1
        
        # Effective N (Herfindahl)
        effective_n = 1 / np.sum(weights_array ** 2) if np.sum(weights_array ** 2) > 0 else 1
        
        # Risk contributions
        if portfolio_risk > 0:
            marginal_contrib = self.covariance_matrix.values @ weights_array
            risk_contributions = weights_array * marginal_contrib / portfolio_risk
        else:
            risk_contributions = weights_array
            
        risk_contrib_dict = dict(zip(weights.keys(), risk_contributions))
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_n=effective_n,
            risk_contributions=risk_contrib_dict,
            optimization_method=method
        )
    
    def equal_weight_optimization(self) -> OptimizationResult:
        """Fallback equal weight portfolio"""
        
        n_assets = len(self.expected_returns)
        weights = {asset: 1/n_assets for asset in self.expected_returns.index}
        
        return self.create_optimization_result(weights, 'EQUAL_WEIGHT')

# ===================== MARKET REGIME DETECTOR =====================

class MarketRegimeDetector:
    """Detect current market regime using multiple methods"""
    
    def __init__(self):
        self.current_regime = None
        self.regime_history = []
        
    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        
        # Calculate features
        returns = market_data['close'].pct_change()
        
        # Volatility regime
        volatility = returns.rolling(20).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1] if not volatility.empty else 0.15
        
        # Trend regime
        if len(market_data) >= 200:
            sma_50 = market_data['close'].rolling(50).mean()
            sma_200 = market_data['close'].rolling(200).mean()
            trend_strength = (sma_50.iloc[-1] - sma_200.iloc[-1]) / sma_200.iloc[-1]
        else:
            trend_strength = 0
        
        # Determine regime
        if current_vol > 0.30:
            regime_type = 'HIGH_VOL'
        elif current_vol < 0.12:
            regime_type = 'LOW_VOL'
        elif abs(trend_strength) < 0.02:
            regime_type = 'RANGING'
        elif trend_strength > 0.05:
            regime_type = 'BULL'
        else:
            regime_type = 'BEAR'
        
        # Correlation regime
        if len(market_data) > 100:
            rolling_corr = 0.5  # Simplified
            correlation_regime = 'HIGH_CORR' if rolling_corr > 0.6 else 'LOW_CORR'
        else:
            correlation_regime = 'UNKNOWN'
        
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=0.75,  # Simplified
            volatility=current_vol,
            trend_strength=trend_strength,
            correlation_regime=correlation_regime,
            detected_at=datetime.now(),
            features={
                'volatility': current_vol,
                'trend': trend_strength,
                'correlation': rolling_corr if len(market_data) > 100 else 0
            }
        )
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime

# ===================== STRATEGY ORCHESTRATOR =====================

class MultiStrategyOrchestrator:
    """Orchestrate multiple trading strategies"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.active_signals = {}
        self.execution_queue = []
        
    def _initialize_strategies(self) -> Dict[str, StrategyConfig]:
        """Initialize all trading strategies"""
        
        strategies = {
            'OPTIONS': StrategyConfig(
                name='Options Trading',
                step_number=16,
                enabled=True,
                min_allocation=0.10,
                max_allocation=0.40,
                risk_budget=30,
                priority=1,
                dependencies=[]
            ),
            'ML_PREDICTIONS': StrategyConfig(
                name='ML Platform',
                step_number=6,
                enabled=True,
                min_allocation=0.05,
                max_allocation=0.25,
                risk_budget=25,
                priority=2,
                dependencies=[]
            ),
            'MICROSTRUCTURE': StrategyConfig(
                name='Microstructure',
                step_number=13,
                enabled=True,
                min_allocation=0.05,
                max_allocation=0.20,
                risk_budget=20,
                priority=3,
                dependencies=[]
            ),
            'SENTIMENT': StrategyConfig(
                name='Sentiment Analysis',
                step_number=14,
                enabled=True,
                min_allocation=0.05,
                max_allocation=0.15,
                risk_budget=15,
                priority=4,
                dependencies=['ML_PREDICTIONS']
            ),
            'ALTERNATIVE_DATA': StrategyConfig(
                name='Alternative Data',
                step_number=15,
                enabled=True,
                min_allocation=0.05,
                max_allocation=0.15,
                risk_budget=10,
                priority=5,
                dependencies=['SENTIMENT']
            ),
            'COMPREHENSIVE_AI': StrategyConfig(
                name='Comprehensive AI',
                step_number=14.1,
                enabled=True,
                min_allocation=0.05,
                max_allocation=0.20,
                risk_budget=15,
                priority=3,
                dependencies=['ML_PREDICTIONS']
            )
        }
        
        return strategies
    
    async def collect_signals(self) -> Dict[str, Any]:
        """Collect signals from all active strategies"""
        
        signals = {}
        
        for name, strategy in self.strategies.items():
            if strategy.enabled:
                # In production, call actual strategy signal methods
                signal = await self._get_strategy_signal(name)
                if signal:
                    signals[name] = signal
        
        self.active_signals = signals
        return signals
    
    async def _get_strategy_signal(self, strategy_name: str) -> Optional[Dict]:
        """Get signal from specific strategy"""
        
        # Simulate strategy signals with different characteristics
        if strategy_name == 'OPTIONS':
            return {
                'direction': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3]),
                'confidence': np.random.uniform(0.7, 0.95),
                'expected_return': np.random.uniform(0.01, 0.03),
                'risk': np.random.uniform(0.005, 0.015),
                'size': np.random.uniform(0.05, 0.15),
                'strategy_type': 'HEDGED_OPTIONS'
            }
        elif strategy_name == 'ML_PREDICTIONS':
            return {
                'direction': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.45, 0.35, 0.2]),
                'confidence': np.random.uniform(0.6, 0.85),
                'expected_return': np.random.uniform(0.005, 0.025),
                'risk': np.random.uniform(0.01, 0.02),
                'size': np.random.uniform(0.03, 0.12),
                'strategy_type': 'ML_PREDICTION'
            }
        elif strategy_name == 'MICROSTRUCTURE':
            return {
                'direction': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.35, 0.35, 0.3]),
                'confidence': np.random.uniform(0.65, 0.80),
                'expected_return': np.random.uniform(0.002, 0.015),
                'risk': np.random.uniform(0.008, 0.018),
                'size': np.random.uniform(0.02, 0.08),
                'strategy_type': 'MICROSTRUCTURE'
            }
        else:
            return {
                'direction': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': np.random.uniform(0.5, 0.75),
                'expected_return': np.random.uniform(-0.01, 0.02),
                'risk': np.random.uniform(0.01, 0.03),
                'size': np.random.uniform(0.01, 0.10),
                'strategy_type': 'GENERAL'
            }
    
    def resolve_conflicts(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between strategy signals"""
        
        if not signals:
            return {}
        
        # Check for conflicts
        directions = [s['direction'] for s in signals.values()]
        
        if len(set(directions)) == 1:
            # No conflict
            return self._aggregate_signals(signals)
        
        # Conflict resolution by weighted consensus
        buy_weight = sum(s['confidence'] for s in signals.values() if s['direction'] == 'BUY')
        sell_weight = sum(s['confidence'] for s in signals.values() if s['direction'] == 'SELL')
        hold_weight = sum(s['confidence'] for s in signals.values() if s['direction'] == 'HOLD')
        
        total_weight = buy_weight + sell_weight + hold_weight
        
        if total_weight == 0:
            return {}
        
        consensus = {
            'direction': max(
                [('BUY', buy_weight), ('SELL', sell_weight), ('HOLD', hold_weight)],
                key=lambda x: x[1]
            )[0],
            'confidence': max(buy_weight, sell_weight, hold_weight) / total_weight,
            'expected_return': np.mean([s['expected_return'] for s in signals.values()]),
            'risk': np.mean([s['risk'] for s in signals.values()]),
            'size': np.mean([s['size'] for s in signals.values()]),
            'contributing_strategies': len(signals)
        }
        
        return consensus
    
    def _aggregate_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate non-conflicting signals"""
        
        return {
            'direction': list(signals.values())[0]['direction'],
            'confidence': np.mean([s['confidence'] for s in signals.values()]),
            'expected_return': np.mean([s['expected_return'] for s in signals.values()]),
            'risk': np.mean([s['risk'] for s in signals.values()]),
            'size': sum(s['size'] for s in signals.values()),
            'contributing_strategies': len(signals)
        }
    
    def prioritize_execution(self, consensus_signal: Dict) -> List[Dict]:
        """Prioritize execution based on strategy priorities"""
        
        execution_plan = []
        
        # Sort strategies by priority
        sorted_strategies = sorted(
            self.strategies.items(),
            key=lambda x: x[1].priority
        )
        
        for name, strategy in sorted_strategies:
            if name in self.active_signals:
                execution_plan.append({
                    'strategy': name,
                    'signal': self.active_signals[name],
                    'allocation': strategy.current_allocation,
                    'priority': strategy.priority
                })
        
        return execution_plan

# ===================== PORTFOLIO MANAGER =====================

class IntegratedPortfolioManager:
    """Main portfolio management system integrating all components"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.capital = initial_capital
        self.optimizer = AdvancedPortfolioOptimizer({})
        self.regime_detector = MarketRegimeDetector()
        self.orchestrator = MultiStrategyOrchestrator()
        
        self.portfolio_state = PortfolioState(
            total_value=initial_capital,
            cash=initial_capital,
            positions={},
            strategy_allocations={},
            risk_metrics={},
            performance_metrics={},
            last_rebalance=datetime.now(),
            current_drawdown=0
        )
        
        self.performance_history = []
        self.rebalance_history = []
        
    async def run_portfolio_optimization_cycle(self):
        """Main portfolio optimization and execution cycle"""
        
        try:
            # Step 1: Detect market regime
            market_data = await self._fetch_market_data()
            regime = await self.regime_detector.detect_regime(market_data)
            logger.info(f"Detected regime: {regime.regime_type}")
            
            # Step 2: Adjust strategy parameters based on regime
            self._adjust_strategies_for_regime(regime)
            
            # Step 3: Collect signals from all strategies
            signals = await self.orchestrator.collect_signals()
            
            # Step 4: Resolve conflicts
            consensus = self.orchestrator.resolve_conflicts(signals)
            
            # Step 5: Optimize portfolio allocation
            optimization_result = await self._optimize_allocation(regime)
            
            # Step 6: Check if rebalancing needed
            if self._should_rebalance(optimization_result):
                await self._execute_rebalance(optimization_result)
            
            # Step 7: Execute new signals
            if consensus and consensus['direction'] != 'HOLD':
                await self._execute_trades(consensus, optimization_result)
            
            # Step 8: Update portfolio state
            self._update_portfolio_state()
            
            # Step 9: Performance attribution
            self._perform_attribution_analysis()
            
            # Step 10: Risk monitoring
            self._monitor_risk()
            
        except Exception as e:
            logger.error(f"Portfolio optimization cycle error: {e}")
    
    async def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data for regime detection"""
        
        # In production, fetch real market data
        # Placeholder implementation
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(252).cumsum() + 100,
            'volume': np.random.uniform(1000000, 5000000, 252)
        }, index=dates)
        
        return data
    
    def _adjust_strategies_for_regime(self, regime: MarketRegime):
        """Adjust strategy parameters based on market regime"""
        
        if regime.regime_type == 'HIGH_VOL':
            # Increase options allocation
            self.orchestrator.strategies['OPTIONS'].max_allocation = 0.50
            self.orchestrator.strategies['OPTIONS'].risk_budget = 40
            
            # Reduce directional strategies
            self.orchestrator.strategies['ML_PREDICTIONS'].max_allocation = 0.15
            
        elif regime.regime_type == 'BULL':
            # Increase directional long strategies
            self.orchestrator.strategies['ML_PREDICTIONS'].max_allocation = 0.35
            self.orchestrator.strategies['SENTIMENT'].max_allocation = 0.20
            
        elif regime.regime_type == 'BEAR':
            # Increase hedging and short strategies
            self.orchestrator.strategies['OPTIONS'].max_allocation = 0.45
            self.orchestrator.strategies['MICROSTRUCTURE'].max_allocation = 0.25
            
        elif regime.regime_type == 'RANGING':
            # Increase mean reversion strategies
            self.orchestrator.strategies['MICROSTRUCTURE'].max_allocation = 0.30
            self.orchestrator.strategies['ALTERNATIVE_DATA'].max_allocation = 0.20
    
    async def _optimize_allocation(self, regime: MarketRegime) -> OptimizationResult:
        """Optimize portfolio allocation"""
        
        # Prepare returns data for each strategy
        strategy_returns = await self._get_strategy_returns()
        
        self.optimizer.prepare_data(strategy_returns)
        
        # Choose optimization method based on regime
        if regime.regime_type in ['HIGH_VOL', 'BEAR']:
            method = 'HRP'  # More robust in unstable markets
        elif regime.regime_type == 'BULL':
            method = 'MAX_SHARPE'
        else:
            method = 'RISK_PARITY'
        
        result = self.optimizer.optimize(method)
        
        return result
    
    async def _get_strategy_returns(self) -> pd.DataFrame:
        """Get historical returns for each strategy"""
        
        # In production, fetch actual strategy returns
        # Placeholder with simulated returns
        
        strategies = list(self.orchestrator.strategies.keys())
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        returns_data = {}
        for strategy in strategies:
            # Simulate different return characteristics
            if strategy == 'OPTIONS':
                returns_data[strategy] = np.random.randn(252) * 0.015 + 0.001
            elif strategy == 'ML_PREDICTIONS':
                returns_data[strategy] = np.random.randn(252) * 0.020 + 0.0005
            else:
                returns_data[strategy] = np.random.randn(252) * 0.018
        
        return pd.DataFrame(returns_data, index=dates)
    
    def _should_rebalance(self, optimization_result: OptimizationResult) -> bool:
        """Determine if rebalancing is needed"""
        
        # Check time since last rebalance
        time_since_rebalance = datetime.now() - self.portfolio_state.last_rebalance
        if time_since_rebalance.days < 7:
            return False
        
        # Check deviation from target weights
        current_weights = self.portfolio_state.strategy_allocations
        target_weights = optimization_result.weights
        
        max_deviation = 0
        for strategy in target_weights:
            current = current_weights.get(strategy, 0)
            target = target_weights[strategy]
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)
        
        # Rebalance if deviation > 5%
        return max_deviation > 0.05
    
    async def _execute_rebalance(self, optimization_result: OptimizationResult):
        """Execute portfolio rebalancing"""
        
        logger.info(f"Executing rebalance: {optimization_result.weights}")
        
        # Calculate trades needed
        current = self.portfolio_state.strategy_allocations
        target = optimization_result.weights
        
        trades = {}
        for strategy in target:
            current_value = current.get(strategy, 0) * self.portfolio_state.total_value
            target_value = target[strategy] * self.portfolio_state.total_value
            trade_value = target_value - current_value
            
            if abs(trade_value) > 1000:  # Minimum trade size
                trades[strategy] = trade_value
        
        # Execute trades (placeholder)
        for strategy, value in trades.items():
            logger.info(f"Rebalancing {strategy}: ₹{value:,.2f}")
        
        # Update allocations
        self.portfolio_state.strategy_allocations = target
        self.portfolio_state.last_rebalance = datetime.now()
        
        # Record rebalance
        self.rebalance_history.append({
            'timestamp': datetime.now(),
            'old_weights': current,
            'new_weights': target,
            'trades': trades,
            'reason': 'THRESHOLD_BREACH'
        })
    
    async def _execute_trades(self, signal: Dict, optimization: OptimizationResult):
        """Execute trades based on consensus signal"""
        
        # Calculate position size based on optimization
        total_size = signal['size'] * self.portfolio_state.total_value
        
        # Distribute across strategies based on weights
        for strategy, weight in optimization.weights.items():
            if strategy in self.orchestrator.active_signals:
                strategy_size = total_size * weight
                
                logger.info(f"Executing {signal['direction']} for {strategy}: ₹{strategy_size:,.2f}")
                
                # In production, execute actual trades
                # Update positions
                self.portfolio_state.positions[strategy] = \
                    self.portfolio_state.positions.get(strategy, 0) + strategy_size
    
    def _update_portfolio_state(self):
        """Update portfolio state and metrics"""
        
        # Calculate total value
        total_value = self.portfolio_state.cash + sum(self.portfolio_state.positions.values())
        
        # Calculate returns
        returns = (total_value - self.capital) / self.capital
        
        # Update metrics
        self.portfolio_state.total_value = total_value
        self.portfolio_state.performance_metrics['total_return'] = returns
        self.portfolio_state.performance_metrics['sharpe_ratio'] = self._calculate_sharpe()
        
        # Risk metrics
        self.portfolio_state.risk_metrics['var_95'] = self._calculate_var()
        self.portfolio_state.risk_metrics['max_drawdown'] = self._calculate_max_drawdown()
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'value': total_value,
            'return': returns,
            'cash': self.portfolio_state.cash
        })
        
    def _perform_attribution_analysis(self):
        """Perform performance attribution analysis"""
        
        attribution = {}
        
        for strategy in self.orchestrator.strategies:
            if strategy in self.portfolio_state.positions:
                position_value = self.portfolio_state.positions[strategy]
                weight = position_value / self.portfolio_state.total_value if self.portfolio_state.total_value > 0 else 0
                
                # Simplified attribution
                attribution[strategy] = {
                    'weight': weight,
                    'contribution': weight * 0.10,  # Placeholder return
                    'risk_contribution': weight * 0.15  # Placeholder risk
                }
        
        self.portfolio_state.performance_metrics['attribution'] = attribution
    
    def _monitor_risk(self):
        """Monitor portfolio risk metrics"""
        
        # Check risk limits
        if self.portfolio_state.risk_metrics.get('var_95', 0) > 50000:
            logger.warning("VaR limit approaching!")
        
        if self.portfolio_state.current_drawdown < -0.10:
            logger.warning(f"Significant drawdown: {self.portfolio_state.current_drawdown:.1%}")
        
        # Check concentration risk
        if self.portfolio_state.strategy_allocations:
            max_allocation = max(self.portfolio_state.strategy_allocations.values())
            if max_allocation > 0.45:
                logger.warning(f"High concentration: {max_allocation:.1%}")
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio"""
        
        if len(self.performance_history) < 20:
            return 0
        
        returns = pd.Series([p['return'] for p in self.performance_history[-252:]])
        
        if returns.std() == 0:
            return 0
        
        return (returns.mean() - 0.065/252) / returns.std() * np.sqrt(252)
    
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        
        if len(self.performance_history) < 20:
            return 0
        
        returns = pd.Series([p['return'] for p in self.performance_history[-252:]])
        
        if len(returns) < 20:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100) * self.portfolio_state.total_value
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        
        if len(self.performance_history) < 2:
            return 0
        
        values = pd.Series([p['value'] for p in self.performance_history])
        
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        
        return drawdown.min()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive portfolio report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_state.total_value,
            'cash': self.portfolio_state.cash,
            'positions': self.portfolio_state.positions,
            'allocations': self.portfolio_state.strategy_allocations,
            'performance': self.portfolio_state.performance_metrics,
            'risk': self.portfolio_state.risk_metrics,
            'regime': self.regime_detector.current_regime.__dict__ if self.regime_detector.current_regime else None,
            'active_strategies': len([s for s in self.orchestrator.strategies.values() if s.enabled]),
            'last_rebalance': self.portfolio_state.last_rebalance.isoformat(),
            'health_score': self._calculate_health_score()
        }
        
        return report
    
    def _calculate_health_score(self) -> float:
        """Calculate overall portfolio health score"""
        
        score = 100.0
        
        # Deduct for poor performance
        if self.portfolio_state.performance_metrics.get('sharpe_ratio', 0) < 1.0:
            score -= 20
        
        # Deduct for high risk
        if self.portfolio_state.current_drawdown < -0.10:
            score -= 30
        
        # Deduct for concentration
        if self.portfolio_state.strategy_allocations:
            max_weight = max(self.portfolio_state.strategy_allocations.values())
            if max_weight > 0.40:
                score -= 10
        
        return max(0, score)

# ===================== AI PORTFOLIO ADVISOR =====================

class AIPortfolioAdvisor:
    """AI-powered portfolio advisory using Gemini"""
    
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        
    async def get_allocation_recommendation(self, 
                                           portfolio_state: PortfolioState,
                                           market_regime: MarketRegime) -> Dict:
        """Get AI recommendation for portfolio allocation"""
        
        # In production, call Gemini API
        # Placeholder response based on regime
        
        if market_regime.regime_type == 'HIGH_VOL':
            return {
                'recommended_allocation': {
                    'OPTIONS': 0.40,
                    'ML_PREDICTIONS': 0.20,
                    'MICROSTRUCTURE': 0.15,
                    'SENTIMENT': 0.10,
                    'ALTERNATIVE_DATA': 0.10,
                    'COMPREHENSIVE_AI': 0.05
                },
                'emphasis': ['OPTIONS', 'MICROSTRUCTURE'],
                'reduce': ['SENTIMENT', 'ALTERNATIVE_DATA'],
                'risk_adjustments': 'Increase hedging due to high volatility',
                'rebalancing_urgency': 8
            }
        elif market_regime.regime_type == 'BULL':
            return {
                'recommended_allocation': {
                    'ML_PREDICTIONS': 0.30,
                    'SENTIMENT': 0.20,
                    'OPTIONS': 0.25,
                    'COMPREHENSIVE_AI': 0.15,
                    'MICROSTRUCTURE': 0.05,
                    'ALTERNATIVE_DATA': 0.05
                },
                'emphasis': ['ML_PREDICTIONS', 'SENTIMENT'],
                'reduce': ['MICROSTRUCTURE'],
                'risk_adjustments': 'Increase directional exposure',
                'rebalancing_urgency': 6
            }
        else:
            return {
                'recommended_allocation': {
                    'OPTIONS': 0.25,
                    'ML_PREDICTIONS': 0.20,
                    'MICROSTRUCTURE': 0.20,
                    'COMPREHENSIVE_AI': 0.15,
                    'SENTIMENT': 0.10,
                    'ALTERNATIVE_DATA': 0.10
                },
                'emphasis': ['OPTIONS', 'MICROSTRUCTURE'],
                'reduce': [],
                'risk_adjustments': 'Balanced allocation for uncertain regime',
                'rebalancing_urgency': 5
            }
    
    async def analyze_strategy_performance(self, 
                                          performance_data: Dict) -> Dict:
        """Analyze individual strategy performance"""
        
        # Placeholder analysis
        return {
            'best_performers': ['OPTIONS', 'ML_PREDICTIONS'],
            'underperformers': ['SENTIMENT'],
            'correlation_insights': 'Strategies showing good diversification',
            'suggestions': [
                'Options performing well in current regime',
                'ML predictions showing consistent alpha',
                'Consider reducing sentiment allocation during high volatility',
                'Microstructure effective during ranging markets'
            ]
        }

# ===================== TELEGRAM BOT INTEGRATION =====================

def integrate_portfolio_optimization(bot_instance):
    """Integrate portfolio optimization with Telegram bot"""
    
    # Initialize portfolio manager
    bot_instance.portfolio_manager = IntegratedPortfolioManager()
    bot_instance.ai_advisor = AIPortfolioAdvisor(os.getenv('GEMINI_API_KEY'))
    
    async def portfolio_command(update, context):
        """Main portfolio command handler"""
        
        if not context.args:
            help_text = """
📊 **Portfolio Optimization System**

**Commands:**
/portfolio status - Current portfolio state
/portfolio optimize - Run optimization cycle
/portfolio rebalance - Force rebalancing
/portfolio risk - Risk metrics
/portfolio performance - Performance report
/portfolio allocation - View allocations
/portfolio regime - Market regime
/portfolio health - Health check
/portfolio ai - AI recommendations
            """
            await update.message.reply_text(help_text, parse_mode='Markdown')
            return
        
        command = context.args[0].lower()
        
        if command == 'status':
            report = bot_instance.portfolio_manager.generate_report()
            msg = f"""
📊 **Portfolio Status**

**Value:** ₹{report['portfolio_value']:,.2f}
**Cash:** ₹{report['cash']:,.2f}
**Positions:** {len(report['positions'])}
**Active Strategies:** {report['active_strategies']}

**Performance:**
• Total Return: {report['performance'].get('total_return', 0):.2%}
• Sharpe Ratio: {report['performance'].get('sharpe_ratio', 0):.2f}

**Risk:**
• VaR(95%): ₹{report['risk'].get('var_95', 0):,.2f}
• Max Drawdown: {report['risk'].get('max_drawdown', 0):.2%}

**Health Score:** {report['health_score']:.0f}/100
**Last Rebalance:** {report['last_rebalance'][:10]}
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'optimize':
            await update.message.reply_text("🔄 Running portfolio optimization cycle...")
            await bot_instance.portfolio_manager.run_portfolio_optimization_cycle()
            await update.message.reply_text("✅ Portfolio optimization cycle complete!")
        
        elif command == 'allocation':
            allocations = bot_instance.portfolio_manager.portfolio_state.strategy_allocations
            
            if allocations:
                msg = "📈 **Current Strategy Allocations:**\n\n"
                for strategy, weight in allocations.items():
                    value = weight * bot_instance.portfolio_manager.portfolio_state.total_value
                    msg += f"• **{strategy}:** {weight:.1%} (₹{value:,.2f})\n"
            else:
                msg = "No allocations set yet. Run /portfolio optimize first."
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'regime':
            regime = bot_instance.portfolio_manager.regime_detector.current_regime
            
            if regime:
                msg = f"""
🌐 **Market Regime Analysis**

**Type:** {regime.regime_type}
**Confidence:** {regime.confidence:.1%}
**Volatility:** {regime.volatility:.2%}
**Trend Strength:** {regime.trend_strength:.2%}
**Correlation:** {regime.correlation_regime}

**Detected:** {regime.detected_at.strftime('%Y-%m-%d %H:%M')}

**Features:**
• Volatility: {regime.features.get('volatility', 0):.2%}
• Trend: {regime.features.get('trend', 0):.2%}
• Correlation: {regime.features.get('correlation', 0):.2f}
                """
            else:
                msg = "No regime detected yet. Run /portfolio optimize first."
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'ai':
            state = bot_instance.portfolio_manager.portfolio_state
            regime = bot_instance.portfolio_manager.regime_detector.current_regime
            
            if regime:
                recommendation = await bot_instance.ai_advisor.get_allocation_recommendation(
                    state, regime
                )
                
                msg = f"""
🤖 **AI Portfolio Recommendations**

**Recommended Allocation:**
"""
                for strategy, allocation in recommendation['recommended_allocation'].items():
                    msg += f"• {strategy}: {allocation:.1%}\n"
                
                msg += f"""
**📈 Emphasize:** {', '.join(recommendation['emphasis'])}
**📉 Reduce:** {', '.join(recommendation['reduce']) if recommendation['reduce'] else 'None'}

**🛡️ Risk Adjustments:**
{recommendation['risk_adjustments']}

**⚡ Rebalancing Urgency:** {recommendation['rebalancing_urgency']}/10
                """
            else:
                msg = "Please run /portfolio optimize first to detect market regime"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'performance':
            state = bot_instance.portfolio_manager.portfolio_state
            
            msg = f"""
📈 **Portfolio Performance**

**Returns:**
• Total Return: {state.performance_metrics.get('total_return', 0):.2%}
• Sharpe Ratio: {state.performance_metrics.get('sharpe_ratio', 0):.2f}

**Risk Metrics:**
• VaR (95%): ₹{state.risk_metrics.get('var_95', 0):,.2f}
• Max Drawdown: {state.risk_metrics.get('max_drawdown', 0):.2%}
• Current Drawdown: {state.current_drawdown:.2%}

**Attribution:**
"""
            
            attribution = state.performance_metrics.get('attribution', {})
            for strategy, attr in attribution.items():
                msg += f"• {strategy}: {attr['contribution']:.2%} contribution\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'risk':
            state = bot_instance.portfolio_manager.portfolio_state
            
            msg = f"""
🛡️ **Risk Analysis**

**Portfolio Risk:**
• VaR (95%): ₹{state.risk_metrics.get('var_95', 0):,.2f}
• Max Drawdown: {state.risk_metrics.get('max_drawdown', 0):.2%}
• Current Drawdown: {state.current_drawdown:.2%}

**Concentration Risk:**
"""
            
            if state.strategy_allocations:
                max_weight = max(state.strategy_allocations.values())
                max_strategy = max(state.strategy_allocations.items(), key=lambda x: x[1])
                
                msg += f"• Largest Position: {max_strategy[0]} ({max_weight:.1%})\n"
                msg += f"• Concentration Level: {'⚠️ HIGH' if max_weight > 0.4 else '✅ OK'}\n"
            
            msg += f"""
**Risk Budget Utilization:**
• Total Risk Budget: 100%
• Current Utilization: {sum(state.strategy_allocations.values()) * 100:.1f}%
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'health':
            report = bot_instance.portfolio_manager.generate_report()
            health_score = report['health_score']
            
            if health_score >= 80:
                status = "🟢 EXCELLENT"
            elif health_score >= 60:
                status = "🟡 GOOD"
            elif health_score >= 40:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
            
            msg = f"""
💊 **Portfolio Health Check**

**Overall Health:** {status} ({health_score:.0f}/100)

**Health Factors:**
• Performance: {'✅' if report['performance'].get('sharpe_ratio', 0) >= 1.0 else '⚠️'}
• Risk Control: {'✅' if report['risk'].get('max_drawdown', 0) > -0.10 else '⚠️'}
• Diversification: {'✅' if max(report['allocations'].values()) <= 0.40 else '⚠️'} if report['allocations'] else '⚠️'

**Recommendations:**
"""
            
            if health_score < 80:
                if report['performance'].get('sharpe_ratio', 0) < 1.0:
                    msg += "• Improve strategy selection\n"
                if report['risk'].get('max_drawdown', 0) < -0.10:
                    msg += "• Reduce position sizes\n"
                if report['allocations'] and max(report['allocations'].values()) > 0.40:
                    msg += "• Diversify allocations\n"
            else:
                msg += "• Portfolio health is excellent!\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
    
    return portfolio_command

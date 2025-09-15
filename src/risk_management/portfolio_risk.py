"""
Portfolio Risk Management Module
Comprehensive portfolio-level risk analysis and management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize

logger = structlog.get_logger()

@dataclass
class PortfolioMetrics:
    """Portfolio risk metrics"""
    total_value: float
    total_risk: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    tracking_error: float
    correlation_matrix: np.ndarray
    concentration_risk: float
    liquidity_risk: float

class PortfolioRiskManager:
    """Advanced portfolio risk management and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_history = []
        self.returns_history = []
        self.positions = {}
        self.benchmark_returns = []
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    async def calculate_comprehensive_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        portfolio_value = await self._get_portfolio_value()
        returns = await self._get_portfolio_returns()
        positions = await self._get_current_positions()
        
        # Basic metrics
        total_risk = await self._calculate_total_risk(positions)
        volatility = await self._calculate_volatility(returns)
        
        # VaR calculations
        var_95 = await self._calculate_var(returns, 0.95)
        var_99 = await self._calculate_var(returns, 0.99)
        expected_shortfall = await self._calculate_expected_shortfall(returns, 0.95)
        
        # Performance ratios
        sharpe_ratio = await self._calculate_sharpe_ratio(returns)
        sortino_ratio = await self._calculate_sortino_ratio(returns)
        calmar_ratio = await self._calculate_calmar_ratio(returns)
        
        # Drawdown analysis
        max_drawdown = await self._calculate_max_drawdown()
        current_drawdown = await self._calculate_current_drawdown()
        
        # Risk-adjusted metrics
        beta = await self._calculate_beta(returns)
        alpha = await self._calculate_alpha(returns, beta)
        information_ratio = await self._calculate_information_ratio(returns)
        treynor_ratio = await self._calculate_treynor_ratio(returns, beta)
        jensen_alpha = await self._calculate_jensen_alpha(returns, beta)
        tracking_error = await self._calculate_tracking_error(returns)
        
        # Risk concentration
        correlation_matrix = await self._calculate_correlation_matrix(positions)
        concentration_risk = await self._calculate_concentration_risk(positions)
        liquidity_risk = await self._calculate_liquidity_risk(positions)
        
        return PortfolioMetrics(
            total_value=portfolio_value,
            total_risk=total_risk,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk
        )
    
    async def calculate_new_risk_with_position(self, order: Dict) -> float:
        """Calculate portfolio risk if we add this position"""
        current_positions = await self._get_current_positions()
        
        # Add the new position
        new_positions = current_positions.copy()
        new_positions[order["symbol"]] = {
            "quantity": order["quantity"],
            "price": order["price"],
            "value": order["quantity"] * order["price"]
        }
        
        # Calculate new portfolio risk
        return await self._calculate_total_risk(new_positions)
    
    async def optimize_portfolio_risk(self) -> Dict[str, float]:
        """Optimize portfolio for minimum risk"""
        positions = await self._get_current_positions()
        if not positions:
            return {}
        
        # Get historical returns for each position
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        # Calculate covariance matrix
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov().values
        
        # Portfolio optimization using mean-variance optimization
        n_assets = len(positions)
        
        # Equal weight starting point
        x0 = np.array([1/n_assets] * n_assets)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: no short selling, max 20% per position
        bounds = [(0, 0.2) for _ in range(n_assets)]
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Return optimized weights
        symbols = list(positions.keys())
        optimized_weights = {}
        for i, symbol in enumerate(symbols):
            optimized_weights[symbol] = result.x[i]
        
        return optimized_weights
    
    async def calculate_risk_contribution(self) -> Dict[str, float]:
        """Calculate risk contribution of each position"""
        positions = await self._get_current_positions()
        if not positions:
            return {}
        
        # Get returns data
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov().values
        
        # Calculate portfolio weights
        total_value = sum(pos["value"] for pos in positions.values())
        weights = np.array([pos["value"] / total_value for pos in positions.values()])
        
        # Calculate risk contributions
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contributions = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance
        
        # Return as dictionary
        symbols = list(positions.keys())
        risk_contrib = {}
        for i, symbol in enumerate(symbols):
            risk_contrib[symbol] = risk_contributions[i] * 100  # As percentage
        
        return risk_contrib
    
    async def stress_test_portfolio(self, scenarios: Dict[str, Dict]) -> Dict[str, float]:
        """Stress test portfolio under various scenarios"""
        positions = await self._get_current_positions()
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            total_loss = 0.0
            
            for symbol, position in positions.items():
                # Apply scenario shock to this position
                shock = scenario_params.get(symbol, scenario_params.get("market", 0))
                position_loss = position["value"] * abs(shock) / 100
                total_loss += position_loss
            
            # Calculate loss as percentage of portfolio
            portfolio_value = await self._get_portfolio_value()
            loss_percentage = (total_loss / portfolio_value) * 100
            results[scenario_name] = loss_percentage
        
        return results
    
    async def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.portfolio_history:
            return 0.0
        
        peak = max(self.portfolio_history)
        current = self.portfolio_history[-1]
        
        if peak == 0:
            return 0.0
        
        return ((peak - current) / peak) * 100
    
    async def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        peak = self.portfolio_history[0]
        max_dd = 0.0
        
        for value in self.portfolio_history[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    async def calculate_sharpe_ratio(self, returns: List[float] = None) -> float:
        """Calculate Sharpe Ratio"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        risk_free_rate_daily = self.risk_free_rate / 252
        return np.sqrt(252) * (avg_return - risk_free_rate_daily) / std_return
    
    async def calculate_sortino_ratio(self, returns: List[float] = None) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        risk_free_rate_daily = self.risk_free_rate / 252
        return np.sqrt(252) * (avg_return - risk_free_rate_daily) / downside_std
    
    async def calculate_calmar_ratio(self, returns: List[float] = None) -> float:
        """Calculate Calmar Ratio (annual return / max drawdown)"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        if not returns:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = await self.calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd
    
    async def calculate_beta(self, returns: List[float] = None) -> float:
        """Calculate portfolio beta relative to benchmark"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 1.0
        
        # Ensure same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    async def calculate_alpha(self, returns: List[float] = None, beta: float = None) -> float:
        """Calculate portfolio alpha"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        if beta is None:
            beta = await self.calculate_beta(returns)
        
        benchmark_returns = await self._get_benchmark_returns()
        
        if not returns or not benchmark_returns:
            return 0.0
        
        portfolio_return = np.mean(returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        
        # Alpha = Portfolio Return - (Risk-free Rate + Beta * (Benchmark Return - Risk-free Rate))
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return alpha
    
    async def calculate_information_ratio(self, returns: List[float] = None) -> float:
        """Calculate Information Ratio"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        # Calculate excess returns
        min_length = min(len(returns), len(benchmark_returns))
        excess_returns = [r - b for r, b in zip(returns[-min_length:], benchmark_returns[-min_length:])]
        
        if not excess_returns:
            return 0.0
        
        avg_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(252) * avg_excess_return / tracking_error
    
    async def calculate_treynor_ratio(self, returns: List[float] = None, beta: float = None) -> float:
        """Calculate Treynor Ratio"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        if beta is None:
            beta = await self.calculate_beta(returns)
        
        if beta == 0:
            return 0.0
        
        portfolio_return = np.mean(returns) * 252
        return (portfolio_return - self.risk_free_rate) / beta
    
    async def calculate_jensen_alpha(self, returns: List[float] = None, beta: float = None) -> float:
        """Calculate Jensen's Alpha"""
        return await self.calculate_alpha(returns, beta)
    
    async def calculate_tracking_error(self, returns: List[float] = None) -> float:
        """Calculate Tracking Error"""
        if returns is None:
            returns = await self._get_portfolio_returns()
        
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        min_length = min(len(returns), len(benchmark_returns))
        excess_returns = [r - b for r, b in zip(returns[-min_length:], benchmark_returns[-min_length:])]
        
        return np.std(excess_returns) * np.sqrt(252)
    
    # Helper methods
    
    async def _calculate_total_risk(self, positions: Dict) -> float:
        """Calculate total portfolio risk"""
        if not positions:
            return 0.0
        
        # Get returns data for all positions
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        if not returns_data:
            return 0.0
        
        # Calculate portfolio variance
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov().values
        
        # Calculate weights
        total_value = sum(pos["value"] for pos in positions.values())
        weights = np.array([pos["value"] / total_value for pos in positions.values()])
        
        # Portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        return np.sqrt(portfolio_variance) * np.sqrt(252) * 100  # Annualized as percentage
    
    async def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate portfolio volatility"""
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252) * 100  # Annualized as percentage
    
    async def _calculate_var(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return abs(var) * 100  # As percentage
    
    async def _calculate_expected_shortfall(self, returns: List[float], confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if not returns:
            return 0.0
        
        var = await self._calculate_var(returns, confidence_level)
        
        # Calculate average of returns below VaR
        var_threshold = -var / 100  # Convert back to decimal
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return var
        
        expected_shortfall = np.mean(tail_returns)
        return abs(expected_shortfall) * 100  # As percentage
    
    async def _calculate_correlation_matrix(self, positions: Dict) -> np.ndarray:
        """Calculate correlation matrix of positions"""
        if len(positions) < 2:
            return np.array([[1.0]])
        
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        if not returns_data:
            return np.array([[1.0]])
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr().values
    
    async def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        if not positions:
            return 0.0
        
        total_value = sum(pos["value"] for pos in positions.values())
        weights = [pos["value"] / total_value for pos in positions.values()]
        
        # Herfindahl index
        herfindahl = sum(w**2 for w in weights)
        
        return herfindahl * 100  # As percentage
    
    async def _calculate_liquidity_risk(self, positions: Dict) -> float:
        """Calculate portfolio liquidity risk"""
        if not positions:
            return 0.0
        
        liquidity_scores = []
        for symbol in positions.keys():
            liquidity = await self._get_symbol_liquidity(symbol)
            liquidity_scores.append(liquidity)
        
        # Average liquidity risk (inverted - higher score = lower risk)
        avg_liquidity = np.mean(liquidity_scores)
        return (1 - avg_liquidity) * 100  # Convert to risk percentage
    
    # Data access methods (placeholders)
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return 100000.0  # Placeholder
    
    async def _get_portfolio_returns(self) -> List[float]:
        """Get historical portfolio returns"""
        return [0.001] * 252  # Placeholder
    
    async def _get_current_positions(self) -> Dict:
        """Get current portfolio positions"""
        return {}  # Placeholder
    
    async def _get_symbol_returns(self, symbol: str) -> List[float]:
        """Get historical returns for a symbol"""
        return [0.001] * 252  # Placeholder
    
    async def _get_benchmark_returns(self) -> List[float]:
        """Get benchmark returns"""
        return [0.0008] * 252  # Placeholder
    
    async def _get_symbol_liquidity(self, symbol: str) -> float:
        """Get liquidity score for a symbol"""
        return 0.8  # Placeholder

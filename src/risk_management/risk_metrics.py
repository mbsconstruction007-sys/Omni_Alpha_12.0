"""
Risk Metrics Module
Comprehensive risk metrics calculation and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from scipy import stats
import asyncio

logger = structlog.get_logger()

@dataclass
class RiskMetricsResult:
    """Risk metrics calculation result"""
    timestamp: datetime
    portfolio_value: float
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
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    tail_risk: float
    skewness: float
    kurtosis: float

class RiskMetrics:
    """Comprehensive risk metrics calculation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = []
        self.benchmark_data = {}
        self.risk_free_rate = 0.02  # 2% annual
    
    async def calculate_comprehensive_metrics(self) -> RiskMetricsResult:
        """Calculate comprehensive risk metrics"""
        portfolio_value = await self._get_portfolio_value()
        returns = await self._get_portfolio_returns()
        positions = await self._get_current_positions()
        
        # Basic risk metrics
        total_risk = await self._calculate_total_risk(positions)
        volatility = await self._calculate_volatility(returns)
        
        # VaR and Expected Shortfall
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
        
        # Advanced risk metrics
        correlation_risk = await self._calculate_correlation_risk(positions)
        concentration_risk = await self._calculate_concentration_risk(positions)
        liquidity_risk = await self._calculate_liquidity_risk(positions)
        tail_risk = await self._calculate_tail_risk(returns)
        
        # Distribution metrics
        skewness = await self._calculate_skewness(returns)
        kurtosis = await self._calculate_kurtosis(returns)
        
        result = RiskMetricsResult(
            timestamp=datetime.utcnow(),
            portfolio_value=portfolio_value,
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
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            tail_risk=tail_risk,
            skewness=skewness,
            kurtosis=kurtosis
        )
        
        # Store in history
        self.metrics_history.append(result)
        
        logger.info("Risk metrics calculated successfully", 
                   sharpe_ratio=sharpe_ratio, 
                   max_drawdown=max_drawdown)
        
        return result
    
    async def calculate_risk_attribution(self) -> Dict[str, float]:
        """Calculate risk attribution by position"""
        positions = await self._get_current_positions()
        if not positions:
            return {}
        
        # Get returns data for all positions
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        if not returns_data:
            return {}
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr().values
        
        # Calculate portfolio weights
        total_value = sum(pos["value"] for pos in positions.values())
        weights = np.array([pos["value"] / total_value for pos in positions.values()])
        
        # Calculate risk contributions
        portfolio_variance = np.dot(weights.T, np.dot(corr_matrix, weights))
        marginal_contributions = np.dot(corr_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance
        
        # Return as dictionary
        symbols = list(positions.keys())
        risk_attribution = {}
        for i, symbol in enumerate(symbols):
            risk_attribution[symbol] = risk_contributions[i] * 100  # As percentage
        
        return risk_attribution
    
    async def calculate_risk_budget(self, target_risk: float) -> Dict[str, float]:
        """Calculate risk budget allocation"""
        positions = await self._get_current_positions()
        if not positions:
            return {}
        
        # Get current risk contributions
        risk_contributions = await self.calculate_risk_attribution()
        
        # Calculate risk budget
        total_current_risk = sum(risk_contributions.values())
        risk_budget = {}
        
        for symbol, contribution in risk_contributions.items():
            # Allocate risk budget proportionally
            budget_allocation = (contribution / total_current_risk) * target_risk
            risk_budget[symbol] = budget_allocation
        
        return risk_budget
    
    async def calculate_risk_limits_utilization(self) -> Dict[str, float]:
        """Calculate utilization of risk limits"""
        current_metrics = await self.calculate_comprehensive_metrics()
        
        limits = {
            "max_drawdown": self.config.get("MAX_DRAWDOWN_PERCENT", 20.0),
            "max_volatility": self.config.get("MAX_VOLATILITY_PERCENT", 30.0),
            "max_var": self.config.get("MAX_VAR_PERCENT", 10.0),
            "max_concentration": self.config.get("MAX_CONCENTRATION_PERCENT", 20.0)
        }
        
        utilization = {
            "drawdown_utilization": (current_metrics.current_drawdown / limits["max_drawdown"]) * 100,
            "volatility_utilization": (current_metrics.volatility / limits["max_volatility"]) * 100,
            "var_utilization": (current_metrics.var_95 / limits["max_var"]) * 100,
            "concentration_utilization": (current_metrics.concentration_risk / limits["max_concentration"]) * 100
        }
        
        return utilization
    
    async def calculate_risk_adjusted_returns(self) -> Dict[str, float]:
        """Calculate various risk-adjusted return metrics"""
        returns = await self._get_portfolio_returns()
        if not returns:
            return {}
        
        # Calculate different risk-adjusted returns
        sharpe_ratio = await self._calculate_sharpe_ratio(returns)
        sortino_ratio = await self._calculate_sortino_ratio(returns)
        calmar_ratio = await self._calculate_calmar_ratio(returns)
        information_ratio = await self._calculate_information_ratio(returns)
        
        # Calculate return per unit of risk
        annual_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        return_per_volatility = annual_return / volatility if volatility > 0 else 0
        
        # Calculate return per unit of drawdown
        max_dd = await self._calculate_max_drawdown()
        return_per_drawdown = annual_return / max_dd if max_dd > 0 else 0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "information_ratio": information_ratio,
            "return_per_volatility": return_per_volatility,
            "return_per_drawdown": return_per_drawdown,
            "annual_return": annual_return * 100,
            "volatility": volatility * 100,
            "max_drawdown": max_dd
        }
    
    async def calculate_tail_risk_metrics(self) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        returns = await self._get_portfolio_returns()
        if len(returns) < 30:
            return {}
        
        returns_array = np.array(returns)
        
        # Calculate tail risk metrics
        var_95 = np.percentile(returns_array, 5) * 100
        var_99 = np.percentile(returns_array, 1) * 100
        expected_shortfall_95 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]) * 100
        expected_shortfall_99 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 1)]) * 100
        
        # Calculate tail ratio
        tail_ratio = abs(expected_shortfall_95 / var_95) if var_95 != 0 else 0
        
        # Calculate conditional tail expectation
        cte_95 = abs(expected_shortfall_95)
        cte_99 = abs(expected_shortfall_99)
        
        # Calculate tail dependence
        tail_dependence = await self._calculate_tail_dependence(returns_array)
        
        return {
            "var_95": abs(var_95),
            "var_99": abs(var_99),
            "expected_shortfall_95": abs(expected_shortfall_95),
            "expected_shortfall_99": abs(expected_shortfall_99),
            "tail_ratio": tail_ratio,
            "conditional_tail_expectation_95": cte_95,
            "conditional_tail_expectation_99": cte_99,
            "tail_dependence": tail_dependence
        }
    
    async def calculate_liquidity_risk_metrics(self) -> Dict[str, float]:
        """Calculate liquidity risk metrics"""
        positions = await self._get_current_positions()
        if not positions:
            return {}
        
        liquidity_metrics = {}
        
        for symbol, position in positions.items():
            # Get liquidity data for symbol
            liquidity_score = await self._get_symbol_liquidity(symbol)
            bid_ask_spread = await self._get_bid_ask_spread(symbol)
            daily_volume = await self._get_daily_volume(symbol)
            
            # Calculate liquidity risk
            position_value = position["value"]
            liquidity_cost = position_value * bid_ask_spread / 2  # Half spread cost
            
            # Calculate time to liquidate
            liquidation_time = position_value / (daily_volume * 0.1)  # 10% of daily volume
            
            liquidity_metrics[symbol] = {
                "liquidity_score": liquidity_score,
                "bid_ask_spread": bid_ask_spread,
                "liquidity_cost": liquidity_cost,
                "liquidation_time_days": liquidation_time,
                "position_size_vs_volume": position_value / daily_volume
            }
        
        # Calculate portfolio-level liquidity metrics
        total_liquidity_cost = sum(metrics["liquidity_cost"] for metrics in liquidity_metrics.values())
        max_liquidation_time = max(metrics["liquidation_time_days"] for metrics in liquidity_metrics.values())
        avg_liquidity_score = np.mean([metrics["liquidity_score"] for metrics in liquidity_metrics.values()])
        
        return {
            "individual_positions": liquidity_metrics,
            "total_liquidity_cost": total_liquidity_cost,
            "max_liquidation_time_days": max_liquidation_time,
            "average_liquidity_score": avg_liquidity_score,
            "portfolio_liquidity_risk": (1 - avg_liquidity_score) * 100
        }
    
    # Core calculation methods
    
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
        """Calculate Expected Shortfall"""
        if not returns:
            return 0.0
        
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return 0.0
        
        expected_shortfall = np.mean(tail_returns)
        return abs(expected_shortfall) * 100  # As percentage
    
    async def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        risk_free_rate_daily = self.risk_free_rate / 252
        return np.sqrt(252) * (avg_return - risk_free_rate_daily) / std_return
    
    async def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino Ratio"""
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
    
    async def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar Ratio"""
        if not returns:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = await self._calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        portfolio_history = await self._get_portfolio_history()
        
        if len(portfolio_history) < 2:
            return 0.0
        
        peak = portfolio_history[0]
        max_dd = 0.0
        
        for value in portfolio_history[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        portfolio_history = await self._get_portfolio_history()
        
        if not portfolio_history:
            return 0.0
        
        peak = max(portfolio_history)
        current = portfolio_history[-1]
        
        if peak == 0:
            return 0.0
        
        return ((peak - current) / peak) * 100
    
    async def _calculate_beta(self, returns: List[float]) -> float:
        """Calculate portfolio beta"""
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 1.0
        
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    async def _calculate_alpha(self, returns: List[float], beta: float) -> float:
        """Calculate portfolio alpha"""
        benchmark_returns = await self._get_benchmark_returns()
        
        if not returns or not benchmark_returns:
            return 0.0
        
        portfolio_return = np.mean(returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return alpha
    
    async def _calculate_information_ratio(self, returns: List[float]) -> float:
        """Calculate Information Ratio"""
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        min_length = min(len(returns), len(benchmark_returns))
        excess_returns = [r - b for r, b in zip(returns[-min_length:], benchmark_returns[-min_length:])]
        
        if not excess_returns:
            return 0.0
        
        avg_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(252) * avg_excess_return / tracking_error
    
    async def _calculate_treynor_ratio(self, returns: List[float], beta: float) -> float:
        """Calculate Treynor Ratio"""
        if beta == 0:
            return 0.0
        
        portfolio_return = np.mean(returns) * 252
        return (portfolio_return - self.risk_free_rate) / beta
    
    async def _calculate_jensen_alpha(self, returns: List[float], beta: float) -> float:
        """Calculate Jensen's Alpha"""
        return await self._calculate_alpha(returns, beta)
    
    async def _calculate_tracking_error(self, returns: List[float]) -> float:
        """Calculate Tracking Error"""
        benchmark_returns = await self._get_benchmark_returns()
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        min_length = min(len(returns), len(benchmark_returns))
        excess_returns = [r - b for r, b in zip(returns[-min_length:], benchmark_returns[-min_length:])]
        
        return np.std(excess_returns) * np.sqrt(252)
    
    async def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate correlation risk"""
        if len(positions) < 2:
            return 0.0
        
        # Get correlation matrix
        returns_data = {}
        for symbol in positions.keys():
            returns_data[symbol] = await self._get_symbol_returns(symbol)
        
        if not returns_data:
            return 0.0
        
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr().values
        
        # Calculate average correlation
        n = len(corr_matrix)
        avg_correlation = (np.sum(corr_matrix) - n) / (n * (n - 1))
        
        return avg_correlation * 100
    
    async def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        if not positions:
            return 0.0
        
        total_value = sum(pos["value"] for pos in positions.values())
        weights = [pos["value"] / total_value for pos in positions.values()]
        
        herfindahl = sum(w**2 for w in weights)
        return herfindahl * 100
    
    async def _calculate_liquidity_risk(self, positions: Dict) -> float:
        """Calculate liquidity risk"""
        if not positions:
            return 0.0
        
        liquidity_scores = []
        for symbol in positions.keys():
            liquidity = await self._get_symbol_liquidity(symbol)
            liquidity_scores.append(liquidity)
        
        avg_liquidity = np.mean(liquidity_scores)
        return (1 - avg_liquidity) * 100
    
    async def _calculate_tail_risk(self, returns: List[float]) -> float:
        """Calculate tail risk"""
        if len(returns) < 30:
            return 0.0
        
        # Calculate tail risk as the ratio of extreme losses to normal losses
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        tail_ratio = abs(var_99 / var_95) if var_95 != 0 else 0
        
        return tail_ratio * 100
    
    async def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness"""
        if len(returns) < 3:
            return 0.0
        
        return stats.skew(returns)
    
    async def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis"""
        if len(returns) < 4:
            return 0.0
        
        return stats.kurtosis(returns)
    
    async def _calculate_tail_dependence(self, returns: np.ndarray) -> float:
        """Calculate tail dependence"""
        # Simplified tail dependence calculation
        threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) < 2:
            return 0.0
        
        # Calculate correlation in tail
        return np.corrcoef(tail_returns[:-1], tail_returns[1:])[0, 1]
    
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
    
    async def _get_portfolio_history(self) -> List[float]:
        """Get portfolio value history"""
        return [100000.0] * 252  # Placeholder
    
    async def _get_symbol_liquidity(self, symbol: str) -> float:
        """Get liquidity score for a symbol"""
        return 0.8  # Placeholder
    
    async def _get_bid_ask_spread(self, symbol: str) -> float:
        """Get bid-ask spread for a symbol"""
        return 0.01  # Placeholder
    
    async def _get_daily_volume(self, symbol: str) -> float:
        """Get daily volume for a symbol"""
        return 1000000  # Placeholder

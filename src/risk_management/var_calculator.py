"""
Value at Risk (VaR) Calculator Module
Advanced VaR calculations using multiple methodologies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import asyncio

logger = structlog.get_logger()

@dataclass
class VaRResult:
    """VaR calculation result"""
    var_value: float
    confidence_level: float
    time_horizon: int
    method: str
    expected_shortfall: float
    conditional_var: float
    historical_var: float
    parametric_var: float
    monte_carlo_var: float
    confidence_interval: Tuple[float, float]
    backtest_results: Dict

class VaRCalculator:
    """Advanced Value at Risk calculator with multiple methodologies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.returns_cache = {}
        self.var_history = []
        self.backtest_results = {}
    
    async def calculate_comprehensive_var(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "all"
    ) -> VaRResult:
        """
        Calculate comprehensive VaR using multiple methods
        
        Args:
            confidence_level: Confidence level (0.95, 0.99, etc.)
            time_horizon: Time horizon in days
            method: Calculation method ("historical", "parametric", "monte_carlo", "all")
        
        Returns:
            VaRResult with comprehensive VaR analysis
        """
        returns = await self._get_portfolio_returns()
        
        if not returns:
            logger.warning("No returns data available for VaR calculation")
            return VaRResult(
                var_value=0.0,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                method=method,
                expected_shortfall=0.0,
                conditional_var=0.0,
                historical_var=0.0,
                parametric_var=0.0,
                monte_carlo_var=0.0,
                confidence_interval=(0.0, 0.0),
                backtest_results={}
            )
        
        # Calculate VaR using different methods
        historical_var = await self._calculate_historical_var(returns, confidence_level, time_horizon)
        parametric_var = await self._calculate_parametric_var(returns, confidence_level, time_horizon)
        monte_carlo_var = await self._calculate_monte_carlo_var(returns, confidence_level, time_horizon)
        
        # Calculate Expected Shortfall (Conditional VaR)
        expected_shortfall = await self._calculate_expected_shortfall(returns, confidence_level)
        
        # Choose primary VaR value based on method
        if method == "historical":
            primary_var = historical_var
        elif method == "parametric":
            primary_var = parametric_var
        elif method == "monte_carlo":
            primary_var = monte_carlo_var
        else:  # "all" or default
            # Use average of all methods
            primary_var = np.mean([historical_var, parametric_var, monte_carlo_var])
        
        # Calculate confidence interval
        confidence_interval = await self._calculate_var_confidence_interval(returns, confidence_level)
        
        # Backtest results
        backtest_results = await self._backtest_var(returns, primary_var, confidence_level)
        
        result = VaRResult(
            var_value=primary_var,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method,
            expected_shortfall=expected_shortfall,
            conditional_var=expected_shortfall,
            historical_var=historical_var,
            parametric_var=parametric_var,
            monte_carlo_var=monte_carlo_var,
            confidence_interval=confidence_interval,
            backtest_results=backtest_results
        )
        
        # Store in history
        self.var_history.append({
            "timestamp": datetime.utcnow(),
            "var": primary_var,
            "confidence_level": confidence_level,
            "method": method
        })
        
        logger.info(
            "VaR calculated",
            var=primary_var,
            confidence_level=confidence_level,
            method=method,
            expected_shortfall=expected_shortfall
        )
        
        return result
    
    async def calculate_var_with_new_position(self, order: Dict) -> float:
        """Calculate VaR if we add this position"""
        current_positions = await self._get_current_positions()
        
        # Add the new position
        new_positions = current_positions.copy()
        new_positions[order["symbol"]] = {
            "quantity": order["quantity"],
            "price": order["price"],
            "value": order["quantity"] * order["price"]
        }
        
        # Calculate new portfolio returns
        new_returns = await self._simulate_portfolio_returns_with_position(new_positions)
        
        # Calculate VaR with new returns
        var_result = await self.calculate_comprehensive_var()
        
        return var_result.var_value
    
    async def _calculate_historical_var(
        self, 
        returns: List[float], 
        confidence_level: float, 
        time_horizon: int
    ) -> float:
        """Calculate Historical VaR"""
        if not returns:
            return 0.0
        
        # Scale returns for time horizon
        scaled_returns = [r * np.sqrt(time_horizon) for r in returns]
        
        # Calculate VaR percentile
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)
        
        return abs(var) * 100  # Return as percentage
    
    async def _calculate_parametric_var(
        self, 
        returns: List[float], 
        confidence_level: float, 
        time_horizon: int
    ) -> float:
        """Calculate Parametric VaR (assuming normal distribution)"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Scale for time horizon
        mean_return *= time_horizon
        std_return *= np.sqrt(time_horizon)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var = mean_return + z_score * std_return
        
        return abs(var) * 100  # Return as percentage
    
    async def _calculate_monte_carlo_var(
        self, 
        returns: List[float], 
        confidence_level: float, 
        time_horizon: int,
        n_simulations: int = None
    ) -> float:
        """Calculate Monte Carlo VaR"""
        if not returns:
            return 0.0
        
        if n_simulations is None:
            n_simulations = self.config.get("VAR_SIMULATIONS", 10000)
        
        # Fit distribution to historical returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random scenarios
        simulated_returns = []
        for _ in range(n_simulations):
            # Use normal distribution (could be enhanced with other distributions)
            scenario_return = np.random.normal(mean_return, std_return)
            # Scale for time horizon
            scenario_return *= np.sqrt(time_horizon)
            simulated_returns.append(scenario_return)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, var_percentile)
        
        return abs(var) * 100  # Return as percentage
    
    async def _calculate_expected_shortfall(
        self, 
        returns: List[float], 
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if not returns:
            return 0.0
        
        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate average of returns below VaR threshold
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return 0.0
        
        expected_shortfall = np.mean(tail_returns)
        
        return abs(expected_shortfall) * 100  # Return as percentage
    
    async def _calculate_var_confidence_interval(
        self, 
        returns: List[float], 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for VaR"""
        if len(returns) < 30:
            return (0.0, 0.0)
        
        # Bootstrap method for confidence interval
        n_bootstrap = 1000
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(bootstrap_sample, var_percentile)
            bootstrap_vars.append(abs(var) * 100)
        
        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_vars, 2.5)
        upper_bound = np.percentile(bootstrap_vars, 97.5)
        
        return (lower_bound, upper_bound)
    
    async def _backtest_var(
        self, 
        returns: List[float], 
        var_value: float, 
        confidence_level: float
    ) -> Dict:
        """Backtest VaR model"""
        if not returns:
            return {}
        
        # Convert VaR back to decimal
        var_threshold = -var_value / 100
        
        # Count violations (returns worse than VaR)
        violations = [r for r in returns if r <= var_threshold]
        n_violations = len(violations)
        n_observations = len(returns)
        
        # Calculate violation rate
        violation_rate = n_violations / n_observations
        expected_violation_rate = 1 - confidence_level
        
        # Kupiec test for VaR backtesting
        kupiec_stat = self._kupiec_test(n_violations, n_observations, expected_violation_rate)
        
        # Christoffersen test for independence of violations
        christoffersen_stat = self._christoffersen_test(returns, var_threshold)
        
        # Calculate average violation magnitude
        avg_violation = np.mean(violations) if violations else 0.0
        
        return {
            "n_violations": n_violations,
            "n_observations": n_observations,
            "violation_rate": violation_rate,
            "expected_violation_rate": expected_violation_rate,
            "kupiec_statistic": kupiec_stat,
            "christoffersen_statistic": christoffersen_stat,
            "avg_violation_magnitude": abs(avg_violation) * 100,
            "model_quality": "good" if abs(violation_rate - expected_violation_rate) < 0.01 else "poor"
        }
    
    def _kupiec_test(self, n_violations: int, n_observations: int, expected_rate: float) -> float:
        """Kupiec test for VaR backtesting"""
        if n_observations == 0:
            return 0.0
        
        # Likelihood ratio test
        p_hat = n_violations / n_observations
        
        if p_hat == 0 or p_hat == 1:
            return 0.0
        
        # Calculate test statistic
        lr = -2 * (
            n_violations * np.log(expected_rate / p_hat) +
            (n_observations - n_violations) * np.log((1 - expected_rate) / (1 - p_hat))
        )
        
        return lr
    
    def _christoffersen_test(self, returns: List[float], var_threshold: float) -> float:
        """Christoffersen test for independence of violations"""
        if len(returns) < 2:
            return 0.0
        
        # Create violation sequence
        violations = [1 if r <= var_threshold else 0 for r in returns]
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        
        for i in range(len(violations) - 1):
            if violations[i] == 0 and violations[i+1] == 0:
                n00 += 1
            elif violations[i] == 0 and violations[i+1] == 1:
                n01 += 1
            elif violations[i] == 1 and violations[i+1] == 0:
                n10 += 1
            elif violations[i] == 1 and violations[i+1] == 1:
                n11 += 1
        
        # Calculate probabilities
        if n00 + n01 > 0:
            p01 = n01 / (n00 + n01)
        else:
            p01 = 0
        
        if n10 + n11 > 0:
            p11 = n11 / (n10 + n11)
        else:
            p11 = 0
        
        # Calculate test statistic
        if p01 == p11:
            return 0.0
        
        # Likelihood ratio test
        p = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        lr = -2 * (
            n01 * np.log(p) + n00 * np.log(1 - p) + n11 * np.log(p) + n10 * np.log(1 - p) -
            n01 * np.log(p01) - n00 * np.log(1 - p01) - n11 * np.log(p11) - n10 * np.log(1 - p11)
        )
        
        return lr
    
    async def calculate_incremental_var(self, position: Dict) -> float:
        """Calculate incremental VaR for a position"""
        current_var = await self.calculate_comprehensive_var()
        
        # Calculate VaR with position
        new_var = await self.calculate_var_with_new_position(position)
        
        # Incremental VaR
        incremental_var = new_var - current_var.var_value
        
        return incremental_var
    
    async def calculate_marginal_var(self, position: Dict) -> float:
        """Calculate marginal VaR for a position"""
        positions = await self._get_current_positions()
        
        # Calculate portfolio weights
        total_value = sum(pos["value"] for pos in positions.values())
        position_weight = position["value"] / total_value
        
        # Get correlation with portfolio
        correlation = await self._get_position_correlation(position["symbol"])
        
        # Marginal VaR calculation
        portfolio_var = await self.calculate_comprehensive_var()
        marginal_var = portfolio_var.var_value * position_weight * correlation
        
        return marginal_var
    
    async def calculate_component_var(self, position: Dict) -> float:
        """Calculate component VaR for a position"""
        marginal_var = await self.calculate_marginal_var(position)
        positions = await self._get_current_positions()
        
        # Calculate position weight
        total_value = sum(pos["value"] for pos in positions.values())
        position_weight = position["value"] / total_value
        
        # Component VaR
        component_var = marginal_var * position_weight
        
        return component_var
    
    async def stress_test_var(self, stress_scenarios: Dict[str, Dict]) -> Dict[str, float]:
        """Stress test VaR under various scenarios"""
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress scenario to returns
            stressed_returns = await self._apply_stress_scenario(scenario_params)
            
            # Calculate VaR under stress
            var_result = await self._calculate_historical_var(
                stressed_returns, 
                self.config.get("VAR_CONFIDENCE_LEVEL", 0.95), 
                1
            )
            
            results[scenario_name] = var_result
        
        return results
    
    async def _apply_stress_scenario(self, scenario_params: Dict) -> List[float]:
        """Apply stress scenario to returns"""
        base_returns = await self._get_portfolio_returns()
        
        # Apply stress factors
        stressed_returns = []
        for ret in base_returns:
            # Apply market shock
            market_shock = scenario_params.get("market_shock", 0)
            stressed_ret = ret * (1 + market_shock / 100)
            
            # Apply volatility shock
            vol_shock = scenario_params.get("volatility_shock", 0)
            if vol_shock > 0:
                stressed_ret *= (1 + vol_shock / 100)
            
            stressed_returns.append(stressed_ret)
        
        return stressed_returns
    
    async def _simulate_portfolio_returns_with_position(self, positions: Dict) -> List[float]:
        """Simulate portfolio returns with new position"""
        # This would simulate returns including the new position
        # For now, return placeholder
        return [0.001] * 252
    
    # Data access methods (placeholders)
    
    async def _get_portfolio_returns(self) -> List[float]:
        """Get historical portfolio returns"""
        return [0.001] * 252  # Placeholder
    
    async def _get_current_positions(self) -> Dict:
        """Get current portfolio positions"""
        return {}  # Placeholder
    
    async def _get_position_correlation(self, symbol: str) -> float:
        """Get correlation of position with portfolio"""
        return 0.5  # Placeholder

"""
Stress Testing Module
Comprehensive stress testing for portfolio risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
import asyncio

logger = structlog.get_logger()

@dataclass
class StressTestResult:
    """Stress test result"""
    scenario_name: str
    portfolio_loss: float
    loss_percentage: float
    worst_position: str
    worst_position_loss: float
    correlation_impact: float
    liquidity_impact: float
    recovery_time_estimate: int
    confidence_level: float
    
    def __lt__(self, other):
        """Less than comparison based on portfolio loss"""
        return self.portfolio_loss < other.portfolio_loss
    
    def __le__(self, other):
        """Less than or equal comparison based on portfolio loss"""
        return self.portfolio_loss <= other.portfolio_loss
    
    def __gt__(self, other):
        """Greater than comparison based on portfolio loss"""
        return self.portfolio_loss > other.portfolio_loss
    
    def __ge__(self, other):
        """Greater than or equal comparison based on portfolio loss"""
        return self.portfolio_loss >= other.portfolio_loss

class StressTester:
    """Advanced stress testing system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scenarios = self._load_stress_scenarios()
        self.historical_crises = self._load_historical_crises()
        self.custom_scenarios = {}
    
    def _load_stress_scenarios(self) -> Dict:
        """Load predefined stress test scenarios"""
        return {
            "market_crash_10": {
                "name": "10% Market Crash",
                "market_shock": -10,
                "volatility_spike": 200,
                "correlation_increase": 0.3,
                "liquidity_drop": 0.5
            },
            "market_crash_20": {
                "name": "20% Market Crash", 
                "market_shock": -20,
                "volatility_spike": 300,
                "correlation_increase": 0.5,
                "liquidity_drop": 0.7
            },
            "market_crash_30": {
                "name": "30% Market Crash",
                "market_shock": -30,
                "volatility_spike": 400,
                "correlation_increase": 0.7,
                "liquidity_drop": 0.8
            },
            "flash_crash": {
                "name": "Flash Crash",
                "market_shock": -10,
                "volatility_spike": 500,
                "correlation_increase": 0.9,
                "liquidity_drop": 0.9,
                "duration": 1  # 1 day
            },
            "interest_rate_shock": {
                "name": "Interest Rate Shock",
                "rate_increase": 2.0,  # 2% increase
                "bond_shock": -15,
                "equity_impact": -5,
                "volatility_spike": 150
            },
            "currency_crisis": {
                "name": "Currency Crisis",
                "usd_strength": 20,
                "emerging_market_shock": -25,
                "commodity_shock": -15,
                "volatility_spike": 250
            },
            "sector_rotation": {
                "name": "Sector Rotation",
                "tech_shock": -20,
                "value_boost": 15,
                "growth_shock": -25,
                "volatility_spike": 180
            },
            "liquidity_crisis": {
                "name": "Liquidity Crisis",
                "liquidity_drop": 0.8,
                "spread_widening": 300,
                "volatility_spike": 200,
                "correlation_increase": 0.6
            }
        }
    
    def _load_historical_crises(self) -> Dict:
        """Load historical crisis scenarios"""
        return {
            "2008_financial_crisis": {
                "name": "2008 Financial Crisis",
                "market_shock": -50,
                "volatility_spike": 300,
                "correlation_increase": 0.95,
                "liquidity_drop": 0.8,
                "duration": 18,  # months
                "recovery_time": 60  # months
            },
            "covid_19_crash": {
                "name": "COVID-19 Market Crash",
                "market_shock": -35,
                "volatility_spike": 400,
                "correlation_increase": 0.90,
                "liquidity_drop": 0.7,
                "duration": 1,  # month
                "recovery_time": 6  # months
            },
            "dot_com_bubble": {
                "name": "Dot-com Bubble Burst",
                "market_shock": -45,
                "volatility_spike": 250,
                "correlation_increase": 0.85,
                "liquidity_drop": 0.6,
                "duration": 24,  # months
                "recovery_time": 48  # months
            },
            "black_monday": {
                "name": "Black Monday 1987",
                "market_shock": -22,
                "volatility_spike": 500,
                "correlation_increase": 0.98,
                "liquidity_drop": 0.9,
                "duration": 1,  # day
                "recovery_time": 2  # years
            },
            "asian_financial_crisis": {
                "name": "Asian Financial Crisis",
                "emerging_market_shock": -40,
                "currency_shock": -30,
                "volatility_spike": 200,
                "correlation_increase": 0.7,
                "liquidity_drop": 0.6,
                "duration": 12,  # months
                "recovery_time": 24  # months
            }
        }
    
    async def run_comprehensive_stress_test(self) -> Dict[str, StressTestResult]:
        """Run comprehensive stress test across all scenarios"""
        results = {}
        
        # Run predefined scenarios
        for scenario_id, scenario_params in self.scenarios.items():
            result = await self._run_scenario(scenario_id, scenario_params)
            results[scenario_id] = result
        
        # Run historical crisis scenarios
        for crisis_id, crisis_params in self.historical_crises.items():
            result = await self._run_scenario(crisis_id, crisis_params)
            results[crisis_id] = result
        
        # Run custom scenarios
        for custom_id, custom_params in self.custom_scenarios.items():
            result = await self._run_scenario(custom_id, custom_params)
            results[custom_id] = result
        
        logger.info("Comprehensive stress test completed", n_scenarios=len(results))
        
        return results
    
    async def run_all_scenarios(self) -> Dict[str, float]:
        """Run all stress test scenarios and return loss percentages"""
        results = await self.run_comprehensive_stress_test()
        
        # Return simplified results
        simplified_results = {}
        for scenario_id, result in results.items():
            simplified_results[scenario_id] = result.loss_percentage
        
        return simplified_results
    
    async def quick_test(self, order: Dict, scenarios: List[str]) -> Dict:
        """Quick stress test for pre-trade checks"""
        results = {}
        position_value = order["quantity"] * order["price"]
        portfolio_value = await self._get_portfolio_value()
        
        for scenario in scenarios:
            if scenario == "10%_drop":
                loss = (position_value * 0.10) / portfolio_value * 100
                results[scenario] = loss
            elif scenario == "20%_drop":
                loss = (position_value * 0.20) / portfolio_value * 100
                results[scenario] = loss
            elif scenario == "black_swan":
                loss = (position_value * 0.50) / portfolio_value * 100
                results[scenario] = loss
            else:
                # Use predefined scenario
                if scenario in self.scenarios:
                    scenario_params = self.scenarios[scenario]
                    loss = await self._calculate_scenario_loss(order, scenario_params)
                    results[scenario] = loss
        
        return results
    
    async def _run_scenario(self, scenario_id: str, scenario_params: Dict) -> StressTestResult:
        """Run a specific stress test scenario"""
        positions = await self._get_current_positions()
        portfolio_value = await self._get_portfolio_value()
        
        # Calculate portfolio loss under scenario
        total_loss = 0.0
        worst_position = ""
        worst_position_loss = 0.0
        
        for symbol, position in positions.items():
            position_loss = await self._calculate_position_loss(position, scenario_params)
            total_loss += position_loss
            
            if position_loss > worst_position_loss:
                worst_position_loss = position_loss
                worst_position = symbol
        
        # Calculate additional impacts
        correlation_impact = await self._calculate_correlation_impact(scenario_params)
        liquidity_impact = await self._calculate_liquidity_impact(scenario_params)
        
        # Estimate recovery time
        recovery_time = await self._estimate_recovery_time(scenario_params)
        
        # Calculate confidence level
        confidence_level = await self._calculate_scenario_confidence(scenario_params)
        
        result = StressTestResult(
            scenario_name=scenario_params.get("name", scenario_id),
            portfolio_loss=total_loss,
            loss_percentage=(total_loss / portfolio_value) * 100,
            worst_position=worst_position,
            worst_position_loss=worst_position_loss,
            correlation_impact=correlation_impact,
            liquidity_impact=liquidity_impact,
            recovery_time_estimate=recovery_time,
            confidence_level=confidence_level
        )
        
        logger.info(
            "Stress test scenario completed",
            scenario=scenario_id,
            loss_percentage=result.loss_percentage,
            worst_position=worst_position
        )
        
        return result
    
    async def _calculate_position_loss(self, position: Dict, scenario_params: Dict) -> float:
        """Calculate loss for a specific position under scenario"""
        position_value = position["value"]
        
        # Base market shock
        market_shock = scenario_params.get("market_shock", 0) / 100
        
        # Symbol-specific shocks
        symbol = position["symbol"]
        symbol_shock = scenario_params.get(f"{symbol}_shock", 0) / 100
        
        # Sector-specific shocks
        sector = await self._get_symbol_sector(symbol)
        sector_shock = scenario_params.get(f"{sector}_shock", 0) / 100
        
        # Apply the largest shock
        total_shock = max(abs(market_shock), abs(symbol_shock), abs(sector_shock))
        
        # Calculate loss
        loss = position_value * total_shock
        
        return loss
    
    async def _calculate_scenario_loss(self, order: Dict, scenario_params: Dict) -> float:
        """Calculate loss for a new order under scenario"""
        position_value = order["quantity"] * order["price"]
        portfolio_value = await self._get_portfolio_value()
        
        # Apply scenario shock
        market_shock = scenario_params.get("market_shock", 0) / 100
        symbol_shock = scenario_params.get(f"{order['symbol']}_shock", 0) / 100
        
        total_shock = max(abs(market_shock), abs(symbol_shock))
        loss = position_value * total_shock
        
        return (loss / portfolio_value) * 100
    
    async def _calculate_correlation_impact(self, scenario_params: Dict) -> float:
        """Calculate correlation impact under scenario"""
        correlation_increase = scenario_params.get("correlation_increase", 0)
        
        # Higher correlation means higher portfolio risk
        positions = await self._get_current_positions()
        n_positions = len(positions)
        
        if n_positions <= 1:
            return 0.0
        
        # Diversification benefit reduction
        diversification_benefit = 1 / np.sqrt(n_positions)
        correlation_impact = correlation_increase * (1 - diversification_benefit)
        
        return correlation_impact * 100
    
    async def _calculate_liquidity_impact(self, scenario_params: Dict) -> float:
        """Calculate liquidity impact under scenario"""
        liquidity_drop = scenario_params.get("liquidity_drop", 0)
        
        # Liquidity impact affects ability to exit positions
        positions = await self._get_current_positions()
        total_value = sum(pos["value"] for pos in positions.values())
        
        # Estimate liquidity cost
        liquidity_cost = total_value * liquidity_drop * 0.01  # 1% cost per 10% liquidity drop
        
        return liquidity_cost
    
    async def _estimate_recovery_time(self, scenario_params: Dict) -> int:
        """Estimate recovery time in days"""
        # Use historical data to estimate recovery
        market_shock = abs(scenario_params.get("market_shock", 0))
        volatility_spike = scenario_params.get("volatility_spike", 100)
        
        # Simple model: larger shocks take longer to recover
        base_recovery = 30  # 30 days base
        shock_factor = market_shock / 10  # 10 days per 10% shock
        volatility_factor = volatility_spike / 100  # Additional time for volatility
        
        recovery_time = base_recovery + (shock_factor * 10) + (volatility_factor * 5)
        
        return int(recovery_time)
    
    async def _calculate_scenario_confidence(self, scenario_params: Dict) -> float:
        """Calculate confidence level for scenario"""
        # Based on historical frequency and severity
        market_shock = abs(scenario_params.get("market_shock", 0))
        
        if market_shock <= 10:
            return 0.8  # 80% confidence - common
        elif market_shock <= 20:
            return 0.6  # 60% confidence - moderate
        elif market_shock <= 30:
            return 0.4  # 40% confidence - severe
        else:
            return 0.2  # 20% confidence - extreme
    
    async def create_custom_scenario(
        self, 
        scenario_id: str, 
        scenario_params: Dict
    ) -> bool:
        """Create a custom stress test scenario"""
        try:
            self.custom_scenarios[scenario_id] = scenario_params
            logger.info("Custom stress test scenario created", scenario_id=scenario_id)
            return True
        except Exception as e:
            logger.error("Failed to create custom scenario", error=str(e))
            return False
    
    async def run_sensitivity_analysis(self, parameter: str, values: List[float]) -> Dict:
        """Run sensitivity analysis for a parameter"""
        results = {}
        
        for value in values:
            # Create scenario with modified parameter
            base_scenario = self.scenarios["market_crash_20"].copy()
            base_scenario[parameter] = value
            
            # Run scenario
            result = await self._run_scenario(f"sensitivity_{parameter}_{value}", base_scenario)
            results[value] = result.loss_percentage
        
        return results
    
    async def run_monte_carlo_stress_test(
        self, 
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """Run Monte Carlo stress test"""
        results = []
        
        for _ in range(n_simulations):
            # Generate random scenario parameters
            market_shock = np.random.normal(-15, 10)  # Mean -15%, std 10%
            volatility_spike = np.random.exponential(200)  # Exponential distribution
            correlation_increase = np.random.beta(2, 5)  # Beta distribution
            
            scenario_params = {
                "market_shock": market_shock,
                "volatility_spike": volatility_spike,
                "correlation_increase": correlation_increase,
                "liquidity_drop": 0.5
            }
            
            # Run scenario
            result = await self._run_scenario("monte_carlo", scenario_params)
            results.append(result.loss_percentage)
        
        # Calculate statistics
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(results, var_percentile)
        expected_shortfall = np.mean([r for r in results if r >= var])
        
        return {
            "var": var,
            "expected_shortfall": expected_shortfall,
            "mean_loss": np.mean(results),
            "std_loss": np.std(results),
            "max_loss": np.max(results),
            "min_loss": np.min(results),
            "confidence_level": confidence_level,
            "n_simulations": n_simulations
        }
    
    async def generate_stress_test_report(self) -> Dict:
        """Generate comprehensive stress test report"""
        results = await self.run_comprehensive_stress_test()
        
        # Calculate summary statistics
        losses = [result.loss_percentage for result in results.values()]
        
        report = {
            "summary": {
                "total_scenarios": len(results),
                "max_loss": max(losses),
                "min_loss": min(losses),
                "mean_loss": np.mean(losses),
                "median_loss": np.median(losses),
                "std_loss": np.std(losses)
            },
            "worst_case_scenarios": sorted(
                [(name, result.loss_percentage) for name, result in results.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "scenario_results": {
                name: {
                    "loss_percentage": result.loss_percentage,
                    "worst_position": result.worst_position,
                    "recovery_time": result.recovery_time_estimate,
                    "confidence_level": result.confidence_level
                }
                for name, result in results.items()
            },
            "recommendations": await self._generate_stress_test_recommendations(results),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
    
    async def _generate_stress_test_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        max_loss = max(result.loss_percentage for result in results.values())
        
        if max_loss > 50:
            recommendations.append("CRITICAL: Maximum loss exceeds 50% - immediate risk reduction required")
        elif max_loss > 30:
            recommendations.append("HIGH RISK: Maximum loss exceeds 30% - consider reducing position sizes")
        elif max_loss > 20:
            recommendations.append("MODERATE RISK: Maximum loss exceeds 20% - monitor positions closely")
        else:
            recommendations.append("Risk levels appear acceptable based on stress testing")
        
        # Check for concentration risk
        worst_positions = [result.worst_position for result in results.values()]
        if worst_positions:
            most_common_worst = max(set(worst_positions), key=worst_positions.count)
            if worst_positions.count(most_common_worst) > len(results) * 0.3:
                recommendations.append(f"Consider reducing exposure to {most_common_worst} - appears in worst case for multiple scenarios")
        
        return recommendations
    
    # Helper methods (placeholders)
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return 100000.0  # Placeholder
    
    async def _get_current_positions(self) -> Dict:
        """Get current portfolio positions"""
        return {}  # Placeholder
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        return "technology"  # Placeholder

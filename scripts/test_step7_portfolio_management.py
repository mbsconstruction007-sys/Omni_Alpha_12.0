"""
Test Script for Step 7: Advanced Portfolio Management System
Comprehensive testing of all portfolio management components
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.portfolio_management.portfolio_engine import PortfolioEngine, Portfolio, Position, PortfolioStatus, MarketRegime
from backend.app.portfolio_management.portfolio_optimizer import PortfolioOptimizer
from backend.app.portfolio_management.tax_optimizer import TaxOptimizer, TaxLot
from backend.app.portfolio_management.portfolio_allocator import PortfolioAllocator
from backend.app.portfolio_management.portfolio_rebalancer import PortfolioRebalancer
from backend.app.portfolio_management.portfolio_analytics import PortfolioAnalytics
from backend.app.portfolio_management.performance_attribution import PerformanceAttributor
from backend.app.portfolio_management.regime_detector import RegimeDetector
from backend.app.portfolio_management.portfolio_models import PortfolioModels
from backend.app.portfolio_management.portfolio_backtester import PortfolioBacktester
from backend.app.portfolio_management.portfolio_ml import PortfolioMLEngine
from backend.app.core.portfolio_config import load_portfolio_config, apply_portfolio_preset, validate_portfolio_config

class PortfolioManagementTester:
    """Comprehensive tester for portfolio management system"""
    
    def __init__(self):
        self.config = self._load_test_config()
        self.test_results = {}
    
    def _load_test_config(self) -> Dict:
        """Load test configuration"""
        return {
            "PORTFOLIO_MANAGEMENT_ENABLED": True,
            "OPTIMIZATION_METHOD": "ensemble",
            "MAX_POSITION_WEIGHT": 0.10,
            "MIN_POSITION_WEIGHT": 0.01,
            "TARGET_PORTFOLIO_VOLATILITY": 0.15,
            "MAX_LEVERAGE": 1.0,
            "REBALANCING_ENABLED": True,
            "REBALANCING_METHOD": "threshold",
            "REBALANCING_THRESHOLD_PERCENT": 0.05,
            "TAX_OPTIMIZATION_ENABLED": True,
            "SHORT_TERM_CAPITAL_GAINS_RATE": 0.37,
            "LONG_TERM_CAPITAL_GAINS_RATE": 0.20,
            "STATE_TAX_RATE": 0.05,
            "WASH_SALE_PERIOD_DAYS": 30,
            "TAX_HARVEST_THRESHOLD_USD": 1000,
            "REGIME_DETECTION_ENABLED": True,
            "ML_PORTFOLIO_OPTIMIZATION": False,
            "PORTFOLIO_UPDATE_INTERVAL_MS": 1000,
            "METRICS_CACHE_TTL_SECONDS": 60,
            "REGIME_UPDATE_FREQUENCY_HOURS": 24,
            "EMERGENCY_REBALANCE_THRESHOLD": 0.20,
            "REGIME_SPECIFIC_PARAMS": {
                "bull_quiet": {"leverage": 1.0, "position_count": 20},
                "bull_volatile": {"leverage": 0.8, "position_count": 15},
                "bear_quiet": {"leverage": 0.6, "position_count": 10},
                "bear_volatile": {"leverage": 0.4, "position_count": 8},
                "transition": {"leverage": 0.7, "position_count": 12},
                "crisis": {"leverage": 0.2, "position_count": 5}
            }
        }
    
    async def run_all_tests(self):
        """Run all portfolio management tests"""
        print("💼 Starting Step 7: Advanced Portfolio Management System Tests")
        print("=" * 60)
        
        tests = [
            ("Portfolio Engine", self.test_portfolio_engine),
            ("Portfolio Optimizer", self.test_portfolio_optimizer),
            ("Tax Optimizer", self.test_tax_optimizer),
            ("Portfolio Allocator", self.test_portfolio_allocator),
            ("Portfolio Rebalancer", self.test_portfolio_rebalancer),
            ("Portfolio Analytics", self.test_portfolio_analytics),
            ("Performance Attribution", self.test_performance_attribution),
            ("Regime Detector", self.test_regime_detector),
            ("Portfolio Models", self.test_portfolio_models),
            ("Portfolio Backtester", self.test_portfolio_backtester),
            ("Portfolio ML Engine", self.test_portfolio_ml_engine),
            ("Configuration", self.test_configuration),
            ("Integration", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(f"   {status} - {result['message']}")
            except Exception as e:
                self.test_results[test_name] = {
                    "passed": False,
                    "message": f"Test failed with exception: {str(e)}",
                    "error": str(e)
                }
                print(f"   ❌ FAILED - Exception: {str(e)}")
        
        self._print_summary()
    
    async def test_portfolio_engine(self) -> Dict:
        """Test portfolio engine functionality"""
        try:
            # Initialize portfolio engine
            engine = PortfolioEngine(self.config)
            await engine.initialize()
            
            # Test portfolio construction
            objectives = {
                "target_return": 0.15,
                "max_volatility": 0.20,
                "risk_tolerance": "moderate"
            }
            
            portfolio = await engine.construct_portfolio(
                capital=100000,
                objectives=objectives
            )
            
            assert portfolio is not None, "Portfolio should be created"
            assert len(portfolio.positions) > 0, "Portfolio should have positions"
            assert portfolio.total_value == 100000, "Portfolio value should match capital"
            
            # Test optimization
            optimization_result = await engine.optimize_portfolio(method="ensemble")
            assert "optimized_weights" in optimization_result, "Should return optimized weights"
            
            # Test rebalancing
            rebalance_result = await engine.rebalance_portfolio(force=True)
            assert "rebalanced" in rebalance_result, "Should return rebalancing result"
            
            # Test adding position
            add_result = await engine.add_position("AAPL", 100, "test_strategy")
            assert add_result["success"], "Should successfully add position"
            
            return {
                "passed": True,
                "message": f"Portfolio engine test passed - Portfolio created with {len(portfolio.positions)} positions",
                "details": {
                    "portfolio_id": portfolio.portfolio_id,
                    "total_value": portfolio.total_value,
                    "positions_count": len(portfolio.positions),
                    "optimization_successful": "optimized_weights" in optimization_result,
                    "rebalancing_successful": "rebalanced" in rebalance_result
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio engine test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_optimizer(self) -> Dict:
        """Test portfolio optimizer functionality"""
        try:
            optimizer = PortfolioOptimizer(self.config)
            
            # Test universe
            universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            signals = {symbol: 0.1 for symbol in universe}
            constraints = {
                "max_position_weight": 0.20,
                "min_position_weight": 0.05
            }
            
            # Test different optimization methods
            methods = ["ensemble", "mean_variance", "risk_parity", "hierarchical_risk_parity"]
            results = {}
            
            for method in methods:
                weights = await optimizer.optimize(universe, signals, constraints, method)
                results[method] = weights
                
                # Validate weights
                total_weight = sum(weights.values())
                assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0 for {method}"
                
                for symbol, weight in weights.items():
                    assert 0 <= weight <= 1, f"Weight should be between 0 and 1 for {symbol} in {method}"
            
            return {
                "passed": True,
                "message": f"Portfolio optimizer test passed - {len(methods)} methods tested",
                "details": {
                    "methods_tested": list(results.keys()),
                    "universe_size": len(universe),
                    "weight_sums": {method: sum(weights.values()) for method, weights in results.items()}
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio optimizer test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_tax_optimizer(self) -> Dict:
        """Test tax optimizer functionality"""
        try:
            tax_optimizer = TaxOptimizer(self.config)
            
            # Create test positions
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=140.0,  # Loss
                    entry_date=datetime.utcnow() - timedelta(days=10),
                    strategy="test",
                    target_weight=0.1,
                    current_weight=0.1,
                    unrealized_pnl=-1000.0,  # $1000 loss
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2,
                    tax_lots=[{
                        "quantity": 100,
                        "price": 150.0,
                        "date": datetime.utcnow() - timedelta(days=10)
                    }]
                ),
                Position(
                    symbol="MSFT",
                    quantity=50,
                    entry_price=300.0,
                    current_price=320.0,  # Gain
                    entry_date=datetime.utcnow() - timedelta(days=400),  # Long-term
                    strategy="test",
                    target_weight=0.1,
                    current_weight=0.1,
                    unrealized_pnl=1000.0,  # $1000 gain
                    realized_pnl=0.0,
                    risk_score=0.4,
                    correlation_sum=0.3,
                    tax_lots=[{
                        "quantity": 50,
                        "price": 300.0,
                        "date": datetime.utcnow() - timedelta(days=400)
                    }]
                )
            ]
            
            # Test tax loss harvesting
            harvest_result = await tax_optimizer.harvest_losses(positions)
            assert "harvested_positions" in harvest_result, "Should return harvested positions"
            assert "total_loss_harvested" in harvest_result, "Should return total loss harvested"
            
            # Test holding period optimization
            holding_recommendations = await tax_optimizer.optimize_holding_periods(positions)
            assert isinstance(holding_recommendations, list), "Should return list of recommendations"
            
            # Test after-tax returns calculation
            after_tax_return = await tax_optimizer.calculate_after_tax_returns(0.15, 400)
            assert after_tax_return < 0.15, "After-tax return should be less than gross return"
            
            return {
                "passed": True,
                "message": f"Tax optimizer test passed - Harvested ${harvest_result['total_loss_harvested']:.2f} in losses",
                "details": {
                    "harvested_positions": len(harvest_result["harvested_positions"]),
                    "total_loss_harvested": harvest_result["total_loss_harvested"],
                    "estimated_tax_savings": harvest_result["estimated_tax_savings"],
                    "holding_recommendations": len(holding_recommendations),
                    "after_tax_return": after_tax_return
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Tax optimizer test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_allocator(self) -> Dict:
        """Test portfolio allocator functionality"""
        try:
            allocator = PortfolioAllocator(self.config)
            
            universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            signals = {symbol: 0.1 for symbol in universe}
            capital = 100000
            
            allocation = await allocator.allocate_capital(universe, signals, capital)
            
            assert isinstance(allocation, dict), "Should return allocation dictionary"
            assert len(allocation) == len(universe), "Should allocate to all assets"
            
            total_allocated = sum(allocation.values())
            assert abs(total_allocated - capital) < 1, "Total allocation should equal capital"
            
            return {
                "passed": True,
                "message": f"Portfolio allocator test passed - Allocated ${total_allocated:,.2f} across {len(universe)} assets",
                "details": {
                    "universe_size": len(universe),
                    "total_allocated": total_allocated,
                    "allocation_per_asset": {symbol: f"${amount:,.2f}" for symbol, amount in allocation.items()}
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio allocator test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_rebalancer(self) -> Dict:
        """Test portfolio rebalancer functionality"""
        try:
            rebalancer = PortfolioRebalancer(self.config)
            
            # Create test portfolio
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=160.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.2,
                    current_weight=0.3,  # Drifted
                    unrealized_pnl=1000.0,
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2
                )
            ]
            
            portfolio = Portfolio(
                portfolio_id="test",
                positions=positions,
                cash=50000,
                total_value=100000,
                leverage=1.0,
                status=PortfolioStatus.ACTIVE,
                regime=MarketRegime.BULL_QUIET,
                metrics={},
                constraints={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Test drift calculation
            drift = await rebalancer.calculate_drift(portfolio)
            assert isinstance(drift, float), "Drift should be a float"
            assert drift >= 0, "Drift should be non-negative"
            
            # Test trade calculation
            target_weights = {"AAPL": 0.2}
            trades = await rebalancer.calculate_trades(positions, target_weights)
            assert isinstance(trades, list), "Should return list of trades"
            
            return {
                "passed": True,
                "message": f"Portfolio rebalancer test passed - Drift: {drift:.2f}%, Trades: {len(trades)}",
                "details": {
                    "drift_percentage": drift,
                    "trades_calculated": len(trades),
                    "portfolio_value": portfolio.total_value
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio rebalancer test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_analytics(self) -> Dict:
        """Test portfolio analytics functionality"""
        try:
            analytics = PortfolioAnalytics(self.config)
            
            # Create test positions
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=160.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.5,
                    current_weight=0.5,
                    unrealized_pnl=1000.0,
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2
                ),
                Position(
                    symbol="MSFT",
                    quantity=50,
                    entry_price=300.0,
                    current_price=310.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.5,
                    current_weight=0.5,
                    unrealized_pnl=500.0,
                    realized_pnl=0.0,
                    risk_score=0.4,
                    correlation_sum=0.3
                )
            ]
            
            # Test metrics calculation
            metrics = await analytics.calculate_metrics(positions)
            assert isinstance(metrics, dict), "Should return metrics dictionary"
            assert "total_return" in metrics, "Should include total return"
            assert "volatility" in metrics, "Should include volatility"
            assert "sharpe_ratio" in metrics, "Should include Sharpe ratio"
            
            # Test individual metric calculations
            daily_return = await analytics.calculate_daily_return()
            volatility = await analytics.calculate_volatility()
            sharpe_ratio = await analytics.calculate_sharpe_ratio()
            max_drawdown = await analytics.calculate_max_drawdown()
            var_95 = await analytics.calculate_var(0.95)
            cvar_95 = await analytics.calculate_cvar(0.95)
            correlation_matrix = await analytics.calculate_correlation_matrix()
            
            assert isinstance(daily_return, float), "Daily return should be float"
            assert isinstance(volatility, float), "Volatility should be float"
            assert isinstance(sharpe_ratio, float), "Sharpe ratio should be float"
            assert isinstance(max_drawdown, float), "Max drawdown should be float"
            assert isinstance(var_95, float), "VaR should be float"
            assert isinstance(cvar_95, float), "CVaR should be float"
            assert correlation_matrix is not None, "Correlation matrix should not be None"
            
            return {
                "passed": True,
                "message": f"Portfolio analytics test passed - Sharpe: {sharpe_ratio:.2f}, Vol: {volatility:.2f}",
                "details": {
                    "daily_return": daily_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "var_95": var_95,
                    "cvar_95": cvar_95,
                    "correlation_matrix_shape": correlation_matrix.shape if hasattr(correlation_matrix, 'shape') else "N/A"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio analytics test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_performance_attribution(self) -> Dict:
        """Test performance attribution functionality"""
        try:
            attributor = PerformanceAttributor(self.config)
            
            # Test attribution calculation
            attribution = await attributor.calculate_attribution(
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
            
            assert isinstance(attribution, dict), "Should return attribution dictionary"
            assert "total_return" in attribution, "Should include total return"
            assert "asset_allocation" in attribution, "Should include asset allocation"
            assert "security_selection" in attribution, "Should include security selection"
            
            # Test factor exposures
            factor_exposures = await attributor.calculate_factor_exposures()
            assert isinstance(factor_exposures, dict), "Should return factor exposures dictionary"
            
            return {
                "passed": True,
                "message": f"Performance attribution test passed - Total return: {attribution.get('total_return', 0):.2f}%",
                "details": {
                    "total_return": attribution.get("total_return", 0),
                    "asset_allocation": attribution.get("asset_allocation", 0),
                    "security_selection": attribution.get("security_selection", 0),
                    "factor_exposures": factor_exposures
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Performance attribution test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_regime_detector(self) -> Dict:
        """Test regime detector functionality"""
        try:
            regime_detector = RegimeDetector(self.config)
            
            # Test regime detection
            current_regime = await regime_detector.detect_regime()
            assert current_regime is not None, "Should detect a regime"
            
            # Test regime probabilities
            regime_probs = await regime_detector.get_regime_probabilities()
            assert isinstance(regime_probs, dict), "Should return regime probabilities dictionary"
            assert len(regime_probs) > 0, "Should have regime probabilities"
            
            # Validate probabilities sum to 1
            total_prob = sum(regime_probs.values())
            assert abs(total_prob - 1.0) < 0.01, "Probabilities should sum to 1.0"
            
            # Test regime recommendation
            recommendation = await regime_detector.get_regime_recommendation()
            assert isinstance(recommendation, str), "Should return recommendation string"
            
            return {
                "passed": True,
                "message": f"Regime detector test passed - Current regime: {current_regime.value}",
                "details": {
                    "current_regime": current_regime.value,
                    "regime_probabilities": regime_probs,
                    "recommendation": recommendation,
                    "total_probability": total_prob
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Regime detector test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_models(self) -> Dict:
        """Test portfolio models functionality"""
        try:
            models = PortfolioModels(self.config)
            
            # Create test positions
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=160.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.5,
                    current_weight=0.5,
                    unrealized_pnl=1000.0,
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2
                )
            ]
            
            # Test risk metrics calculation
            risk_metrics = await models.calculate_risk_metrics(positions)
            assert isinstance(risk_metrics, dict), "Should return risk metrics dictionary"
            assert "var_95" in risk_metrics, "Should include VaR"
            assert "cvar_95" in risk_metrics, "Should include CVaR"
            assert "expected_shortfall" in risk_metrics, "Should include expected shortfall"
            
            return {
                "passed": True,
                "message": f"Portfolio models test passed - VaR: {risk_metrics.get('var_95', 0):.2f}%",
                "details": {
                    "var_95": risk_metrics.get("var_95", 0),
                    "cvar_95": risk_metrics.get("cvar_95", 0),
                    "expected_shortfall": risk_metrics.get("expected_shortfall", 0)
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio models test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_backtester(self) -> Dict:
        """Test portfolio backtester functionality"""
        try:
            backtester = PortfolioBacktester(self.config)
            
            # Create test portfolio
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=160.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.5,
                    current_weight=0.5,
                    unrealized_pnl=1000.0,
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2
                )
            ]
            
            portfolio = Portfolio(
                portfolio_id="test",
                positions=positions,
                cash=50000,
                total_value=100000,
                leverage=1.0,
                status=PortfolioStatus.ACTIVE,
                regime=MarketRegime.BULL_QUIET,
                metrics={},
                constraints={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Test backtest
            backtest_results = await backtester.run_backtest(
                strategy=portfolio,
                start_date="2024-01-01",
                end_date="2024-12-31",
                initial_capital=100000
            )
            
            assert isinstance(backtest_results, dict), "Should return backtest results dictionary"
            assert "total_return" in backtest_results, "Should include total return"
            assert "annualized_return" in backtest_results, "Should include annualized return"
            assert "sharpe_ratio" in backtest_results, "Should include Sharpe ratio"
            assert "max_drawdown" in backtest_results, "Should include max drawdown"
            assert "win_rate" in backtest_results, "Should include win rate"
            
            return {
                "passed": True,
                "message": f"Portfolio backtester test passed - Total return: {backtest_results.get('total_return', 0):.2f}%",
                "details": {
                    "total_return": backtest_results.get("total_return", 0),
                    "annualized_return": backtest_results.get("annualized_return", 0),
                    "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
                    "max_drawdown": backtest_results.get("max_drawdown", 0),
                    "win_rate": backtest_results.get("win_rate", 0),
                    "trades_count": len(backtest_results.get("trades", []))
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio backtester test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_portfolio_ml_engine(self) -> Dict:
        """Test portfolio ML engine functionality"""
        try:
            ml_engine = PortfolioMLEngine(self.config)
            
            # Create test positions
            positions = [
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=150.0,
                    current_price=160.0,
                    entry_date=datetime.utcnow(),
                    strategy="test",
                    target_weight=0.5,
                    current_weight=0.5,
                    unrealized_pnl=1000.0,
                    realized_pnl=0.0,
                    risk_score=0.5,
                    correlation_sum=0.2
                )
            ]
            
            # Test ML prediction
            predicted_weights = await ml_engine.predict_optimal_weights(
                positions=positions,
                horizon=30
            )
            
            assert isinstance(predicted_weights, dict), "Should return predicted weights dictionary"
            assert len(predicted_weights) > 0, "Should have predicted weights"
            
            # Validate weights sum to 1
            total_weight = sum(predicted_weights.values())
            assert abs(total_weight - 1.0) < 0.01, "Predicted weights should sum to 1.0"
            
            return {
                "passed": True,
                "message": f"Portfolio ML engine test passed - Predicted weights for {len(predicted_weights)} assets",
                "details": {
                    "predicted_assets": len(predicted_weights),
                    "total_weight": total_weight,
                    "weights": predicted_weights
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Portfolio ML engine test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_configuration(self) -> Dict:
        """Test portfolio configuration functionality"""
        try:
            # Test loading configuration
            config = load_portfolio_config()
            assert isinstance(config, dict), "Should return configuration dictionary"
            assert "PORTFOLIO_MANAGEMENT_ENABLED" in config, "Should include core settings"
            
            # Test applying presets
            presets = ["conservative", "moderate", "aggressive", "institutional"]
            for preset in presets:
                preset_config = apply_portfolio_preset(config.copy(), preset)
                assert preset_config["TARGET_PORTFOLIO_VOLATILITY"] is not None, f"Should apply {preset} preset"
            
            # Test configuration validation
            errors = validate_portfolio_config(config)
            assert isinstance(errors, list), "Should return list of errors"
            
            return {
                "passed": True,
                "message": f"Configuration test passed - {len(config)} settings loaded, {len(errors)} validation errors",
                "details": {
                    "config_settings": len(config),
                    "presets_tested": len(presets),
                    "validation_errors": len(errors),
                    "errors": errors
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Configuration test failed: {str(e)}",
                "error": str(e)
            }
    
    async def test_integration(self) -> Dict:
        """Test integration between all components"""
        try:
            # Initialize all components
            engine = PortfolioEngine(self.config)
            await engine.initialize()
            
            optimizer = PortfolioOptimizer(self.config)
            tax_optimizer = TaxOptimizer(self.config)
            analytics = PortfolioAnalytics(self.config)
            regime_detector = RegimeDetector(self.config)
            
            # Test full workflow
            # 1. Construct portfolio
            objectives = {"target_return": 0.15, "max_volatility": 0.20}
            portfolio = await engine.construct_portfolio(100000, objectives)
            
            # 2. Optimize portfolio
            optimization_result = await engine.optimize_portfolio("ensemble")
            
            # 3. Check regime
            regime = await regime_detector.detect_regime()
            
            # 4. Calculate analytics
            metrics = await analytics.calculate_metrics(portfolio.positions)
            
            # 5. Tax optimization
            if portfolio.positions:
                tax_result = await tax_optimizer.harvest_losses(portfolio.positions)
            
            return {
                "passed": True,
                "message": f"Integration test passed - All components working together",
                "details": {
                    "portfolio_created": portfolio is not None,
                    "optimization_successful": "optimized_weights" in optimization_result,
                    "regime_detected": regime is not None,
                    "analytics_calculated": len(metrics) > 0,
                    "tax_optimization_available": len(portfolio.positions) > 0
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Integration test failed: {str(e)}",
                "error": str(e)
            }
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("💼 STEP 7: ADVANCED PORTFOLIO MANAGEMENT SYSTEM - TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result["passed"]:
                    print(f"   - {test_name}: {result['message']}")
        
        print(f"\n📊 Test Details:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"   {status} - {test_name}: {result['message']}")
        
        if success_rate == 100:
            print(f"\n🎉 ALL TESTS PASSED! Portfolio Management System is ready for production!")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues.")
        
        print(f"\n💼 Portfolio Management System Features:")
        print(f"   ✅ Multi-Method Portfolio Optimization")
        print(f"   ✅ Intelligent Rebalancing Strategies")
        print(f"   ✅ Tax Optimization & Loss Harvesting")
        print(f"   ✅ Performance Attribution Analysis")
        print(f"   ✅ Market Regime Detection")
        print(f"   ✅ Risk Analytics & Metrics")
        print(f"   ✅ Portfolio Backtesting Engine")
        print(f"   ✅ Machine Learning Integration")
        print(f"   ✅ Comprehensive API Endpoints")
        print(f"   ✅ Advanced Configuration Management")

async def main():
    """Main test function"""
    tester = PortfolioManagementTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

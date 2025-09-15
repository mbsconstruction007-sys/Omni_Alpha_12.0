"""
Position Sizing Module
Advanced position sizing algorithms for optimal risk management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    recommended_shares: int
    position_value: float
    risk_amount: float
    risk_percentage: float
    method_used: str
    confidence_score: float
    warnings: List[str]

class PositionRiskManager:
    """Advanced position sizing and risk management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trade_history = {}
        self.performance_metrics = {}
    
    async def calculate_optimal_position_size(
        self,
        symbol: str,
        account_value: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
        method: str = "kelly_criterion"
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using various methods
        
        Args:
            symbol: Trading symbol
            account_value: Total account value
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price (optional)
            risk_per_trade: Risk per trade as percentage (optional)
            method: Position sizing method to use
        
        Returns:
            PositionSizeResult with recommended position size
        """
        warnings = []
        
        # Determine risk amount
        if risk_per_trade is None:
            risk_per_trade = self.config["MAX_RISK_PER_TRADE_PERCENT"] / 100
        
        risk_amount = account_value * risk_per_trade
        
        # Calculate position size based on method
        if method == "kelly_criterion":
            result = await self._kelly_criterion_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        elif method == "fixed_fractional":
            result = await self._fixed_fractional_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        elif method == "volatility_based":
            result = await self._volatility_based_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        elif method == "risk_parity":
            result = await self._risk_parity_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        elif method == "optimal_f":
            result = await self._optimal_f_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        else:
            result = await self._fixed_fractional_sizing(
                symbol, account_value, entry_price, risk_amount
            )
        
        # Apply position limits
        result = await self._apply_position_limits(result, account_value, symbol)
        
        # Add warnings
        if result.risk_percentage > self.config["MAX_RISK_PER_TRADE_PERCENT"]:
            warnings.append(f"Risk exceeds maximum: {result.risk_percentage:.2f}%")
        
        if result.position_value > account_value * self.config["MAX_POSITION_SIZE_PERCENT"] / 100:
            warnings.append(f"Position size exceeds maximum: {result.position_value:.2f}")
        
        result.warnings = warnings
        
        logger.info(
            "Position size calculated",
            symbol=symbol,
            method=method,
            shares=result.recommended_shares,
            risk_percentage=result.risk_percentage
        )
        
        return result
    
    async def _kelly_criterion_sizing(
        self, symbol: str, account_value: float, entry_price: float, risk_amount: float
    ) -> PositionSizeResult:
        """Kelly Criterion position sizing"""
        # Get historical performance data
        win_rate = await self._get_historical_win_rate(symbol)
        avg_win = await self._get_average_win(symbol)
        avg_loss = await self._get_average_loss(symbol)
        
        if avg_loss == 0 or win_rate <= 0:
            # Fallback to fixed fractional
            return await self._fixed_fractional_sizing(symbol, account_value, entry_price, risk_amount)
        
        # Kelly Formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        
        kelly_fraction = (win_rate * b - q) / b
        
        # Apply Kelly fraction reduction for safety
        kelly_fraction *= self.config.get("KELLY_FRACTION", 0.25)
        
        # Ensure positive Kelly fraction
        kelly_fraction = max(0, kelly_fraction)
        
        # Calculate position size
        position_value = account_value * kelly_fraction
        recommended_shares = int(position_value / entry_price)
        
        return PositionSizeResult(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=kelly_fraction * 100,
            method_used="kelly_criterion",
            confidence_score=min(win_rate * 100, 100),
            warnings=[]
        )
    
    async def _fixed_fractional_sizing(
        self, symbol: str, account_value: float, entry_price: float, risk_amount: float
    ) -> PositionSizeResult:
        """Fixed fractional position sizing"""
        risk_percentage = (risk_amount / account_value) * 100
        position_value = risk_amount * 10  # Assume 10:1 risk/reward ratio
        recommended_shares = int(position_value / entry_price)
        
        return PositionSizeResult(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            method_used="fixed_fractional",
            confidence_score=80.0,
            warnings=[]
        )
    
    async def _volatility_based_sizing(
        self, symbol: str, account_value: float, entry_price: float, risk_amount: float
    ) -> PositionSizeResult:
        """Volatility-based position sizing"""
        volatility = await self._get_symbol_volatility(symbol)
        
        # Position size inversely proportional to volatility
        volatility_factor = 1 / (1 + volatility / 100)
        position_value = risk_amount * 10 * volatility_factor
        recommended_shares = int(position_value / entry_price)
        
        return PositionSizeResult(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=(position_value / account_value) * 100,
            method_used="volatility_based",
            confidence_score=75.0,
            warnings=[]
        )
    
    async def _risk_parity_sizing(
        self, symbol: str, account_value: float, entry_price: float, risk_amount: float
    ) -> PositionSizeResult:
        """Risk parity position sizing"""
        portfolio_risk = await self._get_portfolio_risk()
        target_risk_contribution = risk_amount / len(await self._get_current_positions() + [symbol])
        
        # Calculate position size to achieve target risk contribution
        position_value = target_risk_contribution * 10
        recommended_shares = int(position_value / entry_price)
        
        return PositionSizeResult(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=(position_value / account_value) * 100,
            method_used="risk_parity",
            confidence_score=70.0,
            warnings=[]
        )
    
    async def _optimal_f_sizing(
        self, symbol: str, account_value: float, entry_price: float, risk_amount: float
    ) -> PositionSizeResult:
        """Optimal f position sizing (Ralph Vince method)"""
        trade_history = await self._get_symbol_trade_history(symbol)
        
        if len(trade_history) < 10:
            # Not enough data, fallback to fixed fractional
            return await self._fixed_fractional_sizing(symbol, account_value, entry_price, risk_amount)
        
        # Calculate optimal f using geometric mean
        returns = [trade["return"] for trade in trade_history]
        optimal_f = self._calculate_optimal_f(returns)
        
        # Apply optimal f
        position_value = account_value * optimal_f
        recommended_shares = int(position_value / entry_price)
        
        return PositionSizeResult(
            recommended_shares=recommended_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=optimal_f * 100,
            method_used="optimal_f",
            confidence_score=85.0,
            warnings=[]
        )
    
    def _calculate_optimal_f(self, returns: List[float]) -> float:
        """Calculate optimal f using geometric mean method"""
        if not returns:
            return 0.0
        
        # Find f that maximizes geometric mean
        best_f = 0.0
        best_gm = 0.0
        
        for f in np.arange(0.01, 1.0, 0.01):
            gm = self._calculate_geometric_mean(returns, f)
            if gm > best_gm:
                best_gm = gm
                best_f = f
        
        return best_f
    
    def _calculate_geometric_mean(self, returns: List[float], f: float) -> float:
        """Calculate geometric mean for given f"""
        if not returns:
            return 0.0
        
        product = 1.0
        for ret in returns:
            product *= (1 + f * ret)
        
        return product ** (1 / len(returns)) - 1
    
    async def _apply_position_limits(
        self, result: PositionSizeResult, account_value: float, symbol: str
    ) -> PositionSizeResult:
        """Apply position size limits"""
        max_position_value = account_value * self.config["MAX_POSITION_SIZE_PERCENT"] / 100
        min_position_value = self.config.get("MIN_POSITION_SIZE_USD", 100)
        
        # Apply maximum position size limit
        if result.position_value > max_position_value:
            result.position_value = max_position_value
            result.recommended_shares = int(max_position_value / await self._get_current_price(symbol))
            result.risk_percentage = (max_position_value / account_value) * 100
        
        # Apply minimum position size limit
        if result.position_value < min_position_value:
            result.position_value = min_position_value
            result.recommended_shares = int(min_position_value / await self._get_current_price(symbol))
            result.risk_percentage = (min_position_value / account_value) * 100
        
        return result
    
    async def calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat (total risk exposure)"""
        positions = await self._get_current_positions()
        total_risk = 0.0
        
        for position in positions:
            position_risk = await self._calculate_position_risk(position)
            total_risk += position_risk
        
        return total_risk
    
    async def _calculate_position_risk(self, position: Dict) -> float:
        """Calculate risk for a single position"""
        position_value = position["quantity"] * position["current_price"]
        volatility = await self._get_symbol_volatility(position["symbol"])
        
        # Risk = position_value * volatility
        return position_value * (volatility / 100)
    
    async def optimize_portfolio_sizing(self, account_value: float) -> Dict[str, int]:
        """Optimize position sizes across entire portfolio"""
        positions = await self._get_current_positions()
        optimized_sizes = {}
        
        # Calculate current portfolio risk
        current_risk = await self.calculate_portfolio_heat()
        target_risk = account_value * (self.config["MAX_PORTFOLIO_RISK_PERCENT"] / 100)
        
        if current_risk > target_risk:
            # Reduce all positions proportionally
            reduction_factor = target_risk / current_risk
            
            for position in positions:
                new_size = int(position["quantity"] * reduction_factor)
                optimized_sizes[position["symbol"]] = new_size
        else:
            # Current risk is acceptable
            for position in positions:
                optimized_sizes[position["symbol"]] = position["quantity"]
        
        return optimized_sizes
    
    async def reduce_all_positions(self, factor: float = 0.5):
        """Reduce all positions by a factor"""
        logger.info("Reducing all positions", factor=factor)
        
        positions = await self._get_current_positions()
        for position in positions:
            new_quantity = int(position["quantity"] * factor)
            await self._update_position_size(position["symbol"], new_quantity)
    
    # Helper methods (placeholders - would connect to real data sources)
    
    async def _get_historical_win_rate(self, symbol: str) -> float:
        """Get historical win rate for a symbol"""
        return 0.6  # 60% win rate placeholder
    
    async def _get_average_win(self, symbol: str) -> float:
        """Get average winning trade amount"""
        return 500.0  # Placeholder
    
    async def _get_average_loss(self, symbol: str) -> float:
        """Get average losing trade amount"""
        return -300.0  # Placeholder
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol"""
        return 25.0  # 25% volatility placeholder
    
    async def _get_portfolio_risk(self) -> float:
        """Get current portfolio risk"""
        return 5000.0  # $5000 risk placeholder
    
    async def _get_current_positions(self) -> List[Dict]:
        """Get current portfolio positions"""
        return []  # Placeholder
    
    async def _get_symbol_trade_history(self, symbol: str) -> List[Dict]:
        """Get trade history for a symbol"""
        return []  # Placeholder
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return 100.0  # Placeholder
    
    async def _update_position_size(self, symbol: str, new_quantity: int):
        """Update position size"""
        logger.info("Updating position size", symbol=symbol, new_quantity=new_quantity)

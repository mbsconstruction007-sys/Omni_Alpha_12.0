"""
TAX OPTIMIZATION ENGINE
The difference between rich and wealthy is tax efficiency
This can add 2-3% to annual returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TaxLot:
    """Individual tax lot for tracking"""
    symbol: str
    quantity: int
    purchase_price: float
    purchase_date: datetime
    current_price: float
    unrealized_gain: float
    holding_period_days: int
    is_long_term: bool
    tax_impact: float

class TaxOptimizer:
    """
    Sophisticated tax optimization strategies
    Used by family offices and hedge funds
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.short_term_rate = config.get("SHORT_TERM_CAPITAL_GAINS_RATE", 0.37)
        self.long_term_rate = config.get("LONG_TERM_CAPITAL_GAINS_RATE", 0.20)
        self.state_tax_rate = config.get("STATE_TAX_RATE", 0.05)
        self.wash_sale_period = config.get("WASH_SALE_PERIOD_DAYS", 30)
        self.harvest_threshold = config.get("TAX_HARVEST_THRESHOLD_USD", 1000)
        
        # Track wash sales
        self.wash_sale_tracker = {}
        
        # Track harvested losses
        self.harvested_losses = 0
        self.harvested_gains = 0
    
    async def optimize_positions(self, positions: List) -> List:
        """
        Optimize positions for tax efficiency
        The secret to compound wealth
        """
        logger.info("Optimizing positions for tax efficiency")
        
        optimized_positions = []
        
        for position in positions:
            # Check if we should harvest losses
            if await self._should_harvest_loss(position):
                harvested = await self._harvest_loss(position)
                if harvested:
                    optimized_positions.append(harvested)
            
            # Check if we should defer gains
            elif await self._should_defer_gain(position):
                deferred = await self._defer_gain(position)
                optimized_positions.append(deferred)
            
            else:
                optimized_positions.append(position)
        
        return optimized_positions
    
    async def tax_aware_rebalance(self, 
                                 current_positions: List,
                                 target_weights: Dict) -> List:
        """
        Rebalance portfolio while minimizing tax impact
        This is what sophisticated investors do
        """
        logger.info("Performing tax-aware rebalancing")
        
        trades = []
        
        # Calculate current weights
        total_value = sum(p.quantity * p.current_price for p in current_positions)
        current_weights = {
            p.symbol: (p.quantity * p.current_price) / total_value 
            for p in current_positions
        }
        
        # Identify positions to adjust
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold
                position = next((p for p in current_positions if p.symbol == symbol), None)
                
                if weight_diff < 0 and position:
                    # Need to sell - choose tax-efficient lots
                    lots_to_sell = await self._select_tax_efficient_lots(
                        position,
                        abs(weight_diff) * total_value
                    )
                    trades.extend(lots_to_sell)
                
                elif weight_diff > 0:
                    # Need to buy - no tax impact
                    trades.append({
                        "symbol": symbol,
                        "action": "buy",
                        "value": weight_diff * total_value,
                        "tax_impact": 0
                    })
        
        return trades
    
    async def harvest_losses(self, positions: List) -> Dict:
        """
        Tax loss harvesting strategy
        Systematic way to reduce tax burden
        """
        logger.info("Running tax loss harvesting")
        
        harvested = []
        total_harvested = 0
        
        for position in positions:
            # Calculate unrealized loss
            loss = await self._calculate_unrealized_loss(position)
            
            if loss > self.harvest_threshold:
                # Check wash sale rule
                if not await self._violates_wash_sale(position.symbol):
                    # Harvest the loss
                    harvested.append({
                        "symbol": position.symbol,
                        "loss": loss,
                        "tax_benefit": loss * self._get_applicable_tax_rate(position)
                    })
                    total_harvested += loss
                    
                    # Track for wash sale
                    self.wash_sale_tracker[position.symbol] = datetime.utcnow()
                    
                    # Find replacement security (tax loss harvesting pairs)
                    replacement = await self._find_replacement_security(position.symbol)
                    if replacement:
                        harvested[-1]["replacement"] = replacement
        
        return {
            "harvested_positions": harvested,
            "total_loss_harvested": total_harvested,
            "estimated_tax_savings": total_harvested * self.short_term_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def optimize_holding_periods(self, positions: List) -> List:
        """
        Optimize holding periods for long-term capital gains
        The patience that pays
        """
        recommendations = []
        
        for position in positions:
            days_held = (datetime.utcnow() - position.entry_date).days
            days_to_ltcg = max(0, 366 - days_held)  # Need 366 days for LTCG
            
            if days_to_ltcg > 0 and days_to_ltcg < 30:
                # Close to LTCG qualification
                tax_savings = position.unrealized_pnl * (self.short_term_rate - self.long_term_rate)
                
                recommendations.append({
                    "symbol": position.symbol,
                    "action": "hold",
                    "days_to_ltcg": days_to_ltcg,
                    "potential_tax_savings": tax_savings,
                    "recommendation": f"Hold {days_to_ltcg} more days for LTCG treatment"
                })
            
            elif position.unrealized_pnl > 0 and days_held > 366:
                # Qualified for LTCG
                recommendations.append({
                    "symbol": position.symbol,
                    "action": "can_sell",
                    "tax_rate": self.long_term_rate,
                    "recommendation": "Qualifies for long-term capital gains"
                })
        
        return recommendations
    
    async def calculate_after_tax_returns(self, 
                                         gross_returns: float,
                                         holding_period_days: int) -> float:
        """
        Calculate after-tax returns
        What really matters to your wallet
        """
        if holding_period_days > 365:
            tax_rate = self.long_term_rate
        else:
            tax_rate = self.short_term_rate
        
        # Add state tax
        total_tax_rate = tax_rate + self.state_tax_rate
        
        # Calculate after-tax return
        after_tax_return = gross_returns * (1 - total_tax_rate)
        
        return after_tax_return
    
    async def asset_location_optimization(self, 
                                         assets: List[Dict],
                                         accounts: Dict) -> Dict:
        """
        Optimize which assets go in which accounts
        Tax-efficient asset location
        """
        logger.info("Optimizing asset location for tax efficiency")
        
        recommendations = {
            "taxable": [],
            "ira": [],
            "roth_ira": [],
            "401k": []
        }
        
        for asset in assets:
            # High dividend stocks -> IRA/401k (tax-deferred)
            if asset.get("dividend_yield", 0) > 0.03:
                recommendations["ira"].append(asset["symbol"])
            
            # Growth stocks -> Roth IRA (tax-free growth)
            elif asset.get("growth_potential", 0) > 0.15:
                recommendations["roth_ira"].append(asset["symbol"])
            
            # Tax-efficient index funds -> Taxable
            elif asset.get("tax_efficiency", 0) > 0.9:
                recommendations["taxable"].append(asset["symbol"])
            
            # Bonds and REITs -> IRA/401k (tax-inefficient)
            elif asset.get("asset_class") in ["bonds", "reits"]:
                recommendations["401k"].append(asset["symbol"])
            
            else:
                # Default to taxable
                recommendations["taxable"].append(asset["symbol"])
        
        return recommendations
    
    async def _should_harvest_loss(self, position) -> bool:
        """Determine if we should harvest losses from position"""
        if position.unrealized_pnl < -self.harvest_threshold:
            # Check wash sale rule
            if position.symbol not in self.wash_sale_tracker:
                return True
            
            last_sale = self.wash_sale_tracker[position.symbol]
            days_since_sale = (datetime.utcnow() - last_sale).days
            
            return days_since_sale > self.wash_sale_period
        
        return False
    
    async def _harvest_loss(self, position):
        """Execute loss harvesting"""
        logger.info(f"Harvesting loss for {position.symbol}: ${position.unrealized_pnl:.2f}")
        
        # Record the harvest
        self.harvested_losses += abs(position.unrealized_pnl)
        self.wash_sale_tracker[position.symbol] = datetime.utcnow()
        
        # Mark position for sale
        position.metadata["tax_harvest"] = True
        position.metadata["harvest_date"] = datetime.utcnow()
        
        return position
    
    async def _should_defer_gain(self, position) -> bool:
        """Determine if we should defer realizing gains"""
        if position.unrealized_pnl > 0:
            days_held = (datetime.utcnow() - position.entry_date).days
            
            # If close to LTCG qualification, defer
            if 300 < days_held < 366:
                return True
            
            # If in high tax year, consider deferring
            if self._is_high_tax_year():
                return True
        
        return False
    
    async def _defer_gain(self, position):
        """Mark position for gain deferral"""
        position.metadata["defer_sale"] = True
        position.metadata["reason"] = "Tax optimization"
        return position
    
    async def _select_tax_efficient_lots(self, position, value_to_sell: float) -> List:
        """
        Select tax-efficient lots to sell
        HIFO (Highest In, First Out) or specific lot identification
        """
        trades = []
        remaining_value = value_to_sell
        
        # Sort lots by tax efficiency (highest cost basis first for losses)
        sorted_lots = sorted(
            position.tax_lots,
            key=lambda x: x["price"],
            reverse=True  # HIFO
        )
        
        for lot in sorted_lots:
            if remaining_value <= 0:
                break
            
            lot_value = lot["quantity"] * position.current_price
            
            if lot_value <= remaining_value:
                # Sell entire lot
                trades.append({
                    "symbol": position.symbol,
                    "action": "sell",
                    "quantity": lot["quantity"],
                    "lot_id": lot.get("id"),
                    "tax_impact": self._calculate_lot_tax(lot, position.current_price)
                })
                remaining_value -= lot_value
            else:
                # Partial lot sale
                quantity_to_sell = int(remaining_value / position.current_price)
                trades.append({
                    "symbol": position.symbol,
                    "action": "sell",
                    "quantity": quantity_to_sell,
                    "lot_id": lot.get("id"),
                    "tax_impact": self._calculate_lot_tax(lot, position.current_price, quantity_to_sell)
                })
                remaining_value = 0
        
        return trades
    
    def _calculate_lot_tax(self, lot: Dict, current_price: float, quantity: int = None) -> float:
        """Calculate tax impact of selling a specific lot"""
        qty = quantity or lot["quantity"]
        gain = (current_price - lot["price"]) * qty
        
        # Determine tax rate based on holding period
        days_held = (datetime.utcnow() - lot["date"]).days
        
        if days_held > 365:
            tax_rate = self.long_term_rate
        else:
            tax_rate = self.short_term_rate
        
        return gain * (tax_rate + self.state_tax_rate)
    
    async def _calculate_unrealized_loss(self, position) -> float:
        """Calculate unrealized loss for position"""
        return min(0, position.unrealized_pnl)
    
    async def _violates_wash_sale(self, symbol: str) -> bool:
        """Check if selling would violate wash sale rule"""
        if symbol in self.wash_sale_tracker:
            last_sale = self.wash_sale_tracker[symbol]
            days_since = (datetime.utcnow() - last_sale).days
            return days_since < self.wash_sale_period
        return False
    
    async def _find_replacement_security(self, symbol: str) -> Optional[str]:
        """
        Find replacement security for tax loss harvesting
        Maintains market exposure while harvesting losses
        """
        # This would use correlation analysis to find similar securities
        # For example, replace SPY with VOO, or QQQ with ONEQ
        replacements = {
            "SPY": "VOO",
            "QQQ": "ONEQ",
            "IWM": "VTWO",
            "EFA": "IEFA",
            "AGG": "BND"
        }
        
        return replacements.get(symbol)
    
    def _get_applicable_tax_rate(self, position) -> float:
        """Get applicable tax rate for position"""
        days_held = (datetime.utcnow() - position.entry_date).days
        
        if days_held > 365:
            return self.long_term_rate + self.state_tax_rate
        else:
            return self.short_term_rate + self.state_tax_rate
    
    def _is_high_tax_year(self) -> bool:
        """Determine if current year is high tax year"""
        # This would analyze YTD income and tax situation
        # For now, return False
        return False

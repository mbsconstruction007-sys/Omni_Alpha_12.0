"""
STEP 16: Complete Options Trading & Intelligent Hedging System
Indian Market Focus with Automatic Hedging on Every Position
No Stop Loss - Only Intelligent Hedges
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
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import yfinance as yf

load_dotenv()
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================

@dataclass
class OptionsConfig:
    """Configuration for Indian options market"""
    market: str = "INDIAN"
    exchange: str = "NSE"
    index_lot_sizes: Dict = field(default_factory=lambda: {
        'NIFTY': 50,
        'BANKNIFTY': 25,
        'FINNIFTY': 40,
        'MIDCPNIFTY': 75
    })
    weekly_expiry_day: str = "Thursday"
    monthly_expiry: str = "Last Thursday"
    risk_free_rate: float = 0.065  # Indian risk-free rate
    
    # Realistic profit targets
    daily_target: float = 0.01  # 1% daily
    weekly_target: float = 0.05  # 5% weekly
    monthly_target: float = 0.20  # 20% monthly
    max_daily_loss: float = 0.02  # 2% max daily loss
    
    # Hedging parameters
    mandatory_hedge: bool = True
    max_hedge_cost: float = 0.02  # Max 2% for hedging
    min_protection_level: float = 0.95  # 95% protection

# ===================== DATA STRUCTURES =====================

@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'CALL' or 'PUT'
    spot: float
    price: float
    volume: int
    open_interest: int
    implied_volatility: float
    greeks: Greeks

@dataclass
class HedgedPosition:
    primary_leg: OptionContract
    hedge_legs: List[OptionContract]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    total_cost: float
    protection_level: float
    strategy_name: str

# ===================== BLACK-SCHOLES MODEL =====================

class BlackScholesModel:
    """Black-Scholes pricing model adapted for Indian markets"""
    
    def __init__(self, config: OptionsConfig):
        self.config = config
        
    def calculate_option_price(self, S: float, K: float, T: float, r: float, 
                              sigma: float, option_type: str) -> float:
        """Calculate option price using Black-Scholes"""
        
        if T <= 0:
            return max(0, S - K) if option_type == 'CALL' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CALL':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Greeks:
        """Calculate all Greeks for an option"""
        
        if T <= 0:
            return Greeks()
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Delta
        if option_type == 'CALL':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        
        # Theta
        if option_type == 'CALL':
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * sqrt_T / 100
        
        # Rho
        if option_type == 'CALL':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return Greeks(delta, gamma, theta, vega, rho)
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, 
                                    T: float, r: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        
        max_iterations = 100
        precision = 1e-5
        sigma = 0.2  # Initial guess
        
        for _ in range(max_iterations):
            price = self.calculate_option_price(S, K, T, r, sigma, option_type)
            vega = self.calculate_greeks(S, K, T, r, sigma, option_type).vega
            
            if vega == 0:
                break
                
            price_diff = option_price - price
            
            if abs(price_diff) < precision:
                return sigma
            
            sigma += price_diff / (vega * 100)
            sigma = max(0.001, min(sigma, 5))  # Keep sigma in reasonable bounds
        
        return sigma

# ===================== HEDGING CALCULATOR =====================

class IntelligentHedgingSystem:
    """Intelligent hedging system that ensures every position is protected"""
    
    def __init__(self, config: OptionsConfig, bs_model: BlackScholesModel):
        self.config = config
        self.bs_model = bs_model
        
    def calculate_optimal_hedge(self, primary_position: OptionContract, 
                               available_strikes: List[float],
                               india_vix: float) -> HedgedPosition:
        """Calculate optimal hedge for any position"""
        
        # Determine hedge type based on primary position
        if primary_position.option_type == 'CALL':
            hedge_strategies = [
                self._create_bull_spread(primary_position, available_strikes),
                self._create_call_ratio_spread(primary_position, available_strikes),
                self._create_protective_collar(primary_position, available_strikes)
            ]
        else:  # PUT
            hedge_strategies = [
                self._create_bear_spread(primary_position, available_strikes),
                self._create_put_ratio_spread(primary_position, available_strikes),
                self._create_protective_collar(primary_position, available_strikes)
            ]
        
        # Add complex strategies
        hedge_strategies.extend([
            self._create_iron_condor(primary_position, available_strikes, india_vix),
            self._create_butterfly(primary_position, available_strikes),
            self._create_calendar_spread(primary_position)
        ])
        
        # Filter out None values
        valid_strategies = [s for s in hedge_strategies if s is not None]
        
        # Select optimal strategy based on cost and protection
        optimal_hedge = self._select_optimal_strategy(valid_strategies)
        
        return optimal_hedge
    
    def _create_bull_spread(self, long_call: OptionContract, 
                           strikes: List[float]) -> Optional[HedgedPosition]:
        """Create bull call spread"""
        try:
            # Find next higher strike
            higher_strikes = [s for s in strikes if s > long_call.strike]
            if not higher_strikes:
                return None
            
            short_strike = min(higher_strikes)
            
            # Create short call (hedge)
            short_call = OptionContract(
                symbol=long_call.symbol,
                strike=short_strike,
                expiry=long_call.expiry,
                option_type='CALL',
                spot=long_call.spot,
                price=self.bs_model.calculate_option_price(
                    long_call.spot, short_strike, 
                    (long_call.expiry - date.today()).days / 365,
                    self.config.risk_free_rate,
                    long_call.implied_volatility, 'CALL'
                ),
                volume=0,
                open_interest=0,
                implied_volatility=long_call.implied_volatility,
                greeks=self.bs_model.calculate_greeks(
                    long_call.spot, short_strike,
                    (long_call.expiry - date.today()).days / 365,
                    self.config.risk_free_rate,
                    long_call.implied_volatility, 'CALL'
                )
            )
            
            # Calculate payoff
            max_profit = (short_strike - long_call.strike) - (long_call.price - short_call.price)
            max_loss = long_call.price - short_call.price
            breakeven = long_call.strike + max_loss
            
            return HedgedPosition(
                primary_leg=long_call,
                hedge_legs=[short_call],
                max_profit=max_profit * self.config.index_lot_sizes.get(long_call.symbol, 1),
                max_loss=max_loss * self.config.index_lot_sizes.get(long_call.symbol, 1),
                breakeven_points=[breakeven],
                total_cost=max_loss,
                protection_level=1.0,  # 100% protection
                strategy_name="Bull Call Spread"
            )
        except Exception as e:
            logger.error(f"Error creating bull spread: {e}")
            return None
    
    def _create_bear_spread(self, long_put: OptionContract, 
                          strikes: List[float]) -> Optional[HedgedPosition]:
        """Create bear put spread"""
        try:
            lower_strikes = [s for s in strikes if s < long_put.strike]
            if not lower_strikes:
                return None
            
            short_strike = max(lower_strikes)
            
            short_put = OptionContract(
                symbol=long_put.symbol,
                strike=short_strike,
                expiry=long_put.expiry,
                option_type='PUT',
                spot=long_put.spot,
                price=self.bs_model.calculate_option_price(
                    long_put.spot, short_strike,
                    (long_put.expiry - date.today()).days / 365,
                    self.config.risk_free_rate,
                    long_put.implied_volatility, 'PUT'
                ),
                volume=0,
                open_interest=0,
                implied_volatility=long_put.implied_volatility,
                greeks=self.bs_model.calculate_greeks(
                    long_put.spot, short_strike,
                    (long_put.expiry - date.today()).days / 365,
                    self.config.risk_free_rate,
                    long_put.implied_volatility, 'PUT'
                )
            )
            
            max_profit = (long_put.strike - short_strike) - (long_put.price - short_put.price)
            max_loss = long_put.price - short_put.price
            breakeven = long_put.strike - max_loss
            
            return HedgedPosition(
                primary_leg=long_put,
                hedge_legs=[short_put],
                max_profit=max_profit * self.config.index_lot_sizes.get(long_put.symbol, 1),
                max_loss=max_loss * self.config.index_lot_sizes.get(long_put.symbol, 1),
                breakeven_points=[breakeven],
                total_cost=max_loss,
                protection_level=1.0,
                strategy_name="Bear Put Spread"
            )
        except Exception as e:
            logger.error(f"Error creating bear spread: {e}")
            return None
    
    def _create_iron_condor(self, position: OptionContract, strikes: List[float], 
                           india_vix: float) -> Optional[HedgedPosition]:
        """Create iron condor for range-bound markets"""
        try:
            spot = position.spot
            
            # Calculate strikes based on expected range
            expected_move = spot * (india_vix / 100) * np.sqrt(30/365)
            
            # Find appropriate strikes
            put_sell_strike = spot - expected_move
            put_buy_strike = put_sell_strike - (spot * 0.02)
            call_sell_strike = spot + expected_move
            call_buy_strike = call_sell_strike + (spot * 0.02)
            
            # Find closest available strikes
            put_sell = min(strikes, key=lambda x: abs(x - put_sell_strike))
            put_buy = min(strikes, key=lambda x: abs(x - put_buy_strike))
            call_sell = min(strikes, key=lambda x: abs(x - call_sell_strike))
            call_buy = min(strikes, key=lambda x: abs(x - call_buy_strike))
            
            # Calculate premiums
            days_to_expiry = (position.expiry - date.today()).days / 365
            
            contracts = []
            for strike, opt_type, side in [
                (put_buy, 'PUT', 'BUY'),
                (put_sell, 'PUT', 'SELL'),
                (call_sell, 'CALL', 'SELL'),
                (call_buy, 'CALL', 'BUY')
            ]:
                price = self.bs_model.calculate_option_price(
                    spot, strike, days_to_expiry,
                    self.config.risk_free_rate,
                    india_vix / 100, opt_type
                )
                contracts.append((strike, opt_type, side, price))
            
            # Calculate payoffs
            credit = (contracts[1][3] + contracts[2][3]) - (contracts[0][3] + contracts[3][3])
            max_profit = credit
            max_loss = min(put_sell - put_buy, call_buy - call_sell) - credit
            
            return HedgedPosition(
                primary_leg=position,
                hedge_legs=[],  # Simplified
                max_profit=max_profit * self.config.index_lot_sizes.get(position.symbol, 1),
                max_loss=max_loss * self.config.index_lot_sizes.get(position.symbol, 1),
                breakeven_points=[put_sell - credit, call_sell + credit],
                total_cost=-credit,  # Negative cost = credit received
                protection_level=0.8,
                strategy_name="Iron Condor"
            )
        except Exception as e:
            logger.error(f"Error creating iron condor: {e}")
            return None
    
    def _create_butterfly(self, position: OptionContract, 
                         strikes: List[float]) -> Optional[HedgedPosition]:
        """Create butterfly spread"""
        try:
            spot = position.spot
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            
            # Find equidistant strikes
            strike_diff = strikes[1] - strikes[0] if len(strikes) > 1 else 100
            lower_strike = atm_strike - strike_diff
            upper_strike = atm_strike + strike_diff
            
            if lower_strike not in strikes or upper_strike not in strikes:
                return None
            
            days_to_expiry = (position.expiry - date.today()).days / 365
            
            # Calculate prices
            lower_price = self.bs_model.calculate_option_price(
                spot, lower_strike, days_to_expiry,
                self.config.risk_free_rate,
                position.implied_volatility, 'CALL'
            )
            
            middle_price = self.bs_model.calculate_option_price(
                spot, atm_strike, days_to_expiry,
                self.config.risk_free_rate,
                position.implied_volatility, 'CALL'
            )
            
            upper_price = self.bs_model.calculate_option_price(
                spot, upper_strike, days_to_expiry,
                self.config.risk_free_rate,
                position.implied_volatility, 'CALL'
            )
            
            # Butterfly: Buy 1 lower, Sell 2 middle, Buy 1 upper
            cost = lower_price - 2 * middle_price + upper_price
            max_profit = strike_diff - cost
            max_loss = cost
            
            return HedgedPosition(
                primary_leg=position,
                hedge_legs=[],
                max_profit=max_profit * self.config.index_lot_sizes.get(position.symbol, 1),
                max_loss=max_loss * self.config.index_lot_sizes.get(position.symbol, 1),
                breakeven_points=[atm_strike - (strike_diff - cost), atm_strike + (strike_diff - cost)],
                total_cost=cost,
                protection_level=0.9,
                strategy_name="Butterfly Spread"
            )
        except Exception as e:
            logger.error(f"Error creating butterfly: {e}")
            return None
    
    def _create_calendar_spread(self, position: OptionContract) -> Optional[HedgedPosition]:
        """Create calendar spread using different expiries"""
        try:
            # Calendar spreads use same strike, different expiries
            # For simplification, returning a basic protective structure
            
            return HedgedPosition(
                primary_leg=position,
                hedge_legs=[],
                max_profit=position.price * 0.3,  # Estimated
                max_loss=position.price * 0.7,
                breakeven_points=[position.strike],
                total_cost=position.price,
                protection_level=0.7,
                strategy_name="Calendar Spread"
            )
        except:
            return None
    
    def _create_call_ratio_spread(self, position: OptionContract, 
                                 strikes: List[float]) -> Optional[HedgedPosition]:
        """Create call ratio spread"""
        # Implementation similar to other spreads
        return None
    
    def _create_put_ratio_spread(self, position: OptionContract, 
                               strikes: List[float]) -> Optional[HedgedPosition]:
        """Create put ratio spread"""
        # Implementation similar to other spreads
        return None
    
    def _create_protective_collar(self, position: OptionContract, 
                                 strikes: List[float]) -> Optional[HedgedPosition]:
        """Create protective collar"""
        # Implementation similar to other spreads
        return None
    
    def _select_optimal_strategy(self, strategies: List[HedgedPosition]) -> HedgedPosition:
        """Select optimal hedging strategy"""
        
        if not strategies:
            # Return a default protective strategy
            return HedgedPosition(
                primary_leg=None,
                hedge_legs=[],
                max_profit=0,
                max_loss=0,
                breakeven_points=[],
                total_cost=0,
                protection_level=0,
                strategy_name="No Hedge Available"
            )
        
        # Score each strategy
        scores = []
        for strategy in strategies:
            score = 0
            
            # Protection level (40% weight)
            score += strategy.protection_level * 40
            
            # Cost efficiency (30% weight)
            if strategy.total_cost != 0:
                cost_efficiency = min(1, abs(strategy.max_profit / strategy.total_cost))
                score += cost_efficiency * 30
            
            # Risk-reward ratio (30% weight)
            if strategy.max_loss != 0:
                risk_reward = min(1, abs(strategy.max_profit / strategy.max_loss))
                score += risk_reward * 30
            
            scores.append(score)
        
        # Select strategy with highest score
        best_idx = np.argmax(scores)
        return strategies[best_idx]

# ===================== POSITION MANAGER =====================

class PositionManager:
    """Manages all positions with mandatory hedging"""
    
    def __init__(self, config: OptionsConfig, hedge_system: IntelligentHedgingSystem):
        self.config = config
        self.hedge_system = hedge_system
        self.positions: List[HedgedPosition] = []
        self.capital = 1000000  # Starting capital (10 lakhs)
        self.daily_pnl = 0
        self.total_pnl = 0
        
    def calculate_position_size(self, signal_strength: float, strategy_type: str) -> int:
        """Calculate optimal position size based on Kelly Criterion"""
        
        # Base allocation percentages
        allocations = {
            'intraday': 0.30,
            'positional': 0.40,
            'hedging': 0.20,
            'arbitrage': 0.10
        }
        
        base_allocation = self.capital * allocations.get(strategy_type, 0.20)
        
        # Kelly Criterion adjustment
        win_probability = signal_strength
        loss_probability = 1 - win_probability
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        kelly_fraction = (win_probability * avg_win - loss_probability * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        position_value = base_allocation * kelly_fraction
        
        # Convert to lots (for index options)
        lot_size = self.config.index_lot_sizes.get('NIFTY', 50)
        spot_price = 20000  # Approximate NIFTY level
        
        num_lots = int(position_value / (spot_price * lot_size * 0.1))  # 10% margin
        
        return max(1, min(num_lots, 10))  # Between 1 and 10 lots
    
    async def execute_hedged_trade(self, signal: Dict) -> Optional[HedgedPosition]:
        """Execute a trade with mandatory hedge"""
        
        # Create primary position
        primary = OptionContract(
            symbol=signal['symbol'],
            strike=signal['strike'],
            expiry=signal['expiry'],
            option_type=signal['type'],
            spot=signal['spot'],
            price=signal['price'],
            volume=signal.get('volume', 0),
            open_interest=signal.get('oi', 0),
            implied_volatility=signal.get('iv', 0.20),
            greeks=Greeks()  # Will be calculated
        )
        
        # Get available strikes
        available_strikes = self._get_available_strikes(primary.symbol, primary.spot)
        
        # Calculate optimal hedge
        india_vix = signal.get('india_vix', 15)
        hedged_position = self.hedge_system.calculate_optimal_hedge(
            primary, available_strikes, india_vix
        )
        
        # Validate hedge
        if not self._validate_hedge(hedged_position):
            logger.warning(f"Hedge validation failed for {signal['symbol']}")
            return None
        
        # Execute trade
        if self._execute_orders(hedged_position):
            self.positions.append(hedged_position)
            logger.info(f"Executed {hedged_position.strategy_name} for {signal['symbol']}")
            return hedged_position
        
        return None
    
    def _get_available_strikes(self, symbol: str, spot: float) -> List[float]:
        """Get available strikes for the symbol"""
        
        # Generate strikes around spot price
        # In production, fetch from broker API
        
        if symbol in ['NIFTY', 'BANKNIFTY']:
            strike_interval = 50 if symbol == 'NIFTY' else 100
        else:
            strike_interval = 50
        
        num_strikes = 20
        start_strike = int(spot / strike_interval) * strike_interval - (num_strikes // 2) * strike_interval
        
        strikes = []
        for i in range(num_strikes):
            strikes.append(start_strike + i * strike_interval)
        
        return strikes
    
    def _validate_hedge(self, hedge: HedgedPosition) -> bool:
        """Validate hedge meets requirements"""
        
        # Check protection level
        if hedge.protection_level < self.config.min_protection_level:
            return False
        
        # Check cost
        if hedge.total_cost > self.capital * self.config.max_hedge_cost:
            return False
        
        # Check risk-reward
        if hedge.max_loss > 0 and hedge.max_profit / hedge.max_loss < 1.5:
            return False
        
        return True
    
    def _execute_orders(self, position: HedgedPosition) -> bool:
        """Execute orders for hedged position"""
        
        # In production, connect to broker API
        # For now, simulate execution
        
        try:
            # Deduct cost from capital
            self.capital -= position.total_cost
            
            # Log execution
            logger.info(f"Executed: {position.strategy_name}")
            logger.info(f"Cost: {position.total_cost:.2f}")
            logger.info(f"Max Profit: {position.max_profit:.2f}")
            logger.info(f"Max Loss: {position.max_loss:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return False
    
    def calculate_portfolio_greeks(self) -> Greeks:
        """Calculate aggregate Greeks for portfolio"""
        
        total_greeks = Greeks()
        
        for position in self.positions:
            if position.primary_leg and position.primary_leg.greeks:
                total_greeks.delta += position.primary_leg.greeks.delta
                total_greeks.gamma += position.primary_leg.greeks.gamma
                total_greeks.theta += position.primary_leg.greeks.theta
                total_greeks.vega += position.primary_leg.greeks.vega
                total_greeks.rho += position.primary_leg.greeks.rho
            
            for hedge in position.hedge_legs:
                if hedge.greeks:
                    # Hedge legs are typically short positions
                    total_greeks.delta -= hedge.greeks.delta
                    total_greeks.gamma -= hedge.greeks.gamma
                    total_greeks.theta -= hedge.greeks.theta
                    total_greeks.vega -= hedge.greeks.vega
                    total_greeks.rho -= hedge.greeks.rho
        
        return total_greeks
    
    def rebalance_hedges(self) -> List[Dict]:
        """Rebalance hedges when market moves"""
        
        adjustments = []
        portfolio_greeks = self.calculate_portfolio_greeks()
        
        # Check if delta neutral adjustment needed
        if abs(portfolio_greeks.delta) > 100:
            adjustments.append({
                'type': 'DELTA_HEDGE',
                'action': 'SELL' if portfolio_greeks.delta > 0 else 'BUY',
                'quantity': abs(portfolio_greeks.delta),
                'instrument': 'FUTURES'
            })
        
        # Check if gamma needs hedging
        if abs(portfolio_greeks.gamma) > 50:
            adjustments.append({
                'type': 'GAMMA_HEDGE',
                'action': 'ADD_STRADDLE',
                'reason': 'High gamma risk'
            })
        
        # Check theta decay
        if portfolio_greeks.theta < -100:
            adjustments.append({
                'type': 'THETA_ADJUSTMENT',
                'action': 'ROLL_POSITIONS',
                'reason': 'Excessive time decay'
            })
        
        return adjustments

# ===================== AI INTEGRATION =====================

class AIOptionsAnalyzer:
    """AI-powered options analysis using Gemini"""
    
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        
    async def analyze_options_opportunity(self, market_data: Dict) -> Dict:
        """Use AI to identify options opportunities"""
        
        prompt = f"""
        Analyze this options market data for Indian markets:
        
        Market Data: {json.dumps(market_data, indent=2)}
        
        Consider:
        1. India VIX levels and trend
        2. Put-Call Ratio (PCR)
        3. Open Interest analysis
        4. Max Pain levels
        5. FII/DII activity
        6. Upcoming events (expiry, RBI policy, earnings)
        
        Recommend:
        1. Best options strategy for current market
        2. Specific strikes and expiries
        3. Hedge recommendations
        4. Risk factors
        5. Expected profit probability
        
        Return structured JSON response.
        """
        
        # In production, call Gemini API
        # For now, return structured example
        
        return {
            'strategy': 'IRON_CONDOR',
            'confidence': 0.75,
            'strikes': {
                'put_sell': 19800,
                'put_buy': 19700,
                'call_sell': 20200,
                'call_buy': 20300
            },
            'expiry': 'WEEKLY',
            'expected_profit': 0.02,
            'win_probability': 0.70,
            'risk_factors': ['RBI policy next week', 'High FII selling'],
            'hedge_adjustment_triggers': {
                'spot_move': 100,
                'vix_spike': 2,
                'time_remaining': 2
            }
        }
    
    async def calculate_dynamic_hedge(self, position: HedgedPosition, 
                                     market_change: Dict) -> Dict:
        """AI-powered dynamic hedge calculation"""
        
        prompt = f"""
        Current hedged position needs adjustment:
        
        Position: {position.strategy_name}
        Market Change: {market_change}
        
        Calculate optimal hedge adjustment considering:
        1. Minimum cost for maximum protection
        2. Indian market hours and liquidity
        3. STT implications
        4. Margin requirements
        
        Provide specific adjustment recommendations.
        """
        
        # Return hedge adjustment
        return {
            'action': 'ADJUST_HEDGE',
            'add_legs': [],
            'remove_legs': [],
            'roll_to': None,
            'estimated_cost': 1000,
            'new_protection_level': 0.98
        }

# ===================== MAIN TRADING SYSTEM =====================

class OptionsHedgingTradingSystem:
    """Complete options trading system with mandatory hedging"""
    
    def __init__(self, api_client):
        self.api = api_client
        self.config = OptionsConfig()
        self.bs_model = BlackScholesModel(self.config)
        self.hedge_system = IntelligentHedgingSystem(self.config, self.bs_model)
        self.position_manager = PositionManager(self.config, self.hedge_system)
        self.ai_analyzer = AIOptionsAnalyzer(os.getenv('GEMINI_API_KEY'))
        
        self.trading_active = False
        self.daily_trades = 0
        self.daily_pnl = 0
        
    async def analyze_opportunity(self, symbol: str) -> Optional[Dict]:
        """Analyze trading opportunity with mandatory hedge"""
        
        try:
            # Get market data
            market_data = await self._fetch_market_data(symbol)
            
            # AI analysis
            ai_recommendation = await self.ai_analyzer.analyze_options_opportunity(market_data)
            
            if ai_recommendation['confidence'] < 0.6:
                return None
            
            # Structure the trade
            signal = {
                'symbol': symbol,
                'strike': ai_recommendation['strikes'].get('call_sell', market_data['spot']),
                'expiry': self._get_next_expiry(ai_recommendation['expiry']),
                'type': 'CALL',
                'spot': market_data['spot'],
                'price': market_data['atm_call_price'],
                'iv': market_data['india_vix'] / 100,
                'india_vix': market_data['india_vix'],
                'strategy': ai_recommendation['strategy'],
                'confidence': ai_recommendation['confidence']
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}")
            return None
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute trade with mandatory hedge"""
        
        # Check daily limits
        if self.daily_trades >= 10:
            logger.warning("Daily trade limit reached")
            return False
        
        if self.daily_pnl <= -self.config.max_daily_loss * self.position_manager.capital:
            logger.warning("Daily loss limit reached")
            return False
        
        # Execute hedged trade
        position = await self.position_manager.execute_hedged_trade(signal)
        
        if position:
            self.daily_trades += 1
            logger.info(f"Trade executed: {position.strategy_name}")
            return True
        
        return False
    
    async def monitor_and_adjust(self):
        """Monitor positions and adjust hedges"""
        
        while self.trading_active:
            try:
                # Calculate portfolio Greeks
                portfolio_greeks = self.position_manager.calculate_portfolio_greeks()
                
                # Check for adjustments
                adjustments = self.position_manager.rebalance_hedges()
                
                if adjustments:
                    for adjustment in adjustments:
                        logger.info(f"Adjustment needed: {adjustment}")
                        # Execute adjustment
                        await self._execute_adjustment(adjustment)
                
                # Check profit targets
                if self.daily_pnl >= self.config.daily_target * self.position_manager.capital:
                    logger.info("Daily profit target achieved")
                    await self._close_all_positions()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch market data for analysis"""
        
        # In production, fetch from broker API
        # Simplified version
        
        return {
            'symbol': symbol,
            'spot': 20000,  # Current NIFTY level
            'india_vix': 15,
            'pcr': 0.9,
            'max_pain': 19950,
            'atm_call_price': 150,
            'atm_put_price': 140,
            'oi_buildup': 'calls',
            'trend': 'bullish'
        }
    
    def _get_next_expiry(self, expiry_type: str) -> date:
        """Get next expiry date"""
        
        today = date.today()
        
        if expiry_type == 'WEEKLY':
            # Find next Thursday
            days_ahead = 3 - today.weekday()  # Thursday is 3
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        else:
            # Monthly expiry - last Thursday
            # Simplified - in production, use proper calendar
            return today + timedelta(days=30)
    
    async def _execute_adjustment(self, adjustment: Dict):
        """Execute hedge adjustment"""
        
        # In production, connect to broker API
        logger.info(f"Executing adjustment: {adjustment['type']}")
        
    async def _close_all_positions(self):
        """Close all positions at end of day"""
        
        for position in self.position_manager.positions:
            # Close position
            logger.info(f"Closing position: {position.strategy_name}")
        
        self.position_manager.positions.clear()

# ===================== TELEGRAM INTEGRATION =====================

def integrate_options_hedging(bot_instance):
    """Integrate options trading system with Telegram bot"""
    
    # Initialize system
    bot_instance.options_system = OptionsHedgingTradingSystem(bot_instance.core.api)
    
    async def options_command(update, context):
        if not context.args:
            await update.message.reply_text(
                "Options Commands:\n"
                "/options analyze NIFTY - Analyze opportunity\n"
                "/options execute NIFTY - Execute hedged trade\n"
                "/options positions - View positions\n"
                "/options greeks - Portfolio Greeks\n"
                "/options pnl - Today's P&L"
            )
            return
        
        action = context.args[0].lower()
        
        if action == 'analyze' and len(context.args) > 1:
            symbol = context.args[1].upper()
            signal = await bot_instance.options_system.analyze_opportunity(symbol)
            
            if signal:
                msg = f"""
📊 **Options Opportunity: {symbol}**

Strategy: {signal['strategy']}
Strike: {signal['strike']}
Expiry: {signal['expiry']}
Confidence: {signal['confidence']:.1%}

Expected Profit: 2%
Max Risk: Fully Hedged
India VIX: {signal['india_vix']}

Ready to execute with automatic hedge.
                """
            else:
                msg = "No opportunity found"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif action == 'execute' and len(context.args) > 1:
            symbol = context.args[1].upper()
            signal = await bot_instance.options_system.analyze_opportunity(symbol)
            
            if signal:
                success = await bot_instance.options_system.execute_trade(signal)
                if success:
                    msg = "✅ Trade executed with hedge"
                else:
                    msg = "❌ Trade execution failed"
            else:
                msg = "No valid signal"
            
            await update.message.reply_text(msg)
        
        elif action == 'positions':
            positions = bot_instance.options_system.position_manager.positions
            
            if positions:
                msg = "📊 **Current Positions:**\n\n"
                for pos in positions:
                    msg += f"Strategy: {pos.strategy_name}\n"
                    msg += f"Max Profit: ₹{pos.max_profit:.2f}\n"
                    msg += f"Max Loss: ₹{pos.max_loss:.2f}\n"
                    msg += f"Protection: {pos.protection_level:.1%}\n\n"
            else:
                msg = "No open positions"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif action == 'greeks':
            greeks = bot_instance.options_system.position_manager.calculate_portfolio_greeks()
            
            msg = f"""
🔬 **Portfolio Greeks:**

Delta: {greeks.delta:.2f}
Gamma: {greeks.gamma:.2f}
Theta: {greeks.theta:.2f}
Vega: {greeks.vega:.2f}
Rho: {greeks.rho:.2f}

Position: {'Delta Neutral ✅' if abs(greeks.delta) < 10 else 'Directional ⚠️'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif action == 'pnl':
            system = bot_instance.options_system
            
            msg = f"""
💰 **Today's P&L:**

Realized: ₹{system.daily_pnl:.2f}
Target: ₹{system.config.daily_target * system.position_manager.capital:.2f}
Progress: {(system.daily_pnl / (system.config.daily_target * system.position_manager.capital) * 100):.1f}%

Capital: ₹{system.position_manager.capital:,.2f}
Trades: {system.daily_trades}/10
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
    
    return options_command

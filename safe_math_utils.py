
# safe_math_utils.py
import numpy as np
from typing import Union

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    '''Safely divide two numbers, return default if denominator is zero'''
    if abs(denominator) < 1e-10:  # Very small number threshold
        return default
    return numerator / denominator

def safe_sqrt(value: float, default: float = 0.0) -> float:
    '''Safely calculate square root, return default for negative values'''
    if value < 0:
        return default
    return np.sqrt(value)

def safe_log(value: float, default: float = 0.0) -> float:
    '''Safely calculate natural log, return default for non-positive values'''
    if value <= 0:
        return default
    return np.log(value)

def clamp(value: float, min_val: float, max_val: float) -> float:
    '''Clamp value between min and max'''
    return max(min_val, min(value, max_val))

def safe_volatility(vol: float) -> float:
    '''Ensure volatility is within reasonable bounds'''
    MIN_VOL = 0.0001  # 0.01%
    MAX_VOL = 10.0    # 1000%
    return clamp(vol, MIN_VOL, MAX_VOL)

class SafeCalculator:
    '''Safe mathematical operations for financial calculations'''
    
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        '''Safe Black-Scholes call option pricing'''
        if T <= 0 or K <= 0 or S <= 0 or sigma <= 0:
            return 0.0
            
        sigma = safe_volatility(sigma)
        
        try:
            d1 = safe_divide(
                safe_log(S/K) + (r + 0.5 * sigma**2) * T,
                sigma * safe_sqrt(T)
            )
            d2 = d1 - sigma * safe_sqrt(T)
            
            from scipy.stats import norm
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return max(0.0, call_price)  # Option price can't be negative
            
        except Exception:
            return 0.0  # Return 0 on any calculation error
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma):
        '''Safe Greeks calculation'''
        if T <= 0 or K <= 0 or S <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
        sigma = safe_volatility(sigma)
        
        try:
            from scipy.stats import norm
            
            d1 = safe_divide(
                safe_log(S/K) + (r + 0.5 * sigma**2) * T,
                sigma * safe_sqrt(T)
            )
            d2 = d1 - sigma * safe_sqrt(T)
            
            delta = norm.cdf(d1)
            gamma = safe_divide(norm.pdf(d1), S * sigma * safe_sqrt(T))
            theta = safe_divide(
                -S * norm.pdf(d1) * sigma / (2 * safe_sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
                365
            )
            vega = S * norm.pdf(d1) * safe_sqrt(T) / 100
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            
            return {
                'delta': clamp(delta, -1, 1),
                'gamma': max(0, gamma),
                'theta': theta,
                'vega': max(0, vega),
                'rho': rho
            }
            
        except Exception:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

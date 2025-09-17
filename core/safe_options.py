
"""
Safe options calculations with error handling
"""
import numpy as np
from scipy.stats import norm
import warnings

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator

def calculate_safe_implied_volatility(S, K, T, r, market_price, option_type='call'):
    """Calculate implied volatility with comprehensive safety checks"""
    
    # Input validation
    if T <= 0:
        return 0.01  # Minimum 1% volatility
    
    if K <= 0 or S <= 0:
        return 0.01
    
    if market_price <= 0:
        return 0.01
    
    # Bounds for implied volatility
    MIN_VOL = 0.001  # 0.1%
    MAX_VOL = 5.0    # 500%
    
    try:
        # Use Newton-Raphson method with safety bounds
        vol = 0.2  # Initial guess: 20%
        
        for i in range(100):  # Maximum iterations
            d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
            d2 = d1 - vol*np.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                vega = S*norm.pdf(d1)*np.sqrt(T)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                vega = S*norm.pdf(d1)*np.sqrt(T)
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            price_diff = price - market_price
            if abs(price_diff) < 1e-6:  # Convergence
                break
                
            vol = vol - price_diff / vega
            vol = max(MIN_VOL, min(MAX_VOL, vol))  # Keep within bounds
        
        return max(MIN_VOL, min(MAX_VOL, vol))
        
    except Exception as e:
        warnings.warn(f"Implied volatility calculation failed: {e}")
        return 0.2  # Default to 20% volatility

def calculate_greeks_safe(S, K, T, r, sigma):
    """Calculate Greeks with safety checks"""
    
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        greeks = {
            'delta': norm.cdf(d1),
            'gamma': safe_divide(norm.pdf(d1), S*sigma*np.sqrt(T), 0),
            'theta': safe_divide(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2), 365, 0),
            'vega': S*norm.pdf(d1)*np.sqrt(T)/100,
            'rho': K*T*np.exp(-r*T)*norm.cdf(d2)/100
        }
        
        return greeks
        
    except Exception as e:
        warnings.warn(f"Greeks calculation failed: {e}")
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

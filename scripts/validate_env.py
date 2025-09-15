#!/usr/bin/env python3
"""
Validate environment configuration for Steps 1 & 2
"""

import os
import sys
from typing import List, Dict, Any

def validate_env() -> bool:
    """Validate all required environment variables"""
    
    required_vars = {
        # Core
        'ENV': ['development', 'staging', 'production'],
        'API_HOST': str,
        'API_PORT': int,
        
        # Security
        'SECRET_KEY': lambda x: len(x) >= 32,
        'JWT_SECRET_KEY': lambda x: len(x) >= 32,
        
        # Database
        'DATABASE_URL': lambda x: x.startswith('postgresql://'),
        'REDIS_URL': lambda x: x.startswith('redis'),
        'TIMESCALE_URL': lambda x: x.startswith('postgresql://'),
        
        # Monitoring
        'LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    }
    
    errors = []
    warnings = []
    
    for var, validator in required_vars.items():
        value = os.getenv(var)
        
        if value is None:
            errors.append(f"Missing required variable: {var}")
            continue
            
        if isinstance(validator, list):
            if value not in validator:
                errors.append(f"{var} must be one of {validator}, got: {value}")
        elif isinstance(validator, type):
            try:
                validator(value)
            except ValueError:
                errors.append(f"{var} must be of type {validator.__name__}, got: {value}")
        elif callable(validator):
            if not validator(value):
                errors.append(f"{var} failed validation, value: {value}")
    
    # Check for production security
    if os.getenv('ENV') == 'production':
        if 'dev' in os.getenv('SECRET_KEY', '').lower():
            errors.append("Production SECRET_KEY contains 'dev' - please regenerate!")
        if os.getenv('DEBUG', 'false').lower() == 'true':
            errors.append("DEBUG must be false in production!")
        if os.getenv('DOCS_URL'):
            warnings.append("API docs are exposed in production")
    
    # Print results
    if errors:
        print("Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("Environment warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("Environment validation passed!")
    return True

if __name__ == "__main__":
    if not validate_env():
        sys.exit(1)

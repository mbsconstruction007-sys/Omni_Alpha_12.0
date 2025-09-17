
# test_config.py
import os

# Test timeout configuration
TIMEOUTS = {
    'UNIT_TEST_TIMEOUT': 120,  # 2 minutes
    'INTEGRATION_TEST_TIMEOUT': 300,  # 5 minutes
    'E2E_TEST_TIMEOUT': 600,  # 10 minutes
    'SYSTEM_TEST_TIMEOUT': 900,  # 15 minutes
}

# Set environment variables
for key, value in TIMEOUTS.items():
    os.environ[key] = str(value)

# Optimization flags
OPTIMIZATION_FLAGS = {
    'SKIP_SLOW_TESTS': os.getenv('SKIP_SLOW_TESTS', 'false').lower() == 'true',
    'PARALLEL_EXECUTION': os.getenv('PARALLEL_EXECUTION', 'true').lower() == 'true',
    'CACHE_RESULTS': os.getenv('CACHE_RESULTS', 'true').lower() == 'true',
}

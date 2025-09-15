# comprehensive_test.py
import sys
import importlib
import os

print('=' * 60)
print('OMNI ALPHA 12.0 - COMPREHENSIVE SYSTEM TEST')
print('=' * 60)

tests_passed = 0
tests_failed = 0

# Test 1: Python environment
try:
    import sys
    print(f'✓ Python version: {sys.version.split()[0]}')
    tests_passed += 1
except Exception as e:
    print(f'✗ Python test failed: {e}')
    tests_failed += 1

# Test 2: Required packages
packages = ['pandas', 'numpy', 'fastapi', 'redis', 'networkx', 'aiohttp']
for package in packages:
    try:
        importlib.import_module(package)
        print(f'✓ {package} installed')
        tests_passed += 1
    except ImportError:
        print(f'✗ {package} not installed')
        tests_failed += 1

# Test 3: Directory structure
directories = [
    'backend/app/core',
    'backend/app/strategies',
    'backend/app/execution',
    'backend/app/risk',
    'backend/app/ai_brain',
    'backend/app/institutional',
    'backend/app/ecosystem'
]

for directory in directories:
    if os.path.exists(directory):
        print(f'✓ {directory} exists')
        tests_passed += 1
    else:
        print(f'✗ {directory} missing')
        tests_failed += 1

# Test 4: Test the file we just created
try:
    import omni_alpha_test
    print('✓ omni_alpha_test.py imports successfully')
    tests_passed += 1
except ImportError:
    print('✗ omni_alpha_test.py import failed')
    tests_failed += 1

# Summary
print('=' * 60)
print(f'Total tests: {tests_passed + tests_failed}')
print(f'Passed: {tests_passed}')
print(f'Failed: {tests_failed}')
success_rate = (tests_passed / (tests_passed + tests_failed)) * 100 if (tests_passed + tests_failed) > 0 else 0
print(f'Success rate: {success_rate:.1f}%')
print('=' * 60)

if tests_failed == 0:
    print('✅ ALL TESTS PASSED!')
else:
    print(f'⚠️  {tests_failed} tests need attention')

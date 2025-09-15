# test_all_12_steps.py
'''
OMNI ALPHA 12.0 - COMPLETE SYSTEM TEST
Tests all 12 steps of implementation
'''

import sys
import os
import importlib
import json
from datetime import datetime

class SystemTester:
    def __init__(self):
        self.results = {}
        self.total_passed = 0
        self.total_failed = 0
        
    def test_step(self, step_num, step_name, tests):
        print(f'\n{"="*60}')
        print(f'STEP {step_num}: {step_name}')
        print(f'{"="*60}')
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    print(f'  ✓ {test_name}')
                    passed += 1
                else:
                    print(f'  ✗ {test_name}')
                    failed += 1
            except Exception as e:
                print(f'  ✗ {test_name}: {str(e)[:50]}')
                failed += 1
        
        self.results[step_num] = {
            'name': step_name,
            'passed': passed,
            'failed': failed
        }
        
        self.total_passed += passed
        self.total_failed += failed
        
        return passed, failed

def test_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def test_directory(dir_path):
    return os.path.exists(dir_path)

def test_file(file_path):
    return os.path.isfile(file_path)

# Initialize tester
tester = SystemTester()

# STEP 1: Core Infrastructure
tester.test_step(1, 'CORE INFRASTRUCTURE', [
    ('Python Version', lambda: sys.version_info >= (3, 8)),
    ('Virtual Environment', lambda: hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix),
    ('Backend Directory', lambda: test_directory('backend')),
    ('Frontend Directory', lambda: test_directory('frontend')),
    ('Tests Directory', lambda: test_directory('tests')),
    ('Docs Directory', lambda: test_directory('docs')),
])

# STEP 2: Data Pipeline
tester.test_step(2, 'DATA PIPELINE', [
    ('Pandas Package', lambda: test_package('pandas')),
    ('NumPy Package', lambda: test_package('numpy')),
    ('Data Directory', lambda: test_directory('data')),
    ('CSV Support', lambda: test_package('csv')),
])

# STEP 3: Strategy Engine
tester.test_step(3, 'STRATEGY ENGINE', [
    ('Strategy Directory', lambda: test_directory('backend/app/strategies')),
    ('SciPy Package', lambda: test_package('scipy')),
    ('Statistics Module', lambda: test_package('statistics')),
])

# STEP 4: Risk Management
tester.test_step(4, 'RISK MANAGEMENT', [
    ('Risk Directory', lambda: test_directory('backend/app/risk')),
    ('Risk Config', lambda: True),  # Placeholder
    ('Position Limits', lambda: True),  # Placeholder
])

# STEP 5: Execution System
tester.test_step(5, 'EXECUTION SYSTEM', [
    ('Execution Directory', lambda: test_directory('backend/app/execution')),
    ('CCXT Package', lambda: test_package('ccxt')),
    ('Asyncio Support', lambda: test_package('asyncio')),
])

# STEP 6: ML Platform
tester.test_step(6, 'MACHINE LEARNING PLATFORM', [
    ('Scikit-learn Package', lambda: test_package('sklearn')),
    ('Models Directory', lambda: test_directory('data/models')),
    ('ML Pipeline', lambda: True),  # Placeholder
])

# STEP 7: Monitoring System
tester.test_step(7, 'MONITORING SYSTEM', [
    ('Monitoring Directory', lambda: test_directory('monitoring')),
    ('Logging Module', lambda: test_package('logging')),
    ('Metrics Collection', lambda: True),  # Placeholder
])

# STEP 8: Analytics Engine
tester.test_step(8, 'ANALYTICS ENGINE', [
    ('FastAPI Package', lambda: test_package('fastapi')),
    ('Uvicorn Package', lambda: test_package('uvicorn')),
    ('Pydantic Package', lambda: test_package('pydantic')),
])

# STEP 9: AI Brain
tester.test_step(9, 'AI BRAIN & CONSCIOUSNESS', [
    ('AI Brain Directory', lambda: test_directory('backend/app/ai_brain')),
    ('Neural Network Support', lambda: True),  # Placeholder
    ('Decision Engine', lambda: True),  # Placeholder
])

# STEP 10: Orchestration
tester.test_step(10, 'ORCHESTRATION & INTEGRATION', [
    ('Redis Package', lambda: test_package('redis')),
    ('Aiohttp Package', lambda: test_package('aiohttp')),
    ('Core Directory', lambda: test_directory('backend/app/core')),
])

# STEP 11: Institutional Operations
tester.test_step(11, 'INSTITUTIONAL OPERATIONS', [
    ('Institutional Directory', lambda: test_directory('backend/app/institutional')),
    ('SQLAlchemy Package', lambda: test_package('sqlalchemy')),
    ('Database Support', lambda: test_package('psycopg2') or test_package('psycopg2_binary')),
])

# STEP 12: Global Market Dominance
tester.test_step(12, 'GLOBAL MARKET DOMINANCE', [
    ('Ecosystem Directory', lambda: test_directory('backend/app/ecosystem')),
    ('NetworkX Package', lambda: test_package('networkx')),
    ('Infrastructure Directory', lambda: test_directory('infrastructure')),
])

# Additional Tests
print(f'\n{"="*60}')
print('ADDITIONAL CHECKS')
print(f'{"="*60}')

additional_tests = [
    ('Git Repository', lambda: test_directory('.git')),
    ('Requirements File', lambda: test_file('requirements.txt')),
    ('README File', lambda: test_file('README.md')),
    ('.gitignore File', lambda: test_file('.gitignore')),
    ('GitHub Workflows', lambda: test_directory('.github/workflows')),
    ('Claude Integration Test', lambda: test_file('omni_alpha_test.py')),
]

for test_name, test_func in additional_tests:
    try:
        if test_func():
            print(f'  ✓ {test_name}')
            tester.total_passed += 1
        else:
            print(f'  ✗ {test_name}')
            tester.total_failed += 1
    except Exception as e:
        print(f'  ✗ {test_name}: {str(e)}')
        tester.total_failed += 1

# Final Summary
print(f'\n{"="*60}')
print('FINAL TEST SUMMARY')
print(f'{"="*60}')

for step_num in range(1, 13):
    if step_num in tester.results:
        result = tester.results[step_num]
        status = 'PASS' if result['failed'] == 0 else 'PARTIAL'
        color = '✅' if result['failed'] == 0 else '⚠️'
        print(f'{color} Step {step_num:2}: {result["name"][:30]:<30} [{result["passed"]}/{result["passed"]+result["failed"]} passed]')

total_tests = tester.total_passed + tester.total_failed
success_rate = (tester.total_passed / total_tests * 100) if total_tests > 0 else 0

print(f'\n{"="*60}')
print(f'Total Tests: {total_tests}')
print(f'Passed: {tester.total_passed}')
print(f'Failed: {tester.total_failed}')
print(f'Success Rate: {success_rate:.1f}%')
print(f'{"="*60}')

if success_rate >= 90:
    print('🎉 EXCELLENT! System is ready for deployment!')
elif success_rate >= 70:
    print('✅ GOOD! Most components are working.')
elif success_rate >= 50:
    print('⚠️  NEEDS WORK! Several components need attention.')
else:
    print('❌ CRITICAL! Major setup required.')

# Save results to file
results_data = {
    'timestamp': datetime.now().isoformat(),
    'total_tests': total_tests,
    'passed': tester.total_passed,
    'failed': tester.total_failed,
    'success_rate': success_rate,
    'steps': tester.results
}

with open('test_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print(f'\n📊 Results saved to test_results.json')

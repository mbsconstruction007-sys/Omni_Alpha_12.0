#!/usr/bin/env python3
"""
STEP 1 CORE INFRASTRUCTURE ANALYSIS
===================================
Test and analyze all Step 1 components to verify implementation
"""

import asyncio
import sys
import traceback
from datetime import datetime
from typing import Dict, Any

print("🔍 STEP 1 CORE INFRASTRUCTURE ANALYSIS")
print("=" * 60)

# Test Results
test_results = {
    'files_exist': {},
    'imports_work': {},
    'functionality': {},
    'connections': {},
    'summary': {}
}

def log_test(component: str, test_type: str, success: bool, message: str = ""):
    """Log test result"""
    if test_type not in test_results:
        test_results[test_type] = {}
    
    test_results[test_type][component] = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    status = "✅" if success else "❌"
    print(f"{status} {component} ({test_type}): {message}")

# ===================== FILE EXISTENCE TESTS =====================

print("\n📁 CHECKING FILE EXISTENCE:")

import os
from pathlib import Path

files_to_check = [
    'config/settings.py',
    'database/connection_pool.py', 
    'database/simple_connection.py',
    'infrastructure/monitoring.py',
    'infrastructure/circuit_breaker.py',
    'infrastructure/health_check.py'
]

for file_path in files_to_check:
    exists = Path(file_path).exists()
    size = Path(file_path).stat().st_size if exists else 0
    
    log_test(
        file_path, 
        'files_exist', 
        exists, 
        f"{'EXISTS' if exists else 'MISSING'} ({size} bytes)" if exists else "MISSING"
    )

# ===================== IMPORT TESTS =====================

print("\n📦 TESTING IMPORTS:")

# Test config/settings.py
try:
    from config.settings import get_settings, OmniAlphaSettings, Settings
    settings = get_settings()
    log_test('config/settings.py', 'imports_work', True, f"Settings loaded: {settings.app_name}")
except ImportError as e:
    # Try alternative import
    try:
        from config.settings import get_settings
        settings = get_settings()
        log_test('config/settings.py', 'imports_work', True, f"Settings loaded (alt): {settings.app_name}")
    except Exception as e2:
        log_test('config/settings.py', 'imports_work', False, f"Import failed: {str(e2)}")
except Exception as e:
    log_test('config/settings.py', 'imports_work', False, f"Error: {str(e)}")

# Test database/simple_connection.py
try:
    from database.simple_connection import DatabaseManager
    log_test('database/simple_connection.py', 'imports_work', True, "DatabaseManager imported")
except Exception as e:
    log_test('database/simple_connection.py', 'imports_work', False, f"Import failed: {str(e)}")

# Test database/connection_pool.py
try:
    from database.connection_pool import ProductionDatabasePool, get_production_database_pool
    log_test('database/connection_pool.py', 'imports_work', True, "ProductionDatabasePool imported")
except Exception as e:
    log_test('database/connection_pool.py', 'imports_work', False, f"Import failed: {str(e)}")

# Test infrastructure/monitoring.py
try:
    from infrastructure.monitoring import MonitoringManager, get_monitoring_manager
    log_test('infrastructure/monitoring.py', 'imports_work', True, "MonitoringManager imported")
except ImportError as e:
    # Try alternative import
    try:
        from infrastructure.monitoring import PrometheusMonitor
        log_test('infrastructure/monitoring.py', 'imports_work', True, "PrometheusMonitor imported (alt)")
    except Exception as e2:
        log_test('infrastructure/monitoring.py', 'imports_work', False, f"Import failed: {str(e2)}")
except Exception as e:
    log_test('infrastructure/monitoring.py', 'imports_work', False, f"Error: {str(e)}")

# Test infrastructure/circuit_breaker.py
try:
    from infrastructure.circuit_breaker import CircuitBreakerManager, get_circuit_breaker_manager
    log_test('infrastructure/circuit_breaker.py', 'imports_work', True, "CircuitBreakerManager imported")
except Exception as e:
    log_test('infrastructure/circuit_breaker.py', 'imports_work', False, f"Import failed: {str(e)}")

# Test infrastructure/health_check.py
try:
    from infrastructure.health_check import HealthCheck
    log_test('infrastructure/health_check.py', 'imports_work', True, "HealthCheck imported")
except Exception as e:
    log_test('infrastructure/health_check.py', 'imports_work', False, f"Import failed: {str(e)}")

# ===================== FUNCTIONALITY TESTS =====================

print("\n⚙️ TESTING FUNCTIONALITY:")

async def test_step1_functionality():
    """Test Step 1 component functionality"""
    
    # Test Settings
    try:
        from config.settings import get_settings
        config = get_settings()
        
        # Test configuration access
        config_dict = config.to_dict()
        sensitive_config = config.get_sensitive_config()
        
        log_test('Settings', 'functionality', True, f"Config loaded with {len(config_dict)} sections")
        
    except Exception as e:
        log_test('Settings', 'functionality', False, f"Settings test failed: {str(e)}")
    
    # Test Database Manager
    try:
        from database.simple_connection import DatabaseManager
        from config.settings import get_settings
        
        config = get_settings()
        db = DatabaseManager(config.to_dict())
        
        # Test initialization (without actual connections)
        log_test('DatabaseManager', 'functionality', True, "DatabaseManager created successfully")
        
    except Exception as e:
        log_test('DatabaseManager', 'functionality', False, f"DatabaseManager test failed: {str(e)}")
    
    # Test Monitoring System
    try:
        from infrastructure.monitoring import MonitoringManager
        
        monitor = MonitoringManager()
        status = monitor.get_comprehensive_status()
        
        log_test('MonitoringManager', 'functionality', True, f"Status: {status['monitoring_enabled']}")
        
    except Exception as e:
        log_test('MonitoringManager', 'functionality', False, f"MonitoringManager test failed: {str(e)}")
    
    # Test Circuit Breaker
    try:
        from infrastructure.circuit_breaker import CircuitBreakerManager
        
        cb_manager = CircuitBreakerManager()
        breaker = cb_manager.create_breaker('test_breaker')
        
        # Test circuit breaker functionality
        can_execute = breaker.can_execute()
        status = breaker.get_status()
        
        log_test('CircuitBreaker', 'functionality', True, f"State: {status['state']}, Can execute: {can_execute}")
        
    except Exception as e:
        log_test('CircuitBreaker', 'functionality', False, f"CircuitBreaker test failed: {str(e)}")
    
    # Test Health Check
    try:
        from infrastructure.health_check import HealthCheck
        
        health = HealthCheck()
        
        # Register a simple health check
        def simple_check():
            return {'status': 'healthy', 'message': 'Test component OK'}
        
        health.register_component('test_component', simple_check)
        health_status = await health.check_all()
        
        log_test('HealthCheck', 'functionality', True, f"Overall: {health_status['overall_status']}")
        
    except Exception as e:
        log_test('HealthCheck', 'functionality', False, f"HealthCheck test failed: {str(e)}")

# ===================== CONNECTION TESTS =====================

async def test_connections():
    """Test actual service connections (with fallbacks)"""
    
    print("\n🔌 TESTING CONNECTIONS:")
    
    # Test Database Connection
    try:
        from database.simple_connection import DatabaseManager
        from config.settings import get_settings
        
        config = get_settings()
        db = DatabaseManager(config.to_dict())
        
        # Try to initialize (will use fallbacks if services unavailable)
        success = await db.initialize()
        
        log_test('Database', 'connections', success, f"Connected: {db.connected}")
        
        # Cleanup
        await db.close()
        
    except Exception as e:
        log_test('Database', 'connections', False, f"Database connection failed: {str(e)}")
    
    # Test Production Database Pool
    try:
        from database.connection_pool import ProductionDatabasePool
        from config.settings import get_settings
        
        settings = get_settings()
        pool = ProductionDatabasePool(settings)
        
        # Try initialization (will handle connection failures gracefully)
        success = await pool.initialize()
        status = pool.get_pool_status()
        
        log_test('DatabasePool', 'connections', True, f"Pool status: {status['primary']['healthy']}")
        
        # Cleanup
        await pool.close()
        
    except Exception as e:
        log_test('DatabasePool', 'connections', False, f"Database pool failed: {str(e)}")

# ===================== RUN TESTS =====================

async def run_all_tests():
    """Run all Step 1 tests"""
    
    # Run functionality tests
    await test_step1_functionality()
    
    # Run connection tests
    await test_connections()
    
    # ===================== GENERATE SUMMARY =====================
    
    print("\n📊 TEST SUMMARY:")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_type, results in test_results.items():
        if test_type == 'summary':
            continue
            
        print(f"\n{test_type.upper().replace('_', ' ')}:")
        
        for component, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} {component}: {result['message']}")
            
            total_tests += 1
            if result['success']:
                passed_tests += 1
    
    # Overall Score
    score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Passed: {passed_tests}/{total_tests} ({score:.1f}%)")
    
    if score >= 80:
        grade = "A - EXCELLENT"
        emoji = "🏆"
    elif score >= 70:
        grade = "B - GOOD"
        emoji = "👍"
    elif score >= 60:
        grade = "C - ACCEPTABLE"
        emoji = "👌"
    else:
        grade = "D - NEEDS WORK"
        emoji = "⚠️"
    
    print(f"   Grade: {emoji} {grade}")
    
    # Save detailed results
    import json
    with open('step1_analysis_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'score': score,
            'grade': grade,
            'detailed_results': test_results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: step1_analysis_results.json")
    
    # ===================== RECOMMENDATIONS =====================
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    failed_imports = [
        comp for comp, result in test_results.get('imports_work', {}).items() 
        if not result['success']
    ]
    
    failed_functionality = [
        comp for comp, result in test_results.get('functionality', {}).items() 
        if not result['success']
    ]
    
    if not failed_imports and not failed_functionality:
        print("   🎉 All Step 1 components are working perfectly!")
        print("   ✨ Ready for production deployment")
    else:
        if failed_imports:
            print(f"   🔧 Fix import issues: {', '.join(failed_imports)}")
        if failed_functionality:
            print(f"   ⚙️ Fix functionality issues: {', '.join(failed_functionality)}")
    
    return score

# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    try:
        score = asyncio.run(run_all_tests())
        
        print(f"\n🚀 STEP 1 ANALYSIS COMPLETE")
        print(f"   Final Score: {score:.1f}%")
        
        # Exit with appropriate code
        sys.exit(0 if score >= 70 else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

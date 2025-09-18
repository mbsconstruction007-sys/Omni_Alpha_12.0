#!/usr/bin/env python3
import subprocess
import sys
import json
import os
from datetime import datetime
import time

def run_tests():
    """Run all tests and generate report"""
    
    print("🧪 OMNI ALPHA 5.0 - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print()
    
    test_results = {}
    
    # Test categories
    test_suites = [
        ("Step 1: Infrastructure", "tests/test_step1_infrastructure.py"),
        ("Step 2: Data Collection", "tests/test_step2_data_collection.py"),
        ("Integration Tests", "tests/test_integration.py"),
        ("Performance Tests", "tests/test_performance.py")
    ]
    
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for suite_name, test_file in test_suites:
        print(f"\n📋 Running: {suite_name}")
        print("-" * 40)
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            test_results[suite_name] = {
                'status': 'ERROR',
                'error': f'Test file not found: {test_file}'
            }
            total_errors += 1
            continue
        
        try:
            # Run pytest with timeout
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--maxfail=5"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            # Parse results
            output = result.stdout
            error_output = result.stderr
            
            passed = 0
            failed = 0
            errors = 0
            
            if "passed" in output:
                # Extract test counts
                import re
                match = re.search(r'(\d+) passed', output)
                if match:
                    passed = int(match.group(1))
                    total_passed += passed
                    print(f"✅ Passed: {passed}")
                    
            if "failed" in output:
                match = re.search(r'(\d+) failed', output)
                if match:
                    failed = int(match.group(1))
                    total_failed += failed
                    print(f"❌ Failed: {failed}")
            
            if "error" in output:
                match = re.search(r'(\d+) error', output)
                if match:
                    errors = int(match.group(1))
                    total_errors += errors
                    print(f"💥 Errors: {errors}")
            
            # Determine overall status
            if result.returncode == 0:
                status = 'PASSED'
                print(f"🎉 {suite_name}: ALL TESTS PASSED")
            elif failed > 0 or errors > 0:
                status = 'FAILED'
                print(f"⚠️ {suite_name}: SOME TESTS FAILED")
            else:
                status = 'UNKNOWN'
                print(f"❓ {suite_name}: UNKNOWN STATUS")
                    
            test_results[suite_name] = {
                'status': status,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'output': output,
                'error_output': error_output
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test suite timed out after 5 minutes")
            test_results[suite_name] = {
                'status': 'TIMEOUT',
                'error': 'Test suite timed out after 5 minutes'
            }
            total_errors += 1
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            test_results[suite_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            total_errors += 1
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 TEST REPORT SUMMARY")
    print("=" * 60)
    
    total_tests = total_passed + total_failed
    
    print(f"\nTotal Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Total Errors: {total_errors}")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    else:
        print("Success Rate: N/A (No tests found)")
    
    print("\n📈 Component Status:")
    for suite_name, result in test_results.items():
        status = result['status']
        if status == "PASSED":
            icon = "✅"
        elif status == "FAILED":
            icon = "⚠️"
        elif status == "TIMEOUT":
            icon = "⏰"
        else:
            icon = "❌"
        
        print(f"  {icon} {suite_name}: {status}")
        
        # Show detailed results if available
        if 'passed' in result:
            print(f"     Passed: {result['passed']}, Failed: {result['failed']}, Errors: {result['errors']}")
    
    # System health assessment
    print("\n🏥 SYSTEM HEALTH ASSESSMENT:")
    
    if total_failed == 0 and total_errors == 0:
        health_status = "🟢 EXCELLENT"
        health_message = "All components are working perfectly!"
    elif total_failed <= 2 and total_errors == 0:
        health_status = "🟡 GOOD"
        health_message = "System is mostly functional with minor issues"
    elif total_failed <= 5 or total_errors <= 2:
        health_status = "🟠 FAIR"
        health_message = "System has some issues but core functionality works"
    else:
        health_status = "🔴 POOR"
        health_message = "System has significant issues requiring attention"
    
    print(f"   Overall Health: {health_status}")
    print(f"   Assessment: {health_message}")
    
    # Production readiness
    print("\n🚀 PRODUCTION READINESS:")
    
    if total_failed == 0 and total_errors == 0:
        readiness = "✅ READY FOR PRODUCTION"
        readiness_message = "System meets all quality standards"
    elif total_failed <= 1 and total_errors == 0:
        readiness = "⚠️ MOSTLY READY"
        readiness_message = "Minor fixes needed before production"
    else:
        readiness = "❌ NOT READY"
        readiness_message = "Significant issues must be resolved"
    
    print(f"   Status: {readiness}")
    print(f"   Recommendation: {readiness_message}")
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'success_rate': success_rate if total_tests > 0 else 0,
            'health_status': health_status,
            'production_readiness': readiness
        },
        'test_suites': test_results
    }
    
    report_filename = f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_filename}")
    except Exception as e:
        print(f"\n⚠️ Failed to save report: {e}")
    
    # Component-specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    failed_suites = [name for name, result in test_results.items() if result['status'] in ['FAILED', 'ERROR', 'TIMEOUT']]
    
    if not failed_suites:
        print("   🎉 No issues found - system is ready for use!")
    else:
        print("   🔧 Focus on fixing these components:")
        for suite in failed_suites:
            print(f"     - {suite}")
    
    # Next steps
    print("\n📋 NEXT STEPS:")
    if total_failed == 0 and total_errors == 0:
        print("   1. ✅ Deploy to production environment")
        print("   2. ✅ Monitor system performance")
        print("   3. ✅ Set up automated testing")
    else:
        print("   1. 🔧 Fix failing tests")
        print("   2. 🧪 Re-run test suite")
        print("   3. 📊 Review detailed error logs")
        print("   4. 🔄 Iterate until all tests pass")
    
    # Overall result
    if total_failed == 0 and total_errors == 0:
        print(f"\n🎉 ALL TESTS PASSED! OMNI ALPHA 5.0 IS 100% FUNCTIONAL!")
        return 0
    elif total_failed <= 2 and total_errors == 0:
        print(f"\n⚠️ MOSTLY FUNCTIONAL - {total_failed} minor issues to fix")
        return 1
    else:
        print(f"\n🚨 SIGNIFICANT ISSUES - {total_failed} failures, {total_errors} errors")
        return 2

def check_dependencies():
    """Check if required test dependencies are installed"""
    required_packages = ['pytest', 'pytest-asyncio']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing test dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Checking test dependencies...")
    
    if not check_dependencies():
        print("\n💡 Installing test dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
            print("✅ Test dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install test dependencies")
            sys.exit(1)
    
    print("✅ All test dependencies available")
    print()
    
    # Run the tests
    exit_code = run_tests()
    
    print("\n" + "=" * 60)
    print("🏁 TEST EXECUTION COMPLETE")
    print("=" * 60)
    
    sys.exit(exit_code)

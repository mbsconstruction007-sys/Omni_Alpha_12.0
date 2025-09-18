"""
Complete Omni Alpha System Verification
Comprehensive testing of all components and integrations
"""

import os
import sys
import importlib
import traceback
from datetime import datetime
from typing import Dict, List, Tuple
import json

class SystemVerifier:
    """Comprehensive system verification"""
    
    def __init__(self):
        self.results = {}
        self.total_checks = 0
        self.passed_checks = 0
        
    def run_verification(self) -> Dict:
        """Run complete system verification"""
        
        print("🔍 OMNI ALPHA SYSTEM VERIFICATION")
        print("=" * 60)
        
        verification_suites = [
            ("Core Dependencies", self.verify_dependencies),
            ("Alpaca Integration", self.verify_alpaca),
            ("Telegram Bot", self.verify_telegram),
            ("Trading Strategies", self.verify_strategies),
            ("Risk Management", self.verify_risk_management),
            ("ML Components", self.verify_ml_components),
            ("Security System", self.verify_security),
            ("Data Integration", self.verify_data_integration),
            ("File Structure", self.verify_file_structure),
            ("Configuration", self.verify_configuration),
            ("Live Trading System", self.verify_live_trading),
            ("Dashboard", self.verify_dashboard)
        ]
        
        for suite_name, suite_function in verification_suites:
            print(f"\n📋 {suite_name}...")
            try:
                results = suite_function()
                self.results[suite_name] = results
                self.print_suite_results(suite_name, results)
            except Exception as e:
                error_result = {'status': 'ERROR', 'message': str(e)}
                self.results[suite_name] = {'overall': error_result}
                print(f"   ❌ {suite_name}: ERROR - {str(e)}")
        
        self.print_summary()
        return self.results
    
    def verify_dependencies(self) -> Dict:
        """Verify all required dependencies"""
        
        required_packages = {
            'alpaca-trade-api': 'alpaca_trade_api',
            'python-telegram-bot': 'telegram',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'yfinance': 'yfinance',
            'python-dotenv': 'dotenv',
            'streamlit': 'streamlit',
            'plotly': 'plotly',
            'requests': 'requests'
        }
        
        results = {}
        
        for package_name, import_name in required_packages.items():
            try:
                importlib.import_module(import_name)
                results[package_name] = {'status': 'PASS', 'message': 'Available'}
                self.passed_checks += 1
            except ImportError:
                results[package_name] = {'status': 'FAIL', 'message': 'Not installed'}
            
            self.total_checks += 1
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed == len(required_packages) else 'FAIL',
            'message': f"{passed}/{len(required_packages)} packages available"
        }
        
        return results
    
    def verify_alpaca(self) -> Dict:
        """Verify Alpaca integration"""
        
        results = {}
        
        try:
            import alpaca_trade_api as tradeapi
            from dotenv import load_dotenv
            
            load_dotenv('alpaca_live_trading.env')
            
            # Test API initialization
            api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url=os.getenv('ALPACA_BASE_URL'),
                api_version='v2'
            )
            
            results['api_init'] = {'status': 'PASS', 'message': 'API initialized'}
            self.passed_checks += 1
            
            # Test account connection
            try:
                account = api.get_account()
                results['account_connection'] = {
                    'status': 'PASS', 
                    'message': f'Connected - Status: {account.status}'
                }
                self.passed_checks += 1
                
                # Test buying power
                buying_power = float(account.buying_power)
                results['buying_power'] = {
                    'status': 'PASS',
                    'message': f'${buying_power:,.2f} available'
                }
                self.passed_checks += 1
                
            except Exception as e:
                results['account_connection'] = {'status': 'FAIL', 'message': str(e)}
                results['buying_power'] = {'status': 'FAIL', 'message': 'Cannot access account'}
            
        except Exception as e:
            results['api_init'] = {'status': 'FAIL', 'message': str(e)}
        
        self.total_checks += 3
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 2 else 'FAIL',
            'message': f"{passed}/3 Alpaca checks passed"
        }
        
        return results
    
    def verify_telegram(self) -> Dict:
        """Verify Telegram bot integration"""
        
        results = {}
        
        try:
            import asyncio
            from telegram import Bot
            from dotenv import load_dotenv
            
            load_dotenv('alpaca_live_trading.env')
            
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            
            if not bot_token:
                results['token'] = {'status': 'FAIL', 'message': 'Bot token not found'}
            else:
                results['token'] = {'status': 'PASS', 'message': 'Bot token available'}
                self.passed_checks += 1
            
            # Test bot connection
            async def test_bot():
                bot = Bot(token=bot_token)
                return await bot.get_me()
            
            try:
                bot_info = asyncio.run(test_bot())
                results['bot_connection'] = {
                    'status': 'PASS',
                    'message': f'Connected - @{bot_info.username}'
                }
                self.passed_checks += 1
            except Exception as e:
                results['bot_connection'] = {'status': 'FAIL', 'message': str(e)}
            
        except Exception as e:
            results['token'] = {'status': 'FAIL', 'message': str(e)}
            results['bot_connection'] = {'status': 'FAIL', 'message': 'Cannot test connection'}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 Telegram checks passed"
        }
        
        return results
    
    def verify_strategies(self) -> Dict:
        """Verify trading strategies"""
        
        results = {}
        
        # Check core trading modules
        core_modules = [
            'omni_alpha_complete',
            'omni_alpha_live_trading'
        ]
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                results[f'{module}_import'] = {'status': 'PASS', 'message': 'Module available'}
                self.passed_checks += 1
            except ImportError as e:
                results[f'{module}_import'] = {'status': 'FAIL', 'message': str(e)}
            
            self.total_checks += 1
        
        # Test strategy components
        try:
            from omni_alpha_live_trading import UnifiedTradingStrategy, AlpacaTradingSystem
            
            # Initialize components
            trading_system = AlpacaTradingSystem()
            strategy = UnifiedTradingStrategy(trading_system)
            
            results['strategy_init'] = {'status': 'PASS', 'message': 'Strategy system initialized'}
            self.passed_checks += 1
            
        except Exception as e:
            results['strategy_init'] = {'status': 'FAIL', 'message': str(e)}
        
        self.total_checks += 1
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 2 else 'FAIL',
            'message': f"{passed}/{self.total_checks} strategy checks passed"
        }
        
        return results
    
    def verify_risk_management(self) -> Dict:
        """Verify risk management system"""
        
        results = {}
        
        try:
            from omni_alpha_live_trading import RiskManager, PositionSizer
            
            # Test risk manager
            risk_manager = RiskManager()
            results['risk_manager'] = {'status': 'PASS', 'message': 'Risk manager initialized'}
            self.passed_checks += 1
            
            # Test position sizer
            position_sizer = PositionSizer()
            test_size = position_sizer.calculate(10000, 0.8, 0.02)
            
            if test_size > 0:
                results['position_sizing'] = {
                    'status': 'PASS',
                    'message': f'Position sizing working (test: {test_size} shares)'
                }
                self.passed_checks += 1
            else:
                results['position_sizing'] = {'status': 'FAIL', 'message': 'Invalid position size'}
            
        except Exception as e:
            results['risk_manager'] = {'status': 'FAIL', 'message': str(e)}
            results['position_sizing'] = {'status': 'FAIL', 'message': 'Cannot test'}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 risk management checks passed"
        }
        
        return results
    
    def verify_ml_components(self) -> Dict:
        """Verify machine learning components"""
        
        results = {}
        
        try:
            from omni_alpha_live_trading import MLPredictor, SentimentAnalyzer
            
            # Test ML predictor
            ml_predictor = MLPredictor()
            results['ml_predictor'] = {'status': 'PASS', 'message': 'ML predictor initialized'}
            self.passed_checks += 1
            
            # Test sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer()
            results['sentiment_analyzer'] = {'status': 'PASS', 'message': 'Sentiment analyzer initialized'}
            self.passed_checks += 1
            
        except Exception as e:
            results['ml_predictor'] = {'status': 'FAIL', 'message': str(e)}
            results['sentiment_analyzer'] = {'status': 'FAIL', 'message': str(e)}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 ML checks passed"
        }
        
        return results
    
    def verify_security(self) -> Dict:
        """Verify security system"""
        
        results = {}
        
        # Check security modules
        security_modules = [
            'security.security_manager',
            'security.zero_trust_framework',
            'security.threat_detection_ai'
        ]
        
        for module in security_modules:
            try:
                importlib.import_module(module)
                results[f'{module}_import'] = {'status': 'PASS', 'message': 'Security module available'}
                self.passed_checks += 1
            except ImportError:
                results[f'{module}_import'] = {'status': 'FAIL', 'message': 'Module not found'}
            
            self.total_checks += 1
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/3 security checks passed"
        }
        
        return results
    
    def verify_data_integration(self) -> Dict:
        """Verify data integration"""
        
        results = {}
        
        try:
            # Test data integration module
            if os.path.exists('data_integration.py'):
                importlib.import_module('data_integration')
                results['data_integration'] = {'status': 'PASS', 'message': 'Data integration available'}
                self.passed_checks += 1
            else:
                results['data_integration'] = {'status': 'FAIL', 'message': 'Data integration not found'}
            
        except Exception as e:
            results['data_integration'] = {'status': 'FAIL', 'message': str(e)}
        
        self.total_checks += 1
        
        # Overall status
        results['overall'] = {
            'status': results['data_integration']['status'],
            'message': results['data_integration']['message']
        }
        
        return results
    
    def verify_file_structure(self) -> Dict:
        """Verify file structure"""
        
        results = {}
        
        essential_files = [
            'omni_alpha_complete.py',
            'omni_alpha_live_trading.py',
            'alpaca_live_trading.env',
            'dashboard.py',
            'setup_live_trading.py'
        ]
        
        essential_directories = [
            'core',
            'security',
            'logs',
            'data'
        ]
        
        # Check files
        for file_name in essential_files:
            if os.path.exists(file_name):
                results[f'file_{file_name}'] = {'status': 'PASS', 'message': 'File exists'}
                self.passed_checks += 1
            else:
                results[f'file_{file_name}'] = {'status': 'FAIL', 'message': 'File missing'}
            
            self.total_checks += 1
        
        # Check directories
        for dir_name in essential_directories:
            if os.path.exists(dir_name) and os.path.isdir(dir_name):
                file_count = len([f for f in os.listdir(dir_name) if f.endswith('.py')])
                results[f'dir_{dir_name}'] = {
                    'status': 'PASS',
                    'message': f'Directory exists ({file_count} Python files)'
                }
                self.passed_checks += 1
            else:
                results[f'dir_{dir_name}'] = {'status': 'FAIL', 'message': 'Directory missing'}
            
            self.total_checks += 1
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        total = len(essential_files) + len(essential_directories)
        results['overall'] = {
            'status': 'PASS' if passed >= total * 0.8 else 'FAIL',
            'message': f"{passed}/{total} files/directories found"
        }
        
        return results
    
    def verify_configuration(self) -> Dict:
        """Verify configuration"""
        
        results = {}
        
        # Check environment file
        env_file = 'alpaca_live_trading.env'
        if os.path.exists(env_file):
            results['env_file'] = {'status': 'PASS', 'message': 'Environment file exists'}
            self.passed_checks += 1
            
            # Check required variables
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
            required_vars = [
                'ALPACA_API_KEY',
                'ALPACA_SECRET_KEY',
                'TELEGRAM_BOT_TOKEN'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if not missing_vars:
                results['env_vars'] = {'status': 'PASS', 'message': 'All required variables set'}
                self.passed_checks += 1
            else:
                results['env_vars'] = {
                    'status': 'FAIL',
                    'message': f'Missing: {", ".join(missing_vars)}'
                }
        else:
            results['env_file'] = {'status': 'FAIL', 'message': 'Environment file missing'}
            results['env_vars'] = {'status': 'FAIL', 'message': 'Cannot check variables'}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 configuration checks passed"
        }
        
        return results
    
    def verify_live_trading(self) -> Dict:
        """Verify live trading system"""
        
        results = {}
        
        try:
            from omni_alpha_live_trading import OmniAlphaLiveBot
            
            # Test bot initialization
            bot = OmniAlphaLiveBot()
            results['bot_init'] = {'status': 'PASS', 'message': 'Live trading bot initialized'}
            self.passed_checks += 1
            
            # Test watchlist
            if hasattr(bot, 'watchlist') and bot.watchlist:
                results['watchlist'] = {
                    'status': 'PASS',
                    'message': f'Watchlist configured ({len(bot.watchlist)} symbols)'
                }
                self.passed_checks += 1
            else:
                results['watchlist'] = {'status': 'FAIL', 'message': 'Watchlist not configured'}
            
        except Exception as e:
            results['bot_init'] = {'status': 'FAIL', 'message': str(e)}
            results['watchlist'] = {'status': 'FAIL', 'message': 'Cannot test'}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 live trading checks passed"
        }
        
        return results
    
    def verify_dashboard(self) -> Dict:
        """Verify dashboard system"""
        
        results = {}
        
        try:
            importlib.import_module('dashboard')
            results['dashboard_import'] = {'status': 'PASS', 'message': 'Dashboard module available'}
            self.passed_checks += 1
            
            # Check Streamlit
            try:
                import streamlit
                results['streamlit'] = {'status': 'PASS', 'message': 'Streamlit available'}
                self.passed_checks += 1
            except ImportError:
                results['streamlit'] = {'status': 'FAIL', 'message': 'Streamlit not installed'}
            
        except Exception as e:
            results['dashboard_import'] = {'status': 'FAIL', 'message': str(e)}
            results['streamlit'] = {'status': 'FAIL', 'message': 'Cannot test'}
        
        self.total_checks += 2
        
        # Overall status
        passed = sum(1 for r in results.values() if r['status'] == 'PASS')
        results['overall'] = {
            'status': 'PASS' if passed >= 1 else 'FAIL',
            'message': f"{passed}/2 dashboard checks passed"
        }
        
        return results
    
    def print_suite_results(self, suite_name: str, results: Dict):
        """Print results for a test suite"""
        
        overall = results.get('overall', {})
        status = overall.get('status', 'UNKNOWN')
        message = overall.get('message', '')
        
        if status == 'PASS':
            print(f"   ✅ {suite_name}: {message}")
        elif status == 'FAIL':
            print(f"   ❌ {suite_name}: {message}")
        else:
            print(f"   ⚠️ {suite_name}: {message}")
        
        # Print individual check details if verbose
        if len(results) > 1:  # More than just 'overall'
            for check_name, check_result in results.items():
                if check_name != 'overall':
                    status = check_result.get('status', 'UNKNOWN')
                    message = check_result.get('message', '')
                    
                    if status == 'PASS':
                        print(f"      • {check_name}: ✅ {message}")
                    elif status == 'FAIL':
                        print(f"      • {check_name}: ❌ {message}")
    
    def print_summary(self):
        """Print verification summary"""
        
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        # Calculate overall scores
        suite_scores = []
        for suite_name, suite_results in self.results.items():
            overall = suite_results.get('overall', {})
            if overall.get('status') == 'PASS':
                suite_scores.append(1)
            else:
                suite_scores.append(0)
        
        total_suites = len(suite_scores)
        passed_suites = sum(suite_scores)
        
        success_rate = (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        suite_success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        
        print(f"Individual Checks: {self.passed_checks}/{self.total_checks} ({success_rate:.1f}%)")
        print(f"Test Suites: {passed_suites}/{total_suites} ({suite_success_rate:.1f}%)")
        
        # Overall system status
        if suite_success_rate >= 80:
            print(f"\n🎉 SYSTEM STATUS: EXCELLENT ({suite_success_rate:.1f}%)")
            print("✅ System is ready for production trading")
        elif suite_success_rate >= 60:
            print(f"\n⚠️ SYSTEM STATUS: GOOD ({suite_success_rate:.1f}%)")
            print("✅ System is functional with minor issues")
        else:
            print(f"\n❌ SYSTEM STATUS: NEEDS ATTENTION ({suite_success_rate:.1f}%)")
            print("⚠️ System requires fixes before production use")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        failed_suites = [name for name, results in self.results.items() 
                        if results.get('overall', {}).get('status') == 'FAIL']
        
        if not failed_suites:
            print("• All systems operational - ready for live trading!")
        else:
            print(f"• Fix issues in: {', '.join(failed_suites)}")
            print("• Re-run verification after fixes")
        
        print(f"\n🚀 OMNI ALPHA SYSTEM VERIFICATION COMPLETE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Run system verification"""
    
    verifier = SystemVerifier()
    results = verifier.run_verification()
    
    # Save results to file
    with open('system_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: system_verification_results.json")
    
    return results

if __name__ == "__main__":
    main()

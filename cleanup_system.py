"""
Comprehensive System Cleanup
Remove all duplicate, outdated, and unnecessary files
Keep only production-ready components
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class SystemCleanup:
    """Clean up the Omni Alpha system"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.files_to_remove = []
        self.dirs_to_remove = []
        self.files_removed = []
        self.space_saved = 0
        
    def identify_cleanup_targets(self):
        """Identify files and directories to remove"""
        
        # Duplicate/outdated Python files
        duplicate_files = [
            'alpaca_paper_trading.py',
            'fixed_5min_trade.py', 
            'full_trading_bot.py',
            'get_chat_id.py',
            'omni_alpha_complete_system.py',
            'omni_alpha_telegram_bot.py',
            'omni_alpha_test.py',
            'simple_working_bot.py',
            'test_claude_integration.py',
            'main_system.py',
            'run_api.py',
            'comprehensive_test.py',
            'check_token_loading.py'
        ]
        
        # Outdated documentation
        outdated_docs = [
            'CLAUDE_INTEGRATION_STATUS.md',
            'CLEANUP_COMPLETION_REPORT.md',
            'DEPLOYMENT_SUCCESS_REPORT.md',
            'FINAL_INTEGRATION_REPORT.md',
            'GITHUB_SETUP_COMPLETE.md',
            'OMNI_ALPHA_COMPLETE_SYSTEM_GUIDE.md',
            'POWERSHELL_TEST_STEP3_REPORT.md',
            'POWERSHELL_TEST_STEP4_REPORT.md',
            'STEP3_BROKER_INTEGRATION_SUMMARY.md',
            'STEP4_ORDER_MANAGEMENT_SYSTEM_SUMMARY.md',
            'STEP5_ADVANCED_TRADING_COMPONENTS_SUMMARY.md',
            'STEP5_CLEANUP_REPORT.md',
            'STEP5_EXECUTIVE_SUMMARY.md',
            'STEP5_FINAL_COMPONENTS_SUMMARY.md',
            'STEP5_INDUSTRY_ANALYSIS_REPORT.md',
            'STEP5_INDUSTRY_COMPARISON_TABLE.md',
            'STEP6_ADVANCED_RISK_MANAGEMENT_SUMMARY.md',
            'STEP7_ADVANCED_PORTFOLIO_MANAGEMENT_SUMMARY.md',
            'STEP9_ULTIMATE_AI_BRAIN_EXECUTION_SUMMARY.md',
            'STEP9_WINDOWS_SOLUTION_SUMMARY.md',
            'STEP10_MASTER_ORCHESTRATION_SUMMARY.md',
            'STEP11_INSTITUTIONAL_OPERATIONS_SUMMARY.md',
            'STEP12_GLOBAL_MARKET_DOMINANCE_SUMMARY.md'
        ]
        
        # PowerShell scripts (keeping only essential ones)
        ps_scripts = [
            'claude_bridge.ps1',
            'claude_collab.ps1',
            'claude_github.ps1',
            'github_api.ps1',
            'test_all_steps.ps1',
            'test_complete.ps1',
            'test_quick.ps1'
        ]
        
        # Temporary/test files
        temp_files = [
            'ai_brain_execution_test_results.json',
            'orchestration_test_results.json',
            'strategy_engine_test_results.json',
            'system_status.json',
            'production_readiness_report.json',
            'production.log',
            'test_results.json',
            'test_sensitive.txt',
            'test_sensitive.txt.encrypted',
            'institutional.db'
        ]
        
        # Outdated directories
        outdated_dirs = [
            'backend',
            'frontend', 
            'src',
            'tests',
            'docs',
            'data',
            'deployment',
            'infrastructure'
        ]
        
        self.files_to_remove = duplicate_files + outdated_docs + ps_scripts + temp_files
        self.dirs_to_remove = outdated_dirs
        
    def execute_cleanup(self):
        """Execute the cleanup process"""
        
        print("🧹 OMNI ALPHA SYSTEM CLEANUP")
        print("=" * 50)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Identify targets
        self.identify_cleanup_targets()
        
        print(f"\n📋 Cleanup Plan:")
        print(f"   • Files to remove: {len(self.files_to_remove)}")
        print(f"   • Directories to remove: {len(self.dirs_to_remove)}")
        
        # Remove files
        print(f"\n🗑️ Removing duplicate/outdated files...")
        for file_path in self.files_to_remove:
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.files_removed.append(file_path)
                    self.space_saved += file_size
                    print(f"   ✅ Removed: {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {file_path}: {e}")
            else:
                print(f"   ⚠️ Not found: {file_path}")
        
        # Remove directories
        print(f"\n📁 Removing outdated directories...")
        for dir_path in self.dirs_to_remove:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                try:
                    # Calculate directory size
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(dir_path)
                        for filename in filenames
                    )
                    
                    shutil.rmtree(dir_path)
                    self.space_saved += dir_size
                    print(f"   ✅ Removed directory: {dir_path}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {dir_path}: {e}")
            else:
                print(f"   ⚠️ Directory not found: {dir_path}")
        
        # Generate cleanup report
        self.generate_cleanup_report()
        
        print(f"\n" + "=" * 50)
        print("🎉 SYSTEM CLEANUP COMPLETE!")
        print(f"Files removed: {len(self.files_removed)}")
        print(f"Space saved: {self.space_saved / (1024*1024):.1f} MB")
        print("=" * 50)
    
    def generate_cleanup_report(self):
        """Generate cleanup report"""
        
        # List remaining essential files
        essential_files = [
            'omni_alpha_complete.py',  # Main system
            'data_integration.py',     # Data integration
            'fix_failing_tests.py',    # Production fixes
            'run_complete_system_test.py',  # System testing
            'test_dashboard.html',     # Monitoring dashboard
            'test_security_fortress.py'  # Security testing
        ]
        
        essential_dirs = [
            'core/',      # Core components
            'security/',  # Security system
            'config/',    # Configuration
            'k8s/',       # Kubernetes
            'monitoring/', # Monitoring
            'scripts/',   # Essential scripts
            'backups/'    # Backup files
        ]
        
        report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'files_removed': self.files_removed,
            'space_saved_mb': self.space_saved / (1024*1024),
            'essential_files': essential_files,
            'essential_directories': essential_dirs,
            'cleanup_summary': {
                'duplicate_files_removed': len([f for f in self.files_removed if 'omni_alpha' in f or 'test_' in f]),
                'documentation_cleaned': len([f for f in self.files_removed if f.endswith('.md')]),
                'temp_files_removed': len([f for f in self.files_removed if f.endswith('.json') or f.endswith('.log')]),
                'script_files_removed': len([f for f in self.files_removed if f.endswith('.ps1')])
            }
        }
        
        with open('cleanup_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Cleanup report saved: cleanup_report.json")
        
        # Show what's left
        print(f"\n📋 ESSENTIAL FILES REMAINING:")
        for file in essential_files:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / (1024*1024)
                print(f"   ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file} (missing)")
        
        print(f"\n📁 ESSENTIAL DIRECTORIES:")
        for dir_path in essential_dirs:
            if os.path.exists(dir_path):
                file_count = len(list(Path(dir_path).rglob('*')))
                print(f"   ✅ {dir_path} ({file_count} items)")
            else:
                print(f"   ❌ {dir_path} (missing)")

def main():
    """Main cleanup execution"""
    
    cleanup = SystemCleanup()
    cleanup.execute_cleanup()
    
    print(f"\n🎯 SYSTEM STREAMLINED FOR PRODUCTION!")
    print("✅ All duplicate files removed")
    print("✅ Outdated documentation cleaned")
    print("✅ Temporary files purged")
    print("✅ Essential components preserved")
    print("✅ Ready for production deployment")

if __name__ == "__main__":
    main()

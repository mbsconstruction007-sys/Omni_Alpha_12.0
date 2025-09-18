"""
Final System Cleanup - Remove All Duplicate and Unwanted Files
Streamlines the system for production deployment
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class FinalSystemCleanup:
    """Comprehensive cleanup for production deployment"""
    
    def __init__(self):
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'files_deleted': [],
            'directories_deleted': [],
            'files_kept': [],
            'space_saved': 0
        }
        
        # Files to KEEP (essential for production)
        self.essential_files = {
            # Main trading systems
            'omni_alpha_enhanced_live.py',      # MAIN enhanced bot
            'omni_alpha_complete.py',           # Complete system (backup)
            
            # Setup and verification
            'QUICK_SETUP_GUIDE.md',             # Setup guide
            'verify_env.py',                    # Environment verification
            'get_chat_id.py',                   # Chat ID finder
            'requirements.txt',                 # Dependencies
            'alpaca_live_trading.env',          # Environment config
            
            # Dashboard and monitoring
            'dashboard.py',                     # Trading dashboard
            'verify_system.py',                 # System verification
            
            # Essential configs
            'README.md',                        # Documentation
            'LICENSE',                          # License
            '.gitignore',                       # Git ignore
            
            # Core modules (keep directory)
            'core/',
            'security/',
            
            # Startup scripts
            'start_live_trading.bat',           # Windows startup
            'start_live_trading.sh',            # Linux startup
        }
        
        # Files/directories to DELETE (duplicates, old versions, test files)
        self.files_to_delete = [
            # Old/duplicate trading bots
            'omni_alpha_live_trading.py',       # Old version
            'setup_live_trading.py',            # Old setup
            'data_integration.py',              # Old data system
            
            # Old guides and docs
            'ADVANCED_FEATURES_GUIDE.md',
            'COMPLETE_SYSTEM_GUIDE.md',
            'DEPLOYMENT_GUIDE.md',
            'QUICK_START_GUIDE.md',
            'CONTRIBUTING.md',
            
            # Test files
            'test_enhanced_fixes.py',
            'test_step*.py',
            'test_security_fortress.py',
            'test_dashboard.html',
            'run_complete_system_test.py',
            
            # Old config files
            'alpaca_keys.py',
            'config.env',
            'env.example',
            'telegram_config.py',
            'test_config.py',
            
            # Utility files (now integrated)
            'cleanup_system.py',
            'fix_failing_tests.py',
            'db_utils.py',
            'encoding_config.py',
            'import_utils.py',
            'memory_utils.py',
            'path_utils.py',
            'safe_math_utils.py',
            'unicode_utils.py',
            
            # Old test results
            'test_results_*.json',
            'comprehensive_test_results.json',
            'system_verification_results.json',
            'cleanup_report.json',
            
            # Old requirements
            'requirements_api.txt',
            
            # Log files (will be recreated)
            '*.log',
            
            # Word document
            'step 1 to 24 deep analiys.docx',
        ]
        
        # Directories to DELETE
        self.directories_to_delete = [
            '__pycache__',
            'backups',
            'scripts',      # Old deployment scripts
            'config',       # Old config directory
            'k8s',          # Kubernetes configs (not needed for basic setup)
            'monitoring',   # Grafana configs (advanced)
            'data',         # Empty data directory
            'logs',         # Empty logs directory  
            'models',       # Empty models directory
            'reports',      # Empty reports directory
        ]
    
    def cleanup_system(self):
        """Perform comprehensive system cleanup"""
        
        print("🧹 FINAL SYSTEM CLEANUP - PRODUCTION STREAMLINING")
        print("=" * 60)
        
        # Delete individual files
        print("\n📄 Removing duplicate and unwanted files...")
        self._delete_files()
        
        # Delete directories
        print("\n📁 Removing unwanted directories...")
        self._delete_directories()
        
        # Clean Docker files
        print("\n🐳 Removing Docker configurations...")
        self._clean_docker_files()
        
        # Generate final report
        print("\n📊 Generating cleanup report...")
        self._generate_report()
        
        print("\n✅ CLEANUP COMPLETE!")
        self._show_final_status()
    
    def _delete_files(self):
        """Delete unwanted files"""
        
        for pattern in self.files_to_delete:
            if '*' in pattern:
                # Handle wildcard patterns
                import glob
                for file_path in glob.glob(pattern):
                    self._delete_file(file_path)
            else:
                self._delete_file(pattern)
    
    def _delete_file(self, file_path):
        """Delete a single file"""
        
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                self.cleanup_report['files_deleted'].append(file_path)
                self.cleanup_report['space_saved'] += file_size
                print(f"   ❌ Deleted: {file_path}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {file_path}: {e}")
        else:
            print(f"   ℹ️ Not found: {file_path}")
    
    def _delete_directories(self):
        """Delete unwanted directories"""
        
        for dir_name in self.directories_to_delete:
            if os.path.exists(dir_name) and os.path.isdir(dir_name):
                try:
                    # Calculate directory size
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(dir_name)
                        for filename in filenames
                    )
                    
                    shutil.rmtree(dir_name)
                    self.cleanup_report['directories_deleted'].append(dir_name)
                    self.cleanup_report['space_saved'] += dir_size
                    print(f"   ❌ Deleted directory: {dir_name}")
                except Exception as e:
                    print(f"   ⚠️ Could not delete directory {dir_name}: {e}")
            else:
                print(f"   ℹ️ Directory not found: {dir_name}")
    
    def _clean_docker_files(self):
        """Remove Docker configuration files"""
        
        docker_files = [
            'docker-compose.yml',
            'docker-compose-ecosystem.yml', 
            'Dockerfile.production'
        ]
        
        for docker_file in docker_files:
            self._delete_file(docker_file)
    
    def _generate_report(self):
        """Generate cleanup report"""
        
        # List remaining essential files
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = os.path.join(root, file).replace('\\', '/')
                    if file_path.startswith('./'):
                        file_path = file_path[2:]
                    self.cleanup_report['files_kept'].append(file_path)
        
        # Save report
        with open('final_cleanup_report.json', 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
    
    def _show_final_status(self):
        """Show final cleanup status"""
        
        files_deleted = len(self.cleanup_report['files_deleted'])
        dirs_deleted = len(self.cleanup_report['directories_deleted'])
        files_kept = len(self.cleanup_report['files_kept'])
        space_saved = self.cleanup_report['space_saved'] / (1024 * 1024)  # MB
        
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"   Files deleted: {files_deleted}")
        print(f"   Directories deleted: {dirs_deleted}")
        print(f"   Files kept: {files_kept}")
        print(f"   Space saved: {space_saved:.2f} MB")
        
        print(f"\n✅ ESSENTIAL FILES KEPT:")
        essential_kept = [f for f in self.cleanup_report['files_kept'] 
                         if any(f.endswith(ef.rstrip('/')) or f.startswith(ef.rstrip('/')) 
                               for ef in self.essential_files)]
        
        for file in sorted(essential_kept)[:15]:  # Show first 15
            print(f"   ✅ {file}")
        
        if len(essential_kept) > 15:
            print(f"   ... and {len(essential_kept) - 15} more essential files")
        
        print(f"\n🚀 PRODUCTION-READY SYSTEM:")
        print(f"   ✅ Enhanced trading bot: omni_alpha_enhanced_live.py")
        print(f"   ✅ Setup guide: QUICK_SETUP_GUIDE.md")
        print(f"   ✅ Environment verification: verify_env.py")
        print(f"   ✅ Dashboard: dashboard.py")
        print(f"   ✅ Core modules: core/ directory")
        print(f"   ✅ Security system: security/ directory")
        
        print(f"\n📱 READY TO USE:")
        print(f"   1. python verify_env.py")
        print(f"   2. python omni_alpha_enhanced_live.py")
        print(f"   3. Send /start in Telegram")

def main():
    """Main cleanup execution"""
    
    print("⚠️ WARNING: This will delete many files!")
    print("Make sure you have committed important changes to Git.")
    
    response = input("\nProceed with cleanup? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        cleanup = FinalSystemCleanup()
        cleanup.cleanup_system()
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()

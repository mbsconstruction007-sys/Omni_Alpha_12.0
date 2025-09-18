"""
INFRASTRUCTURE CLEANUP TOOL
===========================
Clean up duplicate and unwanted files while preserving the new infrastructure
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

class InfrastructureCleanup:
    """Clean up old files and organize new infrastructure"""
    
    def __init__(self):
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'files_deleted': [],
            'files_moved': [],
            'directories_created': [],
            'errors': []
        }
    
    def identify_duplicate_files(self):
        """Identify duplicate and unwanted files"""
        
        # Files to delete (duplicates/old versions)
        files_to_delete = [
            # Old git artifacts
            'et --hard 59df084',
            'et --hard HEAD~1',
            
            # Old cleanup files
            'final_cleanup.py',
            'final_cleanup_report.json',
            
            # Duplicate analysis files (keep the enhanced versions)
            'STEP_1_ANALYSIS_REPORT.md',  # We have STEP_1_ENHANCED_IMPLEMENTATION.md
            'STEP_2_ANALYSIS_REPORT.md',  # We have STEP_2_ENHANCED_COMPARISON.md
            
            # Old individual test files (we'll create comprehensive tests)
            'test_step1_core_infrastructure.py',
            'test_step2_data_collection.py',
            
            # Old infrastructure files (replaced by new structure)
            'step_1_core_infrastructure.py',  # Replaced by config/ and infrastructure/
            'step_2_data_collection.py',      # Replaced by data_collection/
            
            # Migration files (no longer needed after migration)
            'migrate_step2_enhanced.py',
            
            # Old verification files (replaced by orchestrator)
            'verify_env.py',
            'verify_system.py',
            
            # Old setup files (replaced by orchestrator)
            'get_chat_id.py',
            
            # Old dashboard (will be replaced by proper monitoring)
            'dashboard.py'
        ]
        
        return files_to_delete
    
    def identify_directories_to_clean(self):
        """Identify directories that need cleaning"""
        
        # Directories with old cache/temp files
        dirs_to_clean = [
            'core/__pycache__',
            'security/__pycache__',
            '__pycache__'
        ]
        
        return dirs_to_clean
    
    def identify_files_to_keep(self):
        """Identify critical files to preserve"""
        
        critical_files = [
            # Enhanced infrastructure
            'orchestrator.py',
            'config/',
            'infrastructure/',
            'risk_management/',
            'data_collection/',
            
            # Environment and configuration
            'alpaca_live_trading.env',
            'step1_environment_template.env',
            'requirements.txt',
            
            # Production deployment
            'docker-compose.yml',
            'docker-compose-ecosystem.yml',
            'Dockerfile.production',
            'k8s/',
            'monitoring/',
            
            # Security system (keep as is)
            'security/',
            
            # Enhanced bot (main application)
            'omni_alpha_enhanced_live.py',
            'omni_alpha_complete.py',
            
            # Core system components (keep legacy for now)
            'core/',
            
            # Documentation
            'README.md',
            'LICENSE',
            'CRITICAL_FILES_EXPLANATION.md',
            'STEP_1_ENHANCED_IMPLEMENTATION.md',
            'STEP_2_ENHANCED_COMPARISON.md',
            'QUICK_SETUP_GUIDE.md',
            
            # Scripts
            'scripts/',
            'install_step1_enhanced.sh',
            'start_live_trading.bat',
            'start_live_trading.sh',
            
            # Data and logs
            'logs/',
            '*.db',
            '*.log'
        ]
        
        return critical_files
    
    def perform_cleanup(self):
        """Perform the cleanup operation"""
        print("🧹 OMNI ALPHA 5.0 - INFRASTRUCTURE CLEANUP")
        print("=" * 50)
        
        # Get files to delete
        files_to_delete = self.identify_duplicate_files()
        dirs_to_clean = self.identify_directories_to_clean()
        
        # Delete duplicate files
        print(f"\n🗑️ Deleting {len(files_to_delete)} duplicate/unwanted files...")
        for file_path in files_to_delete:
            try:
                if Path(file_path).exists():
                    if Path(file_path).is_file():
                        os.remove(file_path)
                        self.cleanup_report['files_deleted'].append(file_path)
                        print(f"   ✅ Deleted: {file_path}")
                    elif Path(file_path).is_dir():
                        shutil.rmtree(file_path)
                        self.cleanup_report['files_deleted'].append(file_path)
                        print(f"   ✅ Deleted directory: {file_path}")
            except Exception as e:
                error_msg = f"Failed to delete {file_path}: {e}"
                self.cleanup_report['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # Clean cache directories
        print(f"\n🧹 Cleaning {len(dirs_to_clean)} cache directories...")
        for dir_path in dirs_to_clean:
            try:
                if Path(dir_path).exists():
                    shutil.rmtree(dir_path)
                    self.cleanup_report['files_deleted'].append(dir_path)
                    print(f"   ✅ Cleaned: {dir_path}")
            except Exception as e:
                error_msg = f"Failed to clean {dir_path}: {e}"
                self.cleanup_report['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # Create __init__.py files for new modules
        init_files = [
            'data_collection/providers/__init__.py',
            'data_collection/streams/__init__.py', 
            'data_collection/orderbook/__init__.py',
            'data_collection/storage/__init__.py',
            'data_collection/validation/__init__.py',
            'data_collection/news_sentiment/__init__.py'
        ]
        
        print(f"\n📁 Creating {len(init_files)} module init files...")
        for init_file in init_files:
            try:
                Path(init_file).parent.mkdir(parents=True, exist_ok=True)
                with open(init_file, 'w') as f:
                    f.write('"""Module init file"""\n')
                self.cleanup_report['files_moved'].append(init_file)
                print(f"   ✅ Created: {init_file}")
            except Exception as e:
                error_msg = f"Failed to create {init_file}: {e}"
                self.cleanup_report['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # Save cleanup report
        with open('infrastructure_cleanup_report.json', 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"   Files deleted: {len(self.cleanup_report['files_deleted'])}")
        print(f"   Files created: {len(self.cleanup_report['files_moved'])}")
        print(f"   Errors: {len(self.cleanup_report['errors'])}")
        
        if self.cleanup_report['errors']:
            print(f"\n⚠️ ERRORS:")
            for error in self.cleanup_report['errors']:
                print(f"   - {error}")
        
        print(f"\n✅ Infrastructure cleanup completed!")
        print(f"📄 Report saved to: infrastructure_cleanup_report.json")
        
        return len(self.cleanup_report['errors']) == 0

def main():
    """Main cleanup execution"""
    cleanup = InfrastructureCleanup()
    success = cleanup.perform_cleanup()
    
    if success:
        print("\n🎉 Cleanup successful! Infrastructure is now organized.")
    else:
        print("\n⚠️ Cleanup completed with some errors. Check the report.")

if __name__ == "__main__":
    main()

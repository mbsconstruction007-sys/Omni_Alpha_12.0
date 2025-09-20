#!/usr/bin/env python3
"""
OMNI ALPHA 5.0 - AUTO GIT UPDATE SYSTEM
=======================================
Automatically commit and push changes to Git repository
"""

import os
import subprocess
import time
import threading
from datetime import datetime
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_git_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoGitUpdater:
    """Automatic Git update system"""
    
    def __init__(self, repo_path='.', interval_seconds=300):  # 5 minutes default
        self.repo_path = Path(repo_path)
        self.interval_seconds = interval_seconds
        self.running = False
        self.last_commit_hash = None
        
        # Files to ignore for auto-commit
        self.ignore_patterns = [
            '*.log',
            '*.tmp',
            '__pycache__/',
            '.pytest_cache/',
            '*.pyc',
            '.DS_Store',
            'Thumbs.db',
            'auto_git_update.py'  # Don't commit this file itself
        ]
        
        logger.info(f"Auto Git Updater initialized for {self.repo_path}")
        logger.info(f"Update interval: {self.interval_seconds} seconds")
    
    def get_git_status(self):
        """Get current Git status"""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git status error: {e}")
            return ""
    
    def get_current_commit_hash(self):
        """Get current commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git rev-parse error: {e}")
            return None
    
    def has_changes(self):
        """Check if there are uncommitted changes"""
        status = self.get_git_status()
        return len(status) > 0
    
    def get_changed_files(self):
        """Get list of changed files"""
        status = self.get_git_status()
        if not status:
            return []
        
        changed_files = []
        for line in status.split('\n'):
            if line.strip():
                # Parse git status output (e.g., "M  file.py", "?? newfile.py")
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_path = ' '.join(parts[1:])
                    changed_files.append(file_path)
        
        return changed_files
    
    def should_ignore_file(self, file_path):
        """Check if file should be ignored"""
        import fnmatch
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False
    
    def generate_commit_message(self, changed_files):
        """Generate intelligent commit message based on changed files"""
        if not changed_files:
            return "🔄 Auto-update: General improvements"
        
        # Categorize changes
        categories = {
            'strategies': [],
            'risk': [],
            'execution': [],
            'config': [],
            'infrastructure': [],
            'security': [],
            'analysis': [],
            'docs': [],
            'tests': [],
            'other': []
        }
        
        for file in changed_files:
            if self.should_ignore_file(file):
                continue
                
            file_lower = file.lower()
            
            if 'strateg' in file_lower:
                categories['strategies'].append(file)
            elif 'risk' in file_lower:
                categories['risk'].append(file)
            elif 'execution' in file_lower or 'order' in file_lower:
                categories['execution'].append(file)
            elif 'config' in file_lower or 'setting' in file_lower:
                categories['config'].append(file)
            elif 'infrastructure' in file_lower or 'monitor' in file_lower or 'circuit' in file_lower:
                categories['infrastructure'].append(file)
            elif 'security' in file_lower:
                categories['security'].append(file)
            elif 'analysis' in file_lower or 'report' in file_lower:
                categories['analysis'].append(file)
            elif file_lower.endswith('.md') or 'readme' in file_lower or 'doc' in file_lower:
                categories['docs'].append(file)
            elif 'test' in file_lower:
                categories['tests'].append(file)
            else:
                categories['other'].append(file)
        
        # Build commit message
        message_parts = ["🔄 Auto-update: "]
        updates = []
        
        if categories['strategies']:
            updates.append(f"Trading Strategies ({len(categories['strategies'])} files)")
        if categories['risk']:
            updates.append(f"Risk Management ({len(categories['risk'])} files)")
        if categories['execution']:
            updates.append(f"Order Execution ({len(categories['execution'])} files)")
        if categories['config']:
            updates.append(f"Configuration ({len(categories['config'])} files)")
        if categories['infrastructure']:
            updates.append(f"Infrastructure ({len(categories['infrastructure'])} files)")
        if categories['security']:
            updates.append(f"Security ({len(categories['security'])} files)")
        if categories['analysis']:
            updates.append(f"Analysis & Reports ({len(categories['analysis'])} files)")
        if categories['docs']:
            updates.append(f"Documentation ({len(categories['docs'])} files)")
        if categories['tests']:
            updates.append(f"Tests ({len(categories['tests'])} files)")
        if categories['other']:
            updates.append(f"Other ({len(categories['other'])} files)")
        
        if updates:
            message_parts.append(", ".join(updates))
        else:
            message_parts.append("General improvements")
        
        message_parts.append(f"\n\n📊 Total files updated: {len(changed_files)}")
        message_parts.append(f"🕒 Auto-committed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "".join(message_parts)
    
    def commit_changes(self):
        """Commit all changes"""
        try:
            # Add all changes
            subprocess.run(
                ['git', 'add', '.'],
                cwd=self.repo_path,
                check=True
            )
            
            # Get changed files for commit message
            changed_files = self.get_changed_files()
            commit_message = self.generate_commit_message(changed_files)
            
            # Commit changes
            subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=self.repo_path,
                check=True
            )
            
            logger.info(f"✅ Committed changes: {len(changed_files)} files")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git commit error: {e}")
            return False
    
    def push_changes(self):
        """Push changes to remote repository"""
        try:
            subprocess.run(
                ['git', 'push', 'origin', 'main'],
                cwd=self.repo_path,
                check=True
            )
            logger.info("✅ Successfully pushed to remote repository")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git push error: {e}")
            return False
    
    def update_git(self):
        """Perform full Git update (add, commit, push)"""
        if not self.has_changes():
            logger.debug("No changes detected, skipping update")
            return False
        
        changed_files = self.get_changed_files()
        logger.info(f"🔄 Auto-updating Git repository with {len(changed_files)} changed files")
        
        # Commit changes
        if self.commit_changes():
            # Push to remote
            if self.push_changes():
                self.last_commit_hash = self.get_current_commit_hash()
                logger.info(f"🎉 Auto-update successful! Commit: {self.last_commit_hash[:8]}")
                return True
            else:
                logger.warning("⚠️ Commit successful but push failed")
                return False
        else:
            logger.error("❌ Auto-update failed during commit")
            return False
    
    def start_monitoring(self):
        """Start automatic monitoring and updating"""
        self.running = True
        logger.info("🚀 Starting auto Git update monitoring...")
        
        def monitor_loop():
            while self.running:
                try:
                    self.update_git()
                    time.sleep(self.interval_seconds)
                except Exception as e:
                    logger.error(f"❌ Monitor loop error: {e}")
                    time.sleep(30)  # Wait 30 seconds before retrying
        
        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"✅ Auto Git update monitoring started (interval: {self.interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop automatic monitoring"""
        self.running = False
        logger.info("🛑 Stopping auto Git update monitoring...")
    
    def manual_update(self):
        """Manually trigger an update"""
        logger.info("🔄 Manual Git update triggered...")
        return self.update_git()
    
    def get_status(self):
        """Get current status"""
        return {
            'running': self.running,
            'interval_seconds': self.interval_seconds,
            'repo_path': str(self.repo_path),
            'last_commit_hash': self.last_commit_hash,
            'has_changes': self.has_changes(),
            'changed_files_count': len(self.get_changed_files()) if self.has_changes() else 0
        }

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Git Update System for Omni Alpha 5.0')
    parser.add_argument('--interval', type=int, default=300, help='Update interval in seconds (default: 300)')
    parser.add_argument('--manual', action='store_true', help='Run manual update only')
    parser.add_argument('--status', action='store_true', help='Show current status')
    
    args = parser.parse_args()
    
    updater = AutoGitUpdater(interval_seconds=args.interval)
    
    if args.status:
        status = updater.get_status()
        print(f"\n📊 AUTO GIT UPDATE STATUS:")
        print(f"   Running: {status['running']}")
        print(f"   Interval: {status['interval_seconds']} seconds")
        print(f"   Repository: {status['repo_path']}")
        print(f"   Has Changes: {status['has_changes']}")
        print(f"   Changed Files: {status['changed_files_count']}")
        print(f"   Last Commit: {status['last_commit_hash']}")
        return
    
    if args.manual:
        print("🔄 Running manual Git update...")
        success = updater.manual_update()
        print(f"✅ Manual update {'successful' if success else 'failed'}")
        return
    
    # Start continuous monitoring
    try:
        updater.start_monitoring()
        
        print(f"🚀 Auto Git Update System Started!")
        print(f"   Update Interval: {args.interval} seconds")
        print(f"   Repository: {updater.repo_path}")
        print(f"   Press Ctrl+C to stop")
        
        # Keep main thread alive
        while updater.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Auto Git Update System...")
        updater.stop_monitoring()
        print("✅ Auto Git Update System stopped")

if __name__ == "__main__":
    main()

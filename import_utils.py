
# import_utils.py
import importlib
import logging
from typing import Any, Optional, Dict
import warnings

logger = logging.getLogger(__name__)

class SafeImporter:
    '''Safely import optional dependencies with fallbacks'''
    
    def __init__(self):
        self.available_packages = {}
        self.fallback_implementations = {}
    
    def safe_import(self, package_name: str, fallback: Any = None, 
                   silent: bool = True) -> Any:
        '''Safely import package with optional fallback'''
        
        if package_name in self.available_packages:
            return self.available_packages[package_name]
        
        try:
            module = importlib.import_module(package_name)
            self.available_packages[package_name] = module
            return module
            
        except ImportError as e:
            if not silent:
                logger.warning(f"Failed to import {package_name}: {e}")
            
            if fallback is not None:
                self.available_packages[package_name] = fallback
                return fallback
            
            # Return a mock object that raises informative errors
            return self._create_mock_module(package_name)
    
    def _create_mock_module(self, package_name: str):
        '''Create mock module that provides helpful error messages'''
        
        class MockModule:
            def __init__(self, name):
                self._name = name
            
            def __getattr__(self, item):
                raise ImportError(
                    f"Package '{self._name}' is not installed. "
                    f"Install it with: pip install {self._name}"
                )
            
            def __call__(self, *args, **kwargs):
                raise ImportError(
                    f"Package '{self._name}' is not installed. "
                    f"Install it with: pip install {self._name}"
                )
        
        return MockModule(package_name)
    
    def check_package_availability(self, packages: list) -> Dict[str, bool]:
        '''Check availability of multiple packages'''
        
        availability = {}
        for package in packages:
            try:
                importlib.import_module(package)
                availability[package] = True
            except ImportError:
                availability[package] = False
        
        return availability

# Global importer instance
safe_importer = SafeImporter()

# Common optional imports with fallbacks
def import_plotly():
    '''Import plotly with fallback'''
    return safe_importer.safe_import('plotly')

def import_optuna():
    '''Import optuna with fallback'''
    return safe_importer.safe_import('optuna')

def import_h2o():
    '''Import h2o with fallback'''
    return safe_importer.safe_import('h2o')

def import_mlflow():
    '''Import mlflow with fallback'''
    return safe_importer.safe_import('mlflow')

def import_redis():
    '''Import redis with fallback'''
    return safe_importer.safe_import('redis')

def import_clickhouse_driver():
    '''Import clickhouse-driver with fallback'''
    return safe_importer.safe_import('clickhouse_driver')

# Check critical vs optional dependencies
CRITICAL_PACKAGES = [
    'numpy', 'pandas', 'sqlalchemy', 'fastapi', 'python-telegram-bot',
    'alpaca-trade-api', 'scikit-learn', 'asyncio'
]

OPTIONAL_PACKAGES = [
    'plotly', 'optuna', 'h2o', 'mlflow', 'redis', 'clickhouse-driver',
    'influxdb-client', 'prometheus-client'
]

def check_dependencies():
    '''Check all dependencies and provide report'''
    
    critical_status = safe_importer.check_package_availability(CRITICAL_PACKAGES)
    optional_status = safe_importer.check_package_availability(OPTIONAL_PACKAGES)
    
    report = {
        'critical': critical_status,
        'optional': optional_status,
        'critical_missing': [pkg for pkg, available in critical_status.items() if not available],
        'optional_missing': [pkg for pkg, available in optional_status.items() if not available]
    }
    
    return report

def install_missing_packages(packages: list, silent: bool = True):
    '''Attempt to install missing packages'''
    
    import subprocess
    import sys
    
    installed = []
    failed = []
    
    for package in packages:
        try:
            if not silent:
                print(f"Installing {package}...")
            
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package, '--quiet'
            ])
            installed.append(package)
            
        except subprocess.CalledProcessError:
            failed.append(package)
            if not silent:
                print(f"Failed to install {package}")
    
    return {'installed': installed, 'failed': failed}

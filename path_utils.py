
# path_utils.py
from pathlib import Path
import os
import sys
from typing import Union

class PathManager:
    '''Cross-platform path management utilities'''
    
    @staticmethod
    def safe_path(path_str: Union[str, Path]) -> Path:
        '''Convert string path to Path object safely'''
        return Path(path_str)
    
    @staticmethod
    def join_paths(*args) -> Path:
        '''Join multiple path components safely'''
        if not args:
            return Path.cwd()
        
        base_path = Path(args[0])
        for part in args[1:]:
            base_path = base_path / part
        
        return base_path
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        '''Ensure directory exists, create if necessary'''
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def safe_file_path(filename: str, directory: Union[str, Path] = None) -> Path:
        '''Create safe file path with optional directory'''
        if directory:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path / filename
        else:
            return Path(filename)
    
    @staticmethod
    def get_project_root() -> Path:
        '''Get project root directory'''
        current = Path.cwd()
        
        # Look for common project indicators
        indicators = ['.git', 'requirements.txt', 'setup.py', 'pyproject.toml']
        
        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        return current
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> str:
        '''Normalize path for current OS'''
        path_obj = Path(path)
        return str(path_obj.resolve())
    
    @staticmethod
    def safe_open(filepath: Union[str, Path], mode: str = 'r', 
                  encoding: str = 'utf-8', **kwargs):
        '''Safely open file with proper encoding'''
        path_obj = Path(filepath)
        
        # Ensure parent directory exists for write modes
        if 'w' in mode or 'a' in mode:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        return open(path_obj, mode=mode, encoding=encoding, **kwargs)

# Global path manager instance
path_manager = PathManager()

# Convenience functions
safe_path = path_manager.safe_path
join_paths = path_manager.join_paths
ensure_directory = path_manager.ensure_directory
safe_file_path = path_manager.safe_file_path
get_project_root = path_manager.get_project_root
normalize_path = path_manager.normalize_path
safe_open = path_manager.safe_open

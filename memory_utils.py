
# memory_utils.py
import gc
import psutil
import numpy as np
from typing import Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    '''Memory management utilities for large computations'''
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
    def get_memory_usage(self) -> float:
        '''Get current memory usage in GB'''
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024 * 1024)
    
    def check_memory_limit(self) -> bool:
        '''Check if memory usage is within limits'''
        current_usage = self.get_memory_usage()
        return current_usage < (self.max_memory_bytes / (1024 * 1024 * 1024))
    
    def force_cleanup(self):
        '''Force garbage collection and cleanup'''
        gc.collect()
        
    def chunked_operation(self, data: np.ndarray, chunk_size: int = 1000, 
                         operation: callable = None) -> List[Any]:
        '''Process large arrays in chunks to prevent memory overflow'''
        
        if operation is None:
            operation = lambda x: x
            
        results = []
        n_items = len(data)
        
        for i in range(0, n_items, chunk_size):
            # Check memory before processing chunk
            if not self.check_memory_limit():
                logger.warning("Memory limit reached, forcing cleanup")
                self.force_cleanup()
            
            chunk = data[i:i + chunk_size]
            chunk_result = operation(chunk)
            results.append(chunk_result)
            
            # Cleanup after each chunk
            del chunk
            
        return results
    
    def safe_matrix_multiply(self, A: np.ndarray, B: np.ndarray, 
                           chunk_size: int = 500) -> np.ndarray:
        '''Safely multiply large matrices using chunked computation'''
        
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        # If matrices are small enough, use standard multiplication
        if A.shape[0] * B.shape[1] < chunk_size * chunk_size:
            return np.dot(A, B)
        
        # Use chunked multiplication for large matrices
        result_chunks = []
        
        for i in range(0, A.shape[0], chunk_size):
            row_chunk = A[i:i + chunk_size]
            chunk_result = np.dot(row_chunk, B)
            result_chunks.append(chunk_result)
            
            # Memory cleanup
            del row_chunk
            if not self.check_memory_limit():
                self.force_cleanup()
        
        return np.vstack(result_chunks)
    
    def optimize_portfolio_chunked(self, returns: np.ndarray, 
                                 chunk_size: int = 100) -> dict:
        '''Optimize portfolio using chunked processing for large datasets'''
        
        n_assets = returns.shape[1]
        
        if n_assets <= chunk_size:
            # Standard optimization for small portfolios
            return self._standard_optimization(returns)
        
        # Chunked optimization for large portfolios
        logger.info(f"Using chunked optimization for {n_assets} assets")
        
        # Process returns in chunks
        chunk_results = []
        for i in range(0, n_assets, chunk_size):
            chunk_returns = returns[:, i:i + chunk_size]
            chunk_result = self._optimize_chunk(chunk_returns)
            chunk_results.append(chunk_result)
            
            # Memory cleanup
            del chunk_returns
            self.force_cleanup()
        
        # Combine results
        return self._combine_chunk_results(chunk_results)
    
    def _standard_optimization(self, returns: np.ndarray) -> dict:
        '''Standard portfolio optimization'''
        try:
            cov_matrix = np.cov(returns.T)
            mean_returns = np.mean(returns, axis=0)
            
            # Simple equal-weight portfolio as fallback
            n_assets = len(mean_returns)
            weights = np.ones(n_assets) / n_assets
            
            return {
                'weights': weights,
                'expected_return': np.dot(weights, mean_returns),
                'volatility': np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))),
                'method': 'equal_weight'
            }
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {
                'weights': None,
                'expected_return': 0,
                'volatility': 0,
                'method': 'error',
                'error': str(e)
            }
    
    def _optimize_chunk(self, chunk_returns: np.ndarray) -> dict:
        '''Optimize a chunk of assets'''
        return self._standard_optimization(chunk_returns)
    
    def _combine_chunk_results(self, chunk_results: List[dict]) -> dict:
        '''Combine results from multiple chunks'''
        
        all_weights = []
        total_expected_return = 0
        total_volatility = 0
        
        for result in chunk_results:
            if result['weights'] is not None:
                all_weights.extend(result['weights'])
                total_expected_return += result['expected_return']
                total_volatility += result['volatility']
        
        # Normalize weights
        if all_weights:
            all_weights = np.array(all_weights)
            all_weights = all_weights / np.sum(all_weights)
        
        return {
            'weights': all_weights,
            'expected_return': total_expected_return / len(chunk_results),
            'volatility': total_volatility / len(chunk_results),
            'method': 'chunked_optimization'
        }

# Global memory manager instance
memory_manager = MemoryManager()

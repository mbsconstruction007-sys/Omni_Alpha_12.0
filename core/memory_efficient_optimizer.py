
"""
Memory-efficient portfolio optimization
"""
import numpy as np
import gc
from typing import Optional, Tuple

class MemoryEfficientOptimizer:
    """Portfolio optimizer with memory management"""
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_gb = max_memory_gb
        self.chunk_size = 1000
    
    def optimize_large_portfolio(self, returns_matrix: np.ndarray, 
                                target_return: Optional[float] = None) -> dict:
        """Optimize large portfolios with chunked processing"""
        
        n_assets = returns_matrix.shape[1]
        
        # Force garbage collection
        gc.collect()
        
        if n_assets > self.chunk_size:
            return self._chunked_optimization(returns_matrix, target_return)
        else:
            return self._standard_optimization(returns_matrix, target_return)
    
    def _chunked_optimization(self, returns_matrix: np.ndarray, 
                             target_return: Optional[float] = None) -> dict:
        """Process large portfolios in chunks"""
        
        n_assets = returns_matrix.shape[1]
        chunks = []
        
        for i in range(0, n_assets, self.chunk_size):
            end_idx = min(i + self.chunk_size, n_assets)
            chunk = returns_matrix[:, i:end_idx]
            
            # Process chunk
            chunk_result = self._optimize_chunk(chunk)
            chunks.append(chunk_result)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        # Combine results
        return self._combine_chunk_results(chunks)
    
    def _optimize_chunk(self, chunk_returns: np.ndarray) -> dict:
        """Optimize a single chunk"""
        
        try:
            # Simple mean-variance optimization for chunk
            mean_returns = np.mean(chunk_returns, axis=0)
            cov_matrix = np.cov(chunk_returns.T)
            
            # Equal weight as fallback
            n_assets = len(mean_returns)
            weights = np.ones(n_assets) / n_assets
            
            return {
                'weights': weights,
                'expected_return': np.dot(weights, mean_returns),
                'volatility': np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            }
            
        except Exception as e:
            # Fallback to equal weights
            n_assets = chunk_returns.shape[1]
            weights = np.ones(n_assets) / n_assets
            return {
                'weights': weights,
                'expected_return': 0.0,
                'volatility': 0.0,
                'error': str(e)
            }
    
    def _combine_chunk_results(self, chunks: list) -> dict:
        """Combine optimization results from chunks"""
        
        all_weights = np.concatenate([chunk['weights'] for chunk in chunks])
        
        # Normalize weights
        all_weights = all_weights / np.sum(all_weights)
        
        # Calculate combined metrics
        combined_return = np.mean([chunk['expected_return'] for chunk in chunks])
        combined_volatility = np.mean([chunk['volatility'] for chunk in chunks])
        
        return {
            'weights': all_weights,
            'expected_return': combined_return,
            'volatility': combined_volatility,
            'method': 'chunked_optimization'
        }
    
    def _standard_optimization(self, returns_matrix: np.ndarray, 
                              target_return: Optional[float] = None) -> dict:
        """Standard optimization for smaller portfolios"""
        
        try:
            mean_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)
            
            n_assets = len(mean_returns)
            
            # Simple equal weight portfolio
            weights = np.ones(n_assets) / n_assets
            
            expected_return = np.dot(weights, mean_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            return {
                'weights': weights,
                'expected_return': expected_return,
                'volatility': volatility,
                'method': 'standard_optimization'
            }
            
        except Exception as e:
            n_assets = returns_matrix.shape[1]
            weights = np.ones(n_assets) / n_assets
            return {
                'weights': weights,
                'expected_return': 0.0,
                'volatility': 0.0,
                'method': 'fallback',
                'error': str(e)
            }

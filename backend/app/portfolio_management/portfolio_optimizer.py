"""
ADVANCED PORTFOLIO OPTIMIZATION
Multiple optimization methods for different market conditions
This is what billion-dollar funds use
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    State-of-the-art portfolio optimization
    Implements multiple Nobel Prize-winning theories
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = 0.02  # 2% annual
        
    async def optimize(self, 
                      universe: List[str],
                      signals: Dict,
                      constraints: Dict,
                      method: str = "ensemble") -> Dict:
        """Master optimization function"""
        
        if method == "ensemble":
            return await self._ensemble_optimize(universe, signals, constraints)
        elif method == "hierarchical_risk_parity":
            return await self.hierarchical_risk_parity(universe)
        elif method == "black_litterman":
            return await self.black_litterman(universe, signals)
        elif method == "risk_parity":
            return await self.risk_parity(universe)
        else:
            return await self.mean_variance(universe, constraints)
    
    async def mean_variance(self, 
                           universe: List[str],
                           constraints: Dict) -> Dict:
        """
        Markowitz Mean-Variance Optimization
        The foundation of modern portfolio theory
        """
        logger.info("Running mean-variance optimization")
        
        # Get returns and covariance
        returns = await self._get_expected_returns(universe)
        cov_matrix = await self._get_covariance_matrix(universe)
        
        n_assets = len(universe)
        
        # Optimization objective: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds for each weight
        bounds = tuple(
            (constraints.get("min_position_weight", 0), 
             constraints.get("max_position_weight", 1))
            for _ in range(n_assets)
        )
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            return {universe[i]: weights[i] for i in range(n_assets)}
        else:
            logger.warning("Optimization failed, using equal weights")
            return {symbol: 1/n_assets for symbol in universe}
    
    async def black_litterman(self, 
                             universe: List[str],
                             views: Dict) -> Dict:
        """
        Black-Litterman Model
        Combines market equilibrium with investor views
        Used by Goldman Sachs and other institutions
        """
        logger.info("Running Black-Litterman optimization")
        
        # Get market data
        market_caps = await self._get_market_caps(universe)
        cov_matrix = await self._get_covariance_matrix(universe)
        
        # Calculate equilibrium weights (market cap weighted)
        total_market_cap = sum(market_caps.values())
        equilibrium_weights = {
            symbol: cap / total_market_cap 
            for symbol, cap in market_caps.items()
        }
        
        # Risk aversion parameter (typically 2.5)
        risk_aversion = 2.5
        
        # Calculate equilibrium returns
        eq_weights_array = np.array(list(equilibrium_weights.values()))
        equilibrium_returns = risk_aversion * np.dot(cov_matrix, eq_weights_array)
        
        # Incorporate views if provided
        if views:
            # Build P matrix (which assets views are about)
            # Build Q vector (the views themselves)
            # Build Omega (confidence in views)
            
            tau = 0.05  # Scalar representing uncertainty in equilibrium
            
            # Posterior estimates (simplified)
            # In production, this would be more complex
            posterior_returns = equilibrium_returns  # Placeholder
            
            # Optimize with posterior returns
            return await self._optimize_with_returns(universe, posterior_returns, cov_matrix)
        else:
            # No views, return market weights
            return equilibrium_weights
    
    async def risk_parity(self, universe: List[str]) -> Dict:
        """
        Risk Parity Portfolio
        Equal risk contribution from each asset
        Pioneered by Ray Dalio's Bridgewater
        """
        logger.info("Running risk parity optimization")
        
        cov_matrix = await self._get_covariance_matrix(universe)
        n_assets = len(universe)
        
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            """Minimize difference in risk contributions"""
            contrib = risk_contribution(weights, cov_matrix)
            # We want equal risk contribution
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds: all weights positive
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            return {universe[i]: weights[i] for i in range(n_assets)}
        else:
            logger.warning("Risk parity optimization failed")
            return {symbol: 1/n_assets for symbol in universe}
    
    async def hierarchical_risk_parity(self, universe: List[str]) -> Dict:
        """
        Hierarchical Risk Parity (HRP)
        Machine learning meets portfolio theory
        Developed by Marcos Lopez de Prado
        """
        logger.info("Running hierarchical risk parity optimization")
        
        # Get correlation and covariance
        returns_df = await self._get_returns_dataframe(universe)
        cov_matrix = returns_df.cov()
        corr_matrix = returns_df.corr()
        
        # Step 1: Tree clustering
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        linkage_matrix = linkage(distance_matrix, method='single')
        
        # Step 2: Quasi-diagonalization
        sorted_indices = self._get_quasi_diag(linkage_matrix, n_assets=len(universe))
        sorted_symbols = [universe[i] for i in sorted_indices]
        
        # Step 3: Recursive bisection
        weights = self._recursive_bisection(
            cov_matrix.values,
            sorted_indices
        )
        
        # Map weights back to symbols
        weight_dict = {}
        for i, idx in enumerate(sorted_indices):
            weight_dict[universe[idx]] = weights[i]
        
        return weight_dict
    
    def _get_quasi_diag(self, link_matrix, n_assets):
        """Quasi-diagonalization for HRP"""
        link_matrix = link_matrix.astype(int)
        sorted_indices = []
        
        def _recurse(index, level=0):
            if index < n_assets:
                sorted_indices.append(index)
            else:
                left = int(link_matrix[index - n_assets, 0])
                right = int(link_matrix[index - n_assets, 1])
                _recurse(left, level + 1)
                _recurse(right, level + 1)
        
        _recurse(2 * n_assets - 2)
        return sorted_indices
    
    def _recursive_bisection(self, cov_matrix, sorted_indices):
        """Recursive bisection for HRP weight allocation"""
        weights = np.ones(len(sorted_indices))
        clusters = [sorted_indices]
        
        while len(clusters) > 0:
            clusters_new = []
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster
                    mid = len(cluster) // 2
                    left_cluster = cluster[:mid]
                    right_cluster = cluster[mid:]
                    
                    # Calculate variance for each sub-cluster
                    left_var = self._get_cluster_variance(cov_matrix, left_cluster)
                    right_var = self._get_cluster_variance(cov_matrix, right_cluster)
                    
                    # Allocate weights inversely proportional to variance
                    total_var = left_var + right_var
                    if total_var > 0:
                        left_weight = right_var / total_var
                        right_weight = left_var / total_var
                    else:
                        left_weight = right_weight = 0.5
                    
                    # Update weights
                    for i in left_cluster:
                        weights[sorted_indices.index(i)] *= left_weight
                    for i in right_cluster:
                        weights[sorted_indices.index(i)] *= right_weight
                    
                    clusters_new.extend([left_cluster, right_cluster])
            
            clusters = clusters_new
        
        # Normalize weights
        return weights / weights.sum()
    
    def _get_cluster_variance(self, cov_matrix, cluster_indices):
        """Calculate variance of a cluster"""
        if len(cluster_indices) == 1:
            return cov_matrix[cluster_indices[0], cluster_indices[0]]
        
        cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
        weights = np.ones(len(cluster_indices)) / len(cluster_indices)
        
        return np.dot(weights, np.dot(cluster_cov, weights))
    
    async def _ensemble_optimize(self, 
                                universe: List[str],
                                signals: Dict,
                                constraints: Dict) -> Dict:
        """
        Ensemble of multiple optimization methods
        The best of all worlds
        """
        logger.info("Running ensemble optimization")
        
        # Run multiple optimizers
        methods = {
            "mean_variance": 0.25,
            "black_litterman": 0.30,
            "risk_parity": 0.25,
            "hierarchical_risk_parity": 0.20
        }
        
        all_weights = {}
        
        for method, weight in methods.items():
            if method == "mean_variance":
                result = await self.mean_variance(universe, constraints)
            elif method == "black_litterman":
                result = await self.black_litterman(universe, signals)
            elif method == "risk_parity":
                result = await self.risk_parity(universe)
            elif method == "hierarchical_risk_parity":
                result = await self.hierarchical_risk_parity(universe)
            
            # Accumulate weighted results
            for symbol, allocation in result.items():
                if symbol not in all_weights:
                    all_weights[symbol] = 0
                all_weights[symbol] += allocation * weight
        
        # Normalize to ensure sum = 1
        total_weight = sum(all_weights.values())
        if total_weight > 0:
            all_weights = {k: v/total_weight for k, v in all_weights.items()}
        
        return all_weights
    
    async def _get_expected_returns(self, universe: List[str]) -> np.ndarray:
        """Get expected returns for assets"""
        # This would connect to your prediction system
        # For now, return random returns
        return np.random.uniform(0.05, 0.15, len(universe))
    
    async def _get_covariance_matrix(self, universe: List[str]) -> np.ndarray:
        """Get covariance matrix for assets"""
        # This would calculate from historical data
        # For now, return random positive semi-definite matrix
        n = len(universe)
        A = np.random.randn(n, n)
        return np.dot(A, A.T) * 0.01  # Scale to reasonable volatility
    
    async def _get_market_caps(self, universe: List[str]) -> Dict:
        """Get market capitalizations"""
        # This would fetch real market caps
        return {symbol: np.random.uniform(10e9, 1000e9) for symbol in universe}
    
    async def _get_returns_dataframe(self, universe: List[str]) -> pd.DataFrame:
        """Get returns dataframe for analysis"""
        # This would fetch real returns
        # For now, create synthetic returns
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        data = {}
        for symbol in universe:
            data[symbol] = np.random.randn(252) * 0.02 + 0.0005  # 2% vol, 0.05% daily return
        
        return pd.DataFrame(data, index=dates)
    
    async def _optimize_with_returns(self, 
                                    universe: List[str],
                                    returns: np.ndarray,
                                    cov_matrix: np.ndarray) -> Dict:
        """Optimize portfolio given returns and covariance"""
        n_assets = len(universe)
        
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return {universe[i]: result.x[i] for i in range(n_assets)}
        else:
            return {symbol: 1/n_assets for symbol in universe}

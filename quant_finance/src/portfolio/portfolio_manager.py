import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize

class PortfolioManager:
    """A class for portfolio optimization and analysis using Modern Portfolio Theory."""
    
    def __init__(self, returns_data: pd.DataFrame):
        """Initialize with asset returns data.
        
        Args:
            returns_data: DataFrame with asset returns (columns are assets)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        
    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate portfolio return and risk.
        
        Args:
            weights: Array of asset weights
            
        Returns:
            Tuple of (expected return, volatility)
        """
        portfolio_return = np.sum(self.mean_returns * weights) * 252  # Annualized
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        )
        return portfolio_return, portfolio_std
    
    def optimize_portfolio(
        self,
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.01
    ) -> Dict:
        """Optimize portfolio weights for maximum Sharpe Ratio or minimum volatility.
        
        Args:
            target_return: Target portfolio return (optional)
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with optimization results
        """
        num_assets = len(self.returns.columns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        bounds = tuple((0, 1) for _ in range(num_assets))  # weights between 0 and 1
        
        def negative_sharpe_ratio(weights):
            """Calculate negative Sharpe ratio for minimization."""
            ret, std = self.calculate_portfolio_metrics(weights)
            return -(ret - risk_free_rate) / std
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.calculate_portfolio_metrics(x)[0] - target_return
            })
            
            # Minimize volatility
            result = minimize(
                lambda x: self.calculate_portfolio_metrics(x)[1],
                x0=np.array([1/num_assets] * num_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            # Maximize Sharpe ratio
            result = minimize(
                negative_sharpe_ratio,
                x0=np.array([1/num_assets] * num_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        
        optimal_return, optimal_std = self.calculate_portfolio_metrics(result.x)
        sharpe_ratio = (optimal_return - risk_free_rate) / optimal_std
        
        return {
            'weights': dict(zip(self.returns.columns, result.x)),
            'expected_return': optimal_return,
            'volatility': optimal_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def efficient_frontier(
        self,
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """Generate efficient frontier points.
        
        Args:
            num_portfolios: Number of points to generate
            
        Returns:
            DataFrame with efficient frontier portfolios
        """
        returns_range = np.linspace(
            self.mean_returns.min() * 252,
            self.mean_returns.max() * 252,
            num_portfolios
        )
        
        efficient_portfolios = []
        for target in returns_range:
            result = self.optimize_portfolio(target_return=target)
            efficient_portfolios.append({
                'Return': result['expected_return'],
                'Volatility': result['volatility'],
                'Sharpe Ratio': result['sharpe_ratio']
            })
        
        return pd.DataFrame(efficient_portfolios)
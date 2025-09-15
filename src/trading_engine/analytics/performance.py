"""
Performance Analytics - Comprehensive performance tracking and analysis
Calculates all trading metrics and generates reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and analyzes trading performance
    Calculates institutional-grade metrics
    """
    
    def __init__(self):
        # Trade tracking
        self.trades = []
        self.open_positions = {}
        self.closed_trades = []
        
        # Performance metrics
        self.total_return = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Time series data
        self.equity_curve = []
        self.daily_returns = []
        self.monthly_returns = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = 100000  # Starting capital
        
        # Advanced metrics
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.information_ratio = 0.0
        
        logger.info("Performance Tracker initialized")
    
    async def calculate_metrics(self, positions: Dict, trade_count: int) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['total_trades'] = self.total_trades
        metrics['open_positions'] = len(positions)
        metrics['winning_trades'] = self.winning_trades
        metrics['losing_trades'] = self.losing_trades
        
        # Win rate
        if self.total_trades > 0:
            metrics['win_rate'] = self.winning_trades / self.total_trades
        else:
            metrics['win_rate'] = 0
        
        # Profit metrics
        metrics['total_pnl'] = self._calculate_total_pnl()
        metrics['realized_pnl'] = self._calculate_realized_pnl()
        metrics['unrealized_pnl'] = self._calculate_unrealized_pnl(positions)
        
        # Average trade metrics
        if self.closed_trades:
            profits = [t['pnl'] for t in self.closed_trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in self.closed_trades if t['pnl'] < 0]
            
            metrics['avg_win'] = np.mean(profits) if profits else 0
            metrics['avg_loss'] = np.mean(losses) if losses else 0
            metrics['profit_factor'] = abs(sum(profits) / sum(losses)) if losses else float('inf')
            
            # Largest win/loss
            metrics['largest_win'] = max(profits) if profits else 0
            metrics['largest_loss'] = min(losses) if losses else 0
        
        # Risk metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio()
        metrics['sortino_ratio'] = self._calculate_sortino_ratio()
        metrics['calmar_ratio'] = self._calculate_calmar_ratio()
        metrics['max_drawdown'] = self.max_drawdown
        metrics['current_drawdown'] = self.current_drawdown
        
        # Additional metrics
        metrics['win_loss_ratio'] = self._calculate_win_loss_ratio()
        metrics['expectancy'] = self._calculate_expectancy()
        metrics['kelly_criterion'] = self._calculate_kelly_criterion()
        metrics['recovery_factor'] = self._calculate_recovery_factor()
        metrics['profit_per_trade'] = self._calculate_profit_per_trade()
        
        # Time-based metrics
        metrics['daily_return'] = self._calculate_daily_return()
        metrics['monthly_return'] = self._calculate_monthly_return()
        metrics['annual_return'] = self._calculate_annual_return()
        
        # Risk-adjusted returns
        metrics['risk_adjusted_return'] = self._calculate_risk_adjusted_return()
        
        return metrics
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trades.append(trade)
        self.total_trades += 1
        
        if trade['pnl'] > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update closed trades
        if trade.get('status') == 'closed':
            self.closed_trades.append(trade)
        
        # Update equity curve
        self._update_equity_curve(trade)
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total P&L"""
        if not self.trades:
            return 0.0
        
        return sum(trade.get('pnl', 0) for trade in self.trades)
    
    def _calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades"""
        if not self.closed_trades:
            return 0.0
        
        return sum(trade.get('pnl', 0) for trade in self.closed_trades)
    
    def _calculate_unrealized_pnl(self, positions: Dict) -> float:
        """Calculate unrealized P&L from open positions"""
        unrealized = 0.0
        
        for symbol, position in positions.items():
            if 'unrealized_pnl' in position:
                unrealized += float(position['unrealized_pnl'])
        
        return unrealized
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 30:
            return 0.0
        
        returns = np.array(self.daily_returns)
        
        # Assuming 2% annual risk-free rate
        risk_free_rate = 0.02 / 252
        
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        self.sharpe_ratio = sharpe
        return sharpe
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.daily_returns) < 30:
            return 0.0
        
        returns = np.array(self.daily_returns)
        risk_free_rate = 0.02 / 252
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = float('inf')  # No downside
        
        self.sortino_ratio = sortino
        return sortino
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        annual_return = self._calculate_annual_return()
        
        if self.max_drawdown > 0:
            calmar = annual_return / self.max_drawdown
        else:
            calmar = float('inf')
        
        self.calmar_ratio = calmar
        return calmar
    
    def _calculate_win_loss_ratio(self) -> float:
        """Calculate win/loss ratio"""
        if not self.closed_trades:
            return 0.0
        
        wins = [t['pnl'] for t in self.closed_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.closed_trades if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss > 0:
            return avg_win / avg_loss
        
        return float('inf') if avg_win > 0 else 0
    
    def _calculate_expectancy(self) -> float:
        """Calculate trade expectancy"""
        if not self.closed_trades:
            return 0.0
        
        win_rate = self.winning_trades / len(self.closed_trades) if self.closed_trades else 0
        
        wins = [t['pnl'] for t in self.closed_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.closed_trades if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return expectancy
    
    def _calculate_kelly_criterion(self) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not self.closed_trades:
            return 0.0
        
        win_rate = self.winning_trades / len(self.closed_trades) if self.closed_trades else 0
        win_loss_ratio = self._calculate_win_loss_ratio()
        
        if win_loss_ratio > 0:
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            return max(0, min(0.25, kelly))  # Cap at 25%
        
        return 0.0
    
    def _calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)"""
        net_profit = self._calculate_total_pnl()
        
        if self.max_drawdown > 0:
            return net_profit / self.max_drawdown
        
        return float('inf') if net_profit > 0 else 0
    
    def _calculate_profit_per_trade(self) -> float:
        """Calculate average profit per trade"""
        if self.total_trades == 0:
            return 0.0
        
        return self._calculate_total_pnl() / self.total_trades
    
    def _calculate_daily_return(self) -> float:
        """Calculate average daily return"""
        if not self.daily_returns:
            return 0.0
        
        return np.mean(self.daily_returns)
    
    def _calculate_monthly_return(self) -> float:
        """Calculate average monthly return"""
        if not self.monthly_returns:
            return 0.0
        
        return np.mean(self.monthly_returns)
    
    def _calculate_annual_return(self) -> float:
        """Calculate annualized return"""
        if not self.daily_returns:
            return 0.0
        
        # Compound daily returns
        total_return = np.prod(1 + np.array(self.daily_returns)) - 1
        
        # Annualize
        days = len(self.daily_returns)
        if days > 0:
            annual_return = (1 + total_return) ** (252 / days) - 1
        else:
            annual_return = 0.0
        
        return annual_return
    
    def _calculate_risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return"""
        annual_return = self._calculate_annual_return()
        
        if len(self.daily_returns) > 0:
            annual_volatility = np.std(self.daily_returns) * np.sqrt(252)
            if annual_volatility > 0:
                return annual_return / annual_volatility
        
        return 0.0
    
    def _update_equity_curve(self, trade: Dict):
        """Update equity curve with trade result"""
        if self.equity_curve:
            last_equity = self.equity_curve[-1]['equity']
        else:
            last_equity = 100000  # Starting capital
        
        new_equity = last_equity + trade.get('pnl', 0)
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': new_equity,
            'trade_id': trade.get('id'),
            'pnl': trade.get('pnl', 0)
        })
        
        # Update drawdown
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Add to daily returns
        daily_return = trade.get('pnl', 0) / last_equity
        self.daily_returns.append(daily_return)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'summary': {
                'total_trades': self.total_trades,
                'win_rate': f"{(self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0:.1f}%",
                'total_return': f"{self._calculate_annual_return() * 100:.2f}%",
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'max_drawdown': f"{self.max_drawdown * 100:.2f}%"
            },
            'detailed_metrics': {
                'trades': {
                    'total': self.total_trades,
                    'winners': self.winning_trades,
                    'losers': self.losing_trades,
                    'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0
                },
                'returns': {
                    'total_pnl': self._calculate_total_pnl(),
                    'daily_return': self._calculate_daily_return(),
                    'monthly_return': self._calculate_monthly_return(),
                    'annual_return': self._calculate_annual_return()
                },
                'risk_metrics': {
                    'sharpe_ratio': self.sharpe_ratio,
                    'sortino_ratio': self.sortino_ratio,
                    'calmar_ratio': self.calmar_ratio,
                    'max_drawdown': self.max_drawdown,
                    'current_drawdown': self.current_drawdown
                },
                'trade_analysis': {
                    'expectancy': self._calculate_expectancy(),
                    'profit_factor': self._calculate_profit_factor(),
                    'kelly_criterion': self._calculate_kelly_criterion(),
                    'recovery_factor': self._calculate_recovery_factor()
                }
            },
            'equity_curve': self.equity_curve[-100:] if len(self.equity_curve) > 100 else self.equity_curve,
            'daily_returns': self.daily_returns[-252:] if len(self.daily_returns) > 252 else self.daily_returns
        }
        
        return report
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.closed_trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in self.closed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.closed_trades if t['pnl'] < 0))
        
        if gross_loss > 0:
            return gross_profit / gross_loss
        
        return float('inf') if gross_profit > 0 else 0

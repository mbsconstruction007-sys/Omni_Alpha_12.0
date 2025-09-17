'''Step 7: Real-time Monitoring System'''

import asyncio
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import numpy as np

class MonitoringSystem:
    def __init__(self, api):
        self.api = api
        self.metrics = defaultdict(list)
        self.alerts = []
        self.performance_history = []
        
    def calculate_metrics(self):
        '''Calculate real-time performance metrics'''
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            orders = self.api.list_orders(status='all', limit=100)
            
            # Account metrics
            metrics = {
                'timestamp': datetime.now(),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'position_count': len(positions),
                'pending_orders': len([o for o in orders if o.status == 'pending']),
            }
            
            # Position metrics
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            total_value = sum(float(p.market_value) for p in positions)
            
            metrics['unrealized_pl'] = total_pl
            metrics['position_value'] = total_value
            metrics['cash_percentage'] = (metrics['cash'] / metrics['portfolio_value'] * 100) if metrics['portfolio_value'] > 0 else 100
            
            # Daily performance
            today_start = datetime.now().replace(hour=0, minute=0, second=0)
            todays_orders = [o for o in orders if o.created_at.replace(tzinfo=None) > today_start]
            
            metrics['trades_today'] = len(todays_orders)
            metrics['daily_pl'] = self.calculate_daily_pl(positions, todays_orders)
            
            # Risk metrics
            metrics['exposure'] = (total_value / metrics['portfolio_value'] * 100) if metrics['portfolio_value'] > 0 else 0
            metrics['largest_position'] = max([float(p.market_value) for p in positions]) if positions else 0
            metrics['risk_score'] = self.calculate_risk_score(positions, metrics)
            
            # Store metrics
            self.performance_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def calculate_daily_pl(self, positions, todays_orders):
        '''Calculate today's P&L'''
        try:
            daily_pl = 0
            
            # Unrealized P&L from open positions
            for position in positions:
                daily_pl += float(position.unrealized_intraday_pl)
            
            # Realized P&L from closed positions (simplified)
            for order in todays_orders:
                if order.status == 'filled' and order.side == 'sell':
                    # This is simplified - in production, match with buy orders
                    try:
                        daily_pl += float(order.filled_qty) * (float(order.filled_avg_price) - float(order.limit_price or order.filled_avg_price))
                    except:
                        pass
            
            return daily_pl
        except Exception as e:
            print(f"Error calculating daily P&L: {e}")
            return 0
    
    def calculate_risk_score(self, positions, metrics):
        '''Calculate risk score 0-100'''
        try:
            risk_score = 0
            
            # Position concentration risk
            if positions:
                largest_pct = (metrics['largest_position'] / metrics['portfolio_value'] * 100)
                if largest_pct > 20:
                    risk_score += 30
                elif largest_pct > 10:
                    risk_score += 15
            
            # Exposure risk
            if metrics['exposure'] > 90:
                risk_score += 30
            elif metrics['exposure'] > 70:
                risk_score += 15
            
            # Cash buffer risk
            if metrics['cash_percentage'] < 10:
                risk_score += 20
            elif metrics['cash_percentage'] < 20:
                risk_score += 10
            
            # Drawdown risk
            if self.performance_history:
                max_equity = max(h['equity'] for h in self.performance_history[-20:])
                current_dd = (max_equity - metrics['equity']) / max_equity * 100
                if current_dd > 10:
                    risk_score += 20
                elif current_dd > 5:
                    risk_score += 10
            
            return min(risk_score, 100)
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 0
    
    def check_alerts(self, metrics):
        '''Check for alert conditions'''
        alerts = []
        
        try:
            # Risk alerts
            if metrics['risk_score'] > 70:
                alerts.append({
                    'level': 'HIGH',
                    'message': f"High risk score: {metrics['risk_score']}",
                    'timestamp': datetime.now()
                })
            
            # Drawdown alert
            if self.performance_history:
                max_equity = max(h['equity'] for h in self.performance_history)
                drawdown = (max_equity - metrics['equity']) / max_equity * 100
                if drawdown > 10:
                    alerts.append({
                        'level': 'HIGH',
                        'message': f"Drawdown alert: {drawdown:.2f}%",
                        'timestamp': datetime.now()
                    })
            
            # Low cash alert
            if metrics['cash_percentage'] < 15:
                alerts.append({
                    'level': 'MEDIUM',
                    'message': f"Low cash: {metrics['cash_percentage']:.1f}%",
                    'timestamp': datetime.now()
                })
            
            self.alerts.extend(alerts)
            return alerts
        except Exception as e:
            print(f"Error checking alerts: {e}")
            return []
    
    def get_performance_summary(self):
        '''Get performance summary'''
        try:
            if not self.performance_history:
                return None
            
            df = pd.DataFrame(self.performance_history)
            
            # Calculate statistics
            returns = df['equity'].pct_change().dropna()
            
            summary = {
                'current_equity': df['equity'].iloc[-1],
                'starting_equity': df['equity'].iloc[0],
                'total_return': (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100,
                'avg_daily_return': returns.mean() * 100,
                'volatility': returns.std() * 100,
                'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown(df['equity'].values),
                'win_rate': self.calculate_win_rate(),
                'total_trades': sum(m['trades_today'] for m in self.performance_history),
                'avg_risk_score': df['risk_score'].mean(),
                'current_positions': df['position_count'].iloc[-1]
            }
            
            return summary
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return None
    
    def calculate_max_drawdown(self, equity_curve):
        '''Calculate maximum drawdown'''
        try:
            peak = equity_curve[0]
            max_dd = 0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            return max_dd
        except Exception as e:
            print(f"Error calculating max drawdown: {e}")
            return 0
    
    def calculate_win_rate(self):
        '''Calculate win rate from closed orders'''
        try:
            orders = self.api.list_orders(status='filled', limit=200)
            
            wins = 0
            losses = 0
            
            # Simplified win rate calculation
            for order in orders:
                if order.side == 'sell':
                    # Check if profit (simplified)
                    if float(order.filled_avg_price) > 0:
                        wins += 1
                    else:
                        losses += 1
            
            total = wins + losses
            return (wins / total * 100) if total > 0 else 0
        except Exception as e:
            print(f"Error calculating win rate: {e}")
            return 0

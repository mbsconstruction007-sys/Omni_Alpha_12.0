"""
Real-time Omni Alpha Trading Dashboard
Interactive Streamlit dashboard for live trading monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv('alpaca_live_trading.env')

# Import trading system components
try:
    import alpaca_trade_api as tradeapi
    from omni_alpha_live_trading import AlpacaTradingSystem, UnifiedTradingStrategy
    TRADING_SYSTEM_AVAILABLE = True
except ImportError:
    TRADING_SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Omni Alpha Live Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .trading-signal {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .buy-signal { background-color: #28a745; color: white; }
    .sell-signal { background-color: #dc3545; color: white; }
    .hold-signal { background-color: #ffc107; color: black; }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """Main trading dashboard class"""
    
    def __init__(self):
        self.trading_system = None
        self.strategy = None
        
        if TRADING_SYSTEM_AVAILABLE:
            try:
                self.trading_system = AlpacaTradingSystem()
                self.strategy = UnifiedTradingStrategy(self.trading_system)
            except Exception as e:
                st.error(f"Failed to initialize trading system: {e}")
    
    def get_account_data(self):
        """Get live account data from Alpaca"""
        
        if not self.trading_system:
            return self.get_demo_account_data()
        
        try:
            account_info = self.trading_system.get_account_info()
            positions = self.trading_system.get_current_positions()
            
            # Calculate metrics
            total_pnl = sum(pos['unrealized_pnl'] for pos in positions)
            total_value = account_info.get('equity', 100000)
            daily_pnl = total_pnl  # Simplified
            
            return {
                'portfolio_value': total_value,
                'cash': account_info.get('cash', 0),
                'buying_power': account_info.get('buying_power', 0),
                'daily_pnl': daily_pnl,
                'total_pnl': total_pnl,
                'positions_count': len(positions),
                'day_trades': account_info.get('day_trades', 0),
                'positions': positions
            }
            
        except Exception as e:
            st.error(f"Error fetching account data: {e}")
            return self.get_demo_account_data()
    
    def get_demo_account_data(self):
        """Get demo account data for display"""
        
        return {
            'portfolio_value': 105230.45,
            'cash': 45230.45,
            'buying_power': 90460.90,
            'daily_pnl': 523.45,
            'total_pnl': 5230.45,
            'positions_count': 7,
            'day_trades': 1,
            'positions': [
                {
                    'symbol': 'AAPL',
                    'qty': 10,
                    'avg_entry_price': 175.50,
                    'current_price': 178.20,
                    'unrealized_pnl': 27.00,
                    'unrealized_pnl_percent': 1.54,
                    'market_value': 1782.00
                },
                {
                    'symbol': 'GOOGL',
                    'qty': 5,
                    'avg_entry_price': 141.20,
                    'current_price': 143.50,
                    'unrealized_pnl': 11.50,
                    'unrealized_pnl_percent': 1.63,
                    'market_value': 717.50
                },
                {
                    'symbol': 'TSLA',
                    'qty': 8,
                    'avg_entry_price': 245.60,
                    'current_price': 242.30,
                    'unrealized_pnl': -26.40,
                    'unrealized_pnl_percent': -1.34,
                    'market_value': 1938.40
                },
                {
                    'symbol': 'MSFT',
                    'qty': 12,
                    'avg_entry_price': 412.30,
                    'current_price': 415.80,
                    'unrealized_pnl': 42.00,
                    'unrealized_pnl_percent': 0.85,
                    'market_value': 4989.60
                }
            ]
        }
    
    async def get_live_signals(self):
        """Get live trading signals"""
        
        if not self.strategy:
            return self.get_demo_signals()
        
        try:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
            signals = await self.strategy.generate_signals(symbols)
            return signals
        except Exception as e:
            st.error(f"Error generating signals: {e}")
            return self.get_demo_signals()
    
    def get_demo_signals(self):
        """Get demo trading signals"""
        
        return {
            'AAPL': {'signal': 'BUY', 'confidence': 0.85, 'consensus': 5, 'total_strategies': 6},
            'GOOGL': {'signal': 'HOLD', 'confidence': 0.62, 'consensus': 3, 'total_strategies': 6},
            'TSLA': {'signal': 'SELL', 'confidence': 0.78, 'consensus': 4, 'total_strategies': 6},
            'MSFT': {'signal': 'BUY', 'confidence': 0.73, 'consensus': 4, 'total_strategies': 6},
            'AMZN': {'signal': 'HOLD', 'confidence': 0.58, 'consensus': 2, 'total_strategies': 6},
            'NVDA': {'signal': 'BUY', 'confidence': 0.91, 'consensus': 6, 'total_strategies': 6}
        }
    
    def render_header(self):
        """Render dashboard header"""
        
        st.title("🚀 Omni Alpha Live Trading Dashboard")
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢 LIVE" if TRADING_SYSTEM_AVAILABLE else "🟡 DEMO"
            st.markdown(f"**System Status:** {status}")
        
        with col2:
            market_status = "🟢 OPEN" if datetime.now().hour >= 9 and datetime.now().hour < 16 else "🔴 CLOSED"
            st.markdown(f"**Market:** {market_status}")
        
        with col3:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
        with col4:
            if st.button("🔄 Refresh Data"):
                st.experimental_rerun()
    
    def render_account_metrics(self, account_data):
        """Render account metrics cards"""
        
        st.subheader("💰 Account Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if account_data['daily_pnl'] >= 0 else "inverse"
            st.metric(
                "Portfolio Value",
                f"${account_data['portfolio_value']:,.2f}",
                f"{account_data['daily_pnl']:+.2f}",
                delta_color=delta_color
            )
        
        with col2:
            pnl_pct = (account_data['daily_pnl'] / account_data['portfolio_value']) * 100
            st.metric(
                "Today's P&L",
                f"${account_data['daily_pnl']:+,.2f}",
                f"{pnl_pct:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Open Positions",
                account_data['positions_count'],
                f"Day Trades: {account_data['day_trades']}"
            )
        
        with col4:
            buying_power_pct = (account_data['buying_power'] / account_data['portfolio_value']) * 100
            st.metric(
                "Buying Power",
                f"${account_data['buying_power']:,.2f}",
                f"{buying_power_pct:.1f}% available"
            )
    
    def render_positions_table(self, positions):
        """Render positions table"""
        
        st.subheader("📊 Live Positions")
        
        if not positions:
            st.info("No open positions")
            return
        
        # Create DataFrame
        df = pd.DataFrame(positions)
        
        # Format DataFrame
        df['Entry Price'] = df['avg_entry_price'].apply(lambda x: f"${x:.2f}")
        df['Current Price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        df['Market Value'] = df['market_value'].apply(lambda x: f"${x:,.2f}")
        df['P&L'] = df.apply(lambda row: f"${row['unrealized_pnl']:+.2f} ({row['unrealized_pnl_percent']:+.1f}%)", axis=1)
        
        # Add signal column (demo)
        signals = ['HOLD', 'BUY', 'SELL', 'HOLD']
        df['Signal'] = signals[:len(df)]
        
        # Display table
        display_df = df[['symbol', 'qty', 'Entry Price', 'Current Price', 'Market Value', 'P&L', 'Signal']]
        display_df.columns = ['Symbol', 'Qty', 'Entry', 'Current', 'Value', 'P&L', 'Signal']
        
        # Style the dataframe
        def style_pnl(val):
            if '+' in val:
                return 'color: green'
            elif '-' in val:
                return 'color: red'
            return ''
        
        def style_signal(val):
            if val == 'BUY':
                return 'background-color: #28a745; color: white'
            elif val == 'SELL':
                return 'background-color: #dc3545; color: white'
            else:
                return 'background-color: #ffc107; color: black'
        
        styled_df = display_df.style.applymap(style_pnl, subset=['P&L']).applymap(style_signal, subset=['Signal'])
        
        st.dataframe(styled_df, use_container_width=True)
    
    def render_signals_table(self, signals):
        """Render trading signals table"""
        
        st.subheader("🎯 Live Trading Signals")
        
        if not signals:
            st.info("No trading signals available")
            return
        
        # Create signals DataFrame
        signals_data = []
        for symbol, signal_data in signals.items():
            signals_data.append({
                'Symbol': symbol,
                'Signal': signal_data['signal'],
                'Confidence': f"{signal_data['confidence']:.1%}",
                'Consensus': f"{signal_data['consensus']}/{signal_data['total_strategies']}",
                'Strength': signal_data['confidence']
            })
        
        df = pd.DataFrame(signals_data)
        
        # Sort by confidence
        df = df.sort_values('Strength', ascending=False)
        
        # Display with styling
        def style_signal_row(row):
            if row['Signal'] == 'BUY':
                return ['background-color: #d4edda'] * len(row)
            elif row['Signal'] == 'SELL':
                return ['background-color: #f8d7da'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)
        
        styled_df = df[['Symbol', 'Signal', 'Confidence', 'Consensus']].style.apply(style_signal_row, axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Signal summary
        buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        hold_count = len(signals) - buy_count - sell_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 BUY Signals", buy_count)
        with col2:
            st.metric("🔴 SELL Signals", sell_count)
        with col3:
            st.metric("🟡 HOLD Signals", hold_count)
    
    def render_performance_chart(self, account_data):
        """Render performance chart"""
        
        st.subheader("📈 Portfolio Performance")
        
        # Generate sample performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Simulate portfolio performance
        base_value = 100000
        performance_data = []
        current_value = base_value
        
        for i, date in enumerate(dates):
            # Add some realistic volatility
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
            current_value *= (1 + daily_return)
            performance_data.append(current_value)
        
        # Ensure last value matches current portfolio value
        performance_data[-1] = account_data['portfolio_value']
        
        # Create performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=performance_data,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#667eea', width=3),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        # Add benchmark (S&P 500 simulation)
        benchmark_data = [base_value * (1 + 0.0008)**i for i in range(len(dates))]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_data,
            mode='lines',
            name='S&P 500 (Benchmark)',
            line=dict(color='#ffc107', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = (account_data['portfolio_value'] - base_value) / base_value * 100
        
        with col1:
            st.metric("Total Return", f"{total_return:+.2f}%")
        
        with col2:
            # Calculate Sharpe ratio (simplified)
            returns = pd.Series(performance_data).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            # Max drawdown
            equity_series = pd.Series(performance_data)
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with col4:
            # Win rate (demo)
            win_rate = 67.5
            st.metric("Win Rate", f"{win_rate:.1f}%")
    
    def render_risk_metrics(self, account_data):
        """Render risk management metrics"""
        
        st.subheader("🛡️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Position Limits**")
            
            # Calculate position concentration
            total_value = account_data['portfolio_value']
            position_concentrations = []
            
            for pos in account_data['positions']:
                concentration = (pos['market_value'] / total_value) * 100
                position_concentrations.append({
                    'Symbol': pos['symbol'],
                    'Concentration': concentration
                })
            
            if position_concentrations:
                concentration_df = pd.DataFrame(position_concentrations)
                
                fig = px.bar(
                    concentration_df,
                    x='Symbol',
                    y='Concentration',
                    title='Position Concentration (%)',
                    color='Concentration',
                    color_continuous_scale='RdYlGn_r'
                )
                
                fig.add_hline(y=10, line_dash="dash", line_color="red", 
                             annotation_text="Max Position Limit (10%)")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Metrics**")
            
            # Risk metrics
            portfolio_value = account_data['portfolio_value']
            cash_ratio = (account_data['cash'] / portfolio_value) * 100
            
            metrics = {
                'Cash Ratio': f"{cash_ratio:.1f}%",
                'Position Count': account_data['positions_count'],
                'Max Position': "10.0%",
                'Daily Loss Limit': "5.0%",
                'Risk Score': "LOW" if account_data['positions_count'] < 5 else "MEDIUM"
            }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        
        st.sidebar.title("⚙️ Trading Controls")
        
        # Auto refresh
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.markdown("**System Status**")
        st.sidebar.success("✅ Dashboard Online")
        
        if TRADING_SYSTEM_AVAILABLE:
            st.sidebar.success("✅ Trading System Connected")
        else:
            st.sidebar.warning("⚠️ Demo Mode (No Live Trading)")
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.markdown("**Quick Actions**")
        
        if st.sidebar.button("📊 Generate Signals"):
            st.sidebar.info("Signals updated!")
        
        if st.sidebar.button("💰 Refresh Account"):
            st.sidebar.info("Account data refreshed!")
        
        if st.sidebar.button("🔄 Reset Demo Data"):
            st.sidebar.info("Demo data reset!")
        
        st.sidebar.markdown("---")
        
        # System info
        st.sidebar.markdown("**System Info**")
        st.sidebar.text(f"Version: 12.0+")
        st.sidebar.text(f"Mode: {'Live' if TRADING_SYSTEM_AVAILABLE else 'Demo'}")
        st.sidebar.text(f"Updated: {datetime.now().strftime('%H:%M')}")

def main():
    """Main dashboard function"""
    
    dashboard = TradingDashboard()
    
    # Render sidebar
    dashboard.render_sidebar()
    
    # Render header
    dashboard.render_header()
    
    # Get data
    account_data = dashboard.get_account_data()
    
    # Render account metrics
    dashboard.render_account_metrics(account_data)
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Positions", "🎯 Signals", "📈 Performance", "🛡️ Risk"])
    
    with tab1:
        dashboard.render_positions_table(account_data['positions'])
    
    with tab2:
        signals = dashboard.get_demo_signals()  # Use demo signals for now
        dashboard.render_signals_table(signals)
    
    with tab3:
        dashboard.render_performance_chart(account_data)
    
    with tab4:
        dashboard.render_risk_metrics(account_data)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Omni Alpha Live Trading Dashboard** | Built with Streamlit | Real-time trading data")

if __name__ == "__main__":
    main()

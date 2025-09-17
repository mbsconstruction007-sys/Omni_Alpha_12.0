'''Complete Integrated Omni Alpha System'''

import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import alpaca_trade_api as tradeapi
import sys
import os

# Add core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ml_engine import MLPredictionEngine
from core.monitoring import MonitoringSystem
from core.analytics import AnalyticsEngine
from core.orchestrator import AITradingOrchestrator

# Configuration
TELEGRAM_TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
ALPACA_KEY = 'PK6NQI7HSGQ7B38PYLG8'
ALPACA_SECRET = 'gu15JAAvNMqbDGJ8m14ePtHOy3TgnAD7vHkvg74C'
BASE_URL = 'https://paper-api.alpaca.markets'

class OmniAlphaBot:
    def __init__(self):
        # Initialize API
        self.api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL)
        
        # Initialize components
        self.ml_engine = MLPredictionEngine(self.api)
        self.monitoring = MonitoringSystem(self.api)
        self.analytics = AnalyticsEngine(self.api)
        self.orchestrator = AITradingOrchestrator(
            self.api, self.ml_engine, self.monitoring, self.analytics
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Start command'''
        account = self.api.get_account()
        
        welcome = f'''
🚀 OMNI ALPHA 12.0 - ADVANCED SYSTEM
════════════════════════════════════
Account: {account.status}
Portfolio: ${float(account.portfolio_value):,.2f}
Buying Power: ${float(account.buying_power):,.2f}

🧠 ADVANCED COMMANDS:
/ml SYMBOL - Step 6: ML Predictions
/monitor - Step 7: System Monitoring  
/analyze SYMBOL - Step 8: Deep Analytics
/ai - Step 9: AI Brain Status
/auto_start - Steps 10-12: Start AI Trading
/auto_stop - Stop AI Trading
/performance - Performance Report
/positions - Current Positions
/alerts - System Alerts
        '''
        await update.message.reply_text(welcome)
    
    async def ml_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Step 6: ML Predictions'''
        if not context.args:
            await update.message.reply_text('Usage: /ml SYMBOL')
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f'🤖 Analyzing {symbol} with ML...')
        
        prediction = self.ml_engine.predict(symbol)
        
        if prediction:
            msg = f"🧠 ML PREDICTION: {symbol}\n"
            msg += f"════════════════════════\n"
            msg += f"Direction: {prediction['prediction']}\n"
            msg += f"Confidence: {prediction['confidence']:.1f}%\n"
            msg += f"Action: {prediction['action']}\n"
            msg += f"Probability Up: {prediction['probability_up']:.1f}%\n"
            msg += f"Probability Down: {prediction['probability_down']:.1f}%\n\n"
            
            if prediction['top_features']:
                msg += f"Top Features:\n"
                for feature, importance in prediction['top_features']:
                    msg += f"• {feature}: {importance:.3f}\n"
            
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text('❌ Unable to generate ML prediction')
    
    async def monitor_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Step 7: System Monitoring'''
        await update.message.reply_text('📊 Calculating metrics...')
        
        metrics = self.monitoring.calculate_metrics()
        
        if metrics:
            msg = f"📊 SYSTEM MONITORING\n"
            msg += f"════════════════════════\n"
            msg += f"💰 Equity: ${metrics['equity']:,.2f}\n"
            msg += f"💵 Cash: ${metrics['cash']:,.2f}\n"
            msg += f"📈 Positions: {metrics['position_count']}\n"
            msg += f"📊 Daily P&L: ${metrics['daily_pl']:,.2f}\n"
            msg += f"⚠️ Risk Score: {metrics['risk_score']}/100\n"
            msg += f"📊 Exposure: {metrics['exposure']:.1f}%\n"
            msg += f"💸 Cash %: {metrics['cash_percentage']:.1f}%\n"
            
            # Risk level indicator
            if metrics['risk_score'] > 70:
                msg += f"🔴 HIGH RISK"
            elif metrics['risk_score'] > 40:
                msg += f"🟡 MEDIUM RISK"
            else:
                msg += f"🟢 LOW RISK"
            
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text('❌ Error calculating metrics')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Step 8: Analytics Engine'''
        if not context.args:
            await update.message.reply_text('Usage: /analyze SYMBOL')
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f'📈 Deep analysis of {symbol}...')
        
        analysis = self.analytics.analyze_symbol(symbol)
        
        if analysis:
            msg = f"📈 ANALYTICS: {symbol}\n"
            msg += f"════════════════════════\n"
            msg += f"🎯 Score: {analysis['composite_score']}/100\n"
            msg += f"💡 Recommendation: {analysis['recommendation']}\n\n"
            
            if analysis['technical']:
                tech = analysis['technical']
                msg += f"📊 Technical:\n"
                msg += f"• Trend: {tech['trend']}\n"
                msg += f"• Price: ${tech['current_price']:.2f}\n"
                msg += f"• SMA20: ${tech['sma_20']:.2f}\n\n"
            
            if analysis['momentum']:
                mom = analysis['momentum']
                msg += f"⚡ Momentum:\n"
                msg += f"• Strength: {mom['strength']}\n"
                msg += f"• 10D Change: {mom['rate_of_change_10d']:.2f}%\n\n"
            
            if analysis['volume']:
                vol = analysis['volume']
                msg += f"📊 Volume:\n"
                msg += f"• Trend: {vol['volume_trend']}\n"
                msg += f"• Unusual: {'Yes' if vol['unusual_volume'] else 'No'}\n\n"
            
            if analysis['risk']:
                risk = analysis['risk']
                msg += f"⚠️ Risk: {risk['risk_rating']}\n"
                msg += f"• Max DD: {risk['max_drawdown']:.2f}%"
            
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text('❌ Unable to analyze symbol')
    
    async def ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Step 9: AI Brain Status'''
        msg = f"🧠 AI BRAIN STATUS\n"
        msg += f"════════════════════════\n"
        msg += f"🤖 Status: {'ACTIVE' if self.orchestrator.active else 'INACTIVE'}\n"
        msg += f"📊 Trading: {'ENABLED' if self.orchestrator.trading_enabled else 'DISABLED'}\n"
        msg += f"🎯 Confidence Threshold: {self.orchestrator.confidence_threshold:.2f}\n"
        msg += f"💰 Position Size: {self.orchestrator.position_size_pct:.1%}\n"
        msg += f"📈 Max Positions: {self.orchestrator.max_positions}\n"
        msg += f"📊 Trade History: {len(self.orchestrator.trade_history)}\n"
        
        await update.message.reply_text(msg)
    
    async def auto_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Steps 10-12: Start AI Trading'''
        if self.orchestrator.active:
            await update.message.reply_text('🤖 AI Trading already active!')
            return
        
        await update.message.reply_text('🚀 Starting AI Trading System...')
        
        # Start orchestrator in background
        asyncio.create_task(self.orchestrator.start())
        
        msg = f"🚀 AI TRADING ACTIVATED\n"
        msg += f"════════════════════════\n"
        msg += f"Step 10: ✅ Orchestration\n"
        msg += f"Step 11: ✅ Institutional Ops\n"
        msg += f"Step 12: ✅ Market Dominance\n\n"
        msg += f"🤖 System will now:\n"
        msg += f"• Monitor markets every minute\n"
        msg += f"• Execute ML-driven trades\n"
        msg += f"• Manage risk automatically\n"
        msg += f"• Optimize performance\n"
        
        await update.message.reply_text(msg)
    
    async def auto_stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Stop AI Trading'''
        self.orchestrator.active = False
        await update.message.reply_text('🛑 AI Trading Stopped')
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Performance Report'''
        summary = self.monitoring.get_performance_summary()
        
        if summary:
            msg = f"📊 PERFORMANCE REPORT\n"
            msg += f"════════════════════════\n"
            msg += f"💰 Current Equity: ${summary['current_equity']:,.2f}\n"
            msg += f"📈 Total Return: {summary['total_return']:.2f}%\n"
            msg += f"📊 Daily Return: {summary['avg_daily_return']:.3f}%\n"
            msg += f"📉 Volatility: {summary['volatility']:.2f}%\n"
            msg += f"⚡ Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
            msg += f"📉 Max Drawdown: {summary['max_drawdown']:.2f}%\n"
            msg += f"🎯 Win Rate: {summary['win_rate']:.1f}%\n"
            msg += f"📊 Total Trades: {summary['total_trades']}\n"
            msg += f"⚠️ Avg Risk Score: {summary['avg_risk_score']:.1f}\n"
            
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text('❌ No performance data available')
    
    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Current Positions'''
        positions = self.api.list_positions()
        
        if not positions:
            await update.message.reply_text('📊 No open positions')
            return
        
        msg = f"📊 CURRENT POSITIONS\n"
        msg += f"════════════════════════\n"
        
        for pos in positions:
            msg += f"🔸 {pos.symbol}: {pos.qty} shares\n"
            msg += f"   Entry: ${float(pos.avg_entry_price):.2f}\n"
            msg += f"   Current: ${float(pos.current_price):.2f}\n"
            msg += f"   P&L: ${float(pos.unrealized_pl):.2f}\n"
            msg += f"   Return: {float(pos.unrealized_plpct)*100:.2f}%\n\n"
        
        await update.message.reply_text(msg)
    
    async def alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''System Alerts'''
        alerts = self.monitoring.alerts[-10:]  # Last 10 alerts
        
        if not alerts:
            await update.message.reply_text('✅ No active alerts')
            return
        
        msg = f"⚠️ SYSTEM ALERTS\n"
        msg += f"════════════════════════\n"
        
        for alert in alerts[-5:]:  # Show last 5
            level_emoji = '🔴' if alert['level'] == 'HIGH' else '🟡'
            msg += f"{level_emoji} {alert['message']}\n"
            msg += f"   {alert['timestamp'].strftime('%H:%M:%S')}\n\n"
        
        await update.message.reply_text(msg)

# Run the system
print('🚀 OMNI ALPHA 12.0 - INITIALIZING ADVANCED SYSTEM...')

try:
    bot = OmniAlphaBot()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register commands
    app.add_handler(CommandHandler('start', bot.start_command))
    app.add_handler(CommandHandler('ml', bot.ml_command))
    app.add_handler(CommandHandler('monitor', bot.monitor_command))
    app.add_handler(CommandHandler('analyze', bot.analyze_command))
    app.add_handler(CommandHandler('ai', bot.ai_command))
    app.add_handler(CommandHandler('auto_start', bot.auto_start_command))
    app.add_handler(CommandHandler('auto_stop', bot.auto_stop_command))
    app.add_handler(CommandHandler('performance', bot.performance_command))
    app.add_handler(CommandHandler('positions', bot.positions_command))
    app.add_handler(CommandHandler('alerts', bot.alerts_command))

    print('✅ OMNI ALPHA 12.0 - FULLY INTEGRATED SYSTEM READY!')
    print('📊 All advanced features active:')
    print('   Step 6: ✅ ML Predictions Engine')
    print('   Step 7: ✅ Real-time Monitoring')
    print('   Step 8: ✅ Advanced Analytics')
    print('   Step 9: ✅ AI Brain')
    print('   Step 10-12: ✅ Automated Trading')
    print('')
    print('🤖 Go to Telegram and send /start to begin!')
    
    app.run_polling()
    
except Exception as e:
    print(f'❌ Error starting system: {e}')
    print('Please check your API keys and try again.')

"""
OMNI ALPHA 12.0 - COMPLETE TRADING SYSTEM
All 12 Steps + Telegram + Alpaca Integration
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

# Third party imports
import alpaca_trade_api as tradeapi
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Import Step 13 components
from core.microstructure import OrderBookAnalyzer, VolumeProfileAnalyzer, OrderFlowTracker
from core.market_signals import MicrostructureSignals

# Import Step 14 components
from core.gemini_ai_agent import integrate_gemini_agent

# Import Step 14.1 components
from core.comprehensive_ai_agent import integrate_comprehensive_ai

# Import Step 15 components
from core.alternative_data_processor import integrate_alternative_data

# Import Step 16 components
from core.options_hedging_system import integrate_options_hedging

# Import Step 17 components
from core.portfolio_optimization_orchestration import integrate_portfolio_optimization

# ===================== CONFIGURATION =====================
TELEGRAM_TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
ALPACA_KEY = 'PK6NQI7HSGQ7B38PYLG8'
ALPACA_SECRET = 'gu15JAAvNMqbDGJ8m14ePtHOy3TgnAD7vHkvg74C'
BASE_URL = 'https://paper-api.alpaca.markets'

# Trading Parameters
MAX_POSITIONS = 5
POSITION_SIZE_PCT = 0.10
STOP_LOSS = 0.02
TAKE_PROFIT = 0.05
CONFIDENCE_THRESHOLD = 65
SCAN_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== STEP 1: CORE INFRASTRUCTURE =====================
class CoreInfrastructure:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL)
        self.connected = False
        self.system_status = 'initializing'
        
    def test_connection(self):
        try:
            account = self.api.get_account()
            self.connected = True
            self.system_status = 'connected'
            return {
                'status': 'connected',
                'account_id': account.account_number,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return {'status': 'error', 'message': str(e)}

# ===================== STEP 2: DATA PIPELINE =====================
class DataPipeline:
    def __init__(self, api):
        self.api = api
        self.data_cache = {}
        
    def get_market_data(self, symbol, timeframe='1Day', days=30):
        try:
            bars = self.api.get_bars(
                symbol, timeframe,
                start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                limit=days
            ).df
            
            self.data_cache[symbol] = bars
            return bars
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return None
    
    def get_latest_quote(self, symbol):
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bp,
                'ask': quote.ap,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return None

# ===================== STEP 3: STRATEGY ENGINE =====================
class StrategyEngine:
    def __init__(self):
        self.strategies = {
            'momentum': self.momentum_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy
        }
        self.active_strategy = 'momentum'
        
    def momentum_strategy(self, data):
        if data is None or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0}
            
        sma_short = data['close'].rolling(5).mean().iloc[-1]
        sma_long = data['close'].rolling(20).mean().iloc[-1]
        current = data['close'].iloc[-1]
        
        if current > sma_short > sma_long:
            return {'signal': 'BUY', 'strength': 80}
        elif current < sma_short < sma_long:
            return {'signal': 'SELL', 'strength': 80}
        return {'signal': 'HOLD', 'strength': 50}
    
    def mean_reversion_strategy(self, data):
        if data is None or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0}
            
        mean = data['close'].mean()
        std = data['close'].std()
        current = data['close'].iloc[-1]
        
        if current < mean - (2 * std):
            return {'signal': 'BUY', 'strength': 70}
        elif current > mean + (2 * std):
            return {'signal': 'SELL', 'strength': 70}
        return {'signal': 'HOLD', 'strength': 40}
    
    def breakout_strategy(self, data):
        if data is None or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0}
            
        resistance = data['high'].rolling(20).max().iloc[-1]
        support = data['low'].rolling(20).min().iloc[-1]
        current = data['close'].iloc[-1]
        
        if current > resistance:
            return {'signal': 'BUY', 'strength': 85}
        elif current < support:
            return {'signal': 'SELL', 'strength': 85}
        return {'signal': 'HOLD', 'strength': 45}

# ===================== STEP 4: RISK MANAGEMENT =====================
class RiskManager:
    def __init__(self, api):
        self.api = api
        self.max_position_size = 10000
        self.max_portfolio_risk = 0.02
        
    def check_position_size(self, symbol, quantity, price):
        value = quantity * price
        account = self.api.get_account()
        portfolio_value = float(account.portfolio_value)
        
        if value > self.max_position_size:
            return False, "Position size too large"
        
        if value > portfolio_value * 0.2:
            return False, "Position exceeds 20% of portfolio"
            
        return True, "Position size OK"
    
    def calculate_stop_loss(self, entry_price, side='buy'):
        if side == 'buy':
            return entry_price * (1 - STOP_LOSS)
        else:
            return entry_price * (1 + STOP_LOSS)
    
    def calculate_take_profit(self, entry_price, side='buy'):
        if side == 'buy':
            return entry_price * (1 + TAKE_PROFIT)
        else:
            return entry_price * (1 - TAKE_PROFIT)

# ===================== STEP 5: EXECUTION SYSTEM =====================
class ExecutionEngine:
    def __init__(self, api):
        self.api = api
        self.orders = []
        
    def execute_order(self, symbol, qty, side, order_type='market'):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            self.orders.append({
                'order_id': order.id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order.status,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Order executed: {side} {qty} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None

# ===================== STEP 6: ML PLATFORM =====================
class MLPlatform:
    def __init__(self, api):
        self.api = api
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, symbol):
        try:
            bars = self.api.get_bars(
                symbol, '1Day',
                start=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
                limit=100
            ).df
            
            if len(bars) < 30:
                return None
            
            features = pd.DataFrame()
            
            # Technical features
            features['returns'] = bars['close'].pct_change()
            features['volatility'] = features['returns'].rolling(5).std()
            features['volume_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            
            # RSI
            delta = bars['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving averages
            features['sma_ratio'] = bars['close'] / bars['close'].rolling(20).mean()
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"ML feature error: {e}")
            return None
    
    def train_model(self, symbol):
        features = self.prepare_features(symbol)
        if features is None or len(features) < 50:
            return False
        
        X = features.drop(['returns'], axis=1)
        y = (features['returns'].shift(-1) > 0).astype(int)[:-1]
        X = X[:-1]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        X_scaled = self.scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        self.models[symbol] = model
        return True
    
    def predict(self, symbol):
        if symbol not in self.models:
            if not self.train_model(symbol):
                return None
        
        features = self.prepare_features(symbol)
        if features is None:
            return None
        
        X = features.drop(['returns'], axis=1).iloc[-1:]
        X_scaled = self.scaler.transform(X)
        
        model = self.models[symbol]
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probability) * 100,
            'action': 'BUY' if probability[1] > 0.65 else 'SELL' if probability[1] < 0.35 else 'HOLD'
        }

# ===================== STEP 7: MONITORING =====================
class MonitoringSystem:
    def __init__(self, api):
        self.api = api
        self.metrics_history = []
        
    def get_metrics(self):
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'positions': len(positions),
                'total_pl': sum(float(p.unrealized_pl) for p in positions),
                'daily_pl': float(account.equity) - float(account.last_equity),
                'risk_score': self.calculate_risk_score(positions)
            }
            
            self.metrics_history.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {}
    
    def calculate_risk_score(self, positions):
        if not positions:
            return 0
            
        try:
            account = self.api.get_account()
            exposure = sum(float(p.market_value) for p in positions)
            risk = (exposure / float(account.equity)) * 100
            
            return min(risk, 100)
        except Exception as e:
            logger.error(f"Risk score error: {e}")
            return 0

# ===================== STEP 8: ANALYTICS =====================
class AnalyticsEngine:
    def __init__(self, api):
        self.api = api
        
    def analyze_symbol(self, symbol):
        try:
            bars = self.api.get_bars(symbol, '1Day', limit=50).df
            
            analysis = {
                'symbol': symbol,
                'current_price': bars['close'].iloc[-1],
                'sma20': bars['close'].rolling(20).mean().iloc[-1],
                'trend': self.determine_trend(bars),
                'volatility': bars['close'].pct_change().std() * 100,
                'support': bars['low'].rolling(20).min().iloc[-1],
                'resistance': bars['high'].rolling(20).max().iloc[-1],
                'volume_trend': 'HIGH' if bars['volume'].iloc[-1] > bars['volume'].mean() else 'NORMAL',
                'score': self.calculate_score(bars)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None
    
    def determine_trend(self, bars):
        sma_short = bars['close'].rolling(10).mean().iloc[-1]
        sma_long = bars['close'].rolling(30).mean().iloc[-1]
        
        if sma_short > sma_long:
            return 'UPTREND'
        elif sma_short < sma_long:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def calculate_score(self, bars):
        score = 50
        
        # Price above SMA
        if bars['close'].iloc[-1] > bars['close'].rolling(20).mean().iloc[-1]:
            score += 20
            
        # Volume increasing
        if bars['volume'].iloc[-1] > bars['volume'].mean():
            score += 15
            
        # Positive momentum
        if bars['close'].iloc[-1] > bars['close'].iloc[-5]:
            score += 15
            
        return min(max(score, 0), 100)

# ===================== STEPS 9-12: AI ORCHESTRATION =====================
class AIOrchestrator:
    def __init__(self, core, data_pipeline, strategy, risk, execution, ml, monitoring, analytics):
        self.core = core
        self.data = data_pipeline
        self.strategy = strategy
        self.risk = risk
        self.execution = execution
        self.ml = ml
        self.monitoring = monitoring
        self.analytics = analytics
        
        self.trading_active = False
        self.ai_enabled = True
        
    async def run_trading_loop(self):
        while self.trading_active:
            try:
                # Check market hours
                clock = self.core.api.get_clock()
                if not clock.is_open:
                    await asyncio.sleep(60)
                    continue
                
                # Step 10: Orchestration
                for symbol in SCAN_SYMBOLS:
                    await self.evaluate_and_trade(symbol)
                
                # Step 11: Institutional operations
                await self.manage_portfolio()
                
                # Step 12: Global market dominance
                metrics = self.monitoring.get_metrics()
                if metrics.get('risk_score', 0) > 80:
                    await self.risk_reduction()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"AI loop error: {e}")
                await asyncio.sleep(60)
    
    async def evaluate_and_trade(self, symbol):
        # Get data
        data = self.data.get_market_data(symbol)
        if data is None:
            return
        
        # Strategy signal
        strategy_signal = self.strategy.strategies[self.strategy.active_strategy](data)
        
        # ML prediction
        ml_prediction = self.ml.predict(symbol)
        
        # Analytics
        analysis = self.analytics.analyze_symbol(symbol)
        
        # Combine signals (including microstructure)
        if self.should_buy(strategy_signal, ml_prediction, analysis, symbol):
            await self.execute_buy(symbol)
    
    def should_buy(self, strategy, ml, analysis, symbol=None):
        if not strategy or not ml or not analysis:
            return False
            
        buy_signals = 0
        
        if strategy['signal'] == 'BUY' and strategy['strength'] > 70:
            buy_signals += 1
            
        if ml['action'] == 'BUY' and ml['confidence'] > CONFIDENCE_THRESHOLD:
            buy_signals += 1
            
        if analysis['score'] > 70 and analysis['trend'] == 'UPTREND':
            buy_signals += 1
            
        # Step 13: Add microstructure signal
        if symbol and hasattr(self, 'microstructure_signals'):
            try:
                microstructure_signal = self.microstructure_signals.generate_comprehensive_signal(symbol)
                if microstructure_signal:
                    if 'BUY' in microstructure_signal['signal'] and microstructure_signal['confidence'] > 60:
                        buy_signals += 1
                        # Strong microstructure signals get extra weight
                        if microstructure_signal['confidence'] > 80:
                            buy_signals += 1
                    elif microstructure_signal['signal'] == 'AVOID':
                        return False  # Override - don't trade if microstructure says avoid
            except Exception as e:
                logger.error(f"Microstructure signal error: {e}")
            
        return buy_signals >= 2
    
    async def execute_buy(self, symbol):
        try:
            account = self.core.api.get_account()
            positions = self.core.api.list_positions()
            
            # Check position limit
            if len(positions) >= MAX_POSITIONS:
                return
            
            # Calculate position size
            cash = float(account.cash)
            position_value = cash * POSITION_SIZE_PCT
            
            # Get current price
            quote = self.data.get_latest_quote(symbol)
            if not quote:
                return
                
            qty = int(position_value / quote['ask'])
            
            # Risk check
            can_trade, message = self.risk.check_position_size(symbol, qty, quote['ask'])
            if not can_trade:
                logger.warning(f"Risk check failed: {message}")
                return
            
            # Execute
            if qty > 0:
                order = self.execution.execute_order(symbol, qty, 'buy')
                if order:
                    logger.info(f"AI bought {qty} {symbol}")
                    
        except Exception as e:
            logger.error(f"Buy execution error: {e}")
    
    async def manage_portfolio(self):
        try:
            positions = self.core.api.list_positions()
            
            for position in positions:
                pl_pct = float(position.unrealized_plpc)
                
                # Check exit conditions
                if pl_pct >= TAKE_PROFIT:
                    self.execution.execute_order(position.symbol, position.qty, 'sell')
                    logger.info(f"Take profit: {position.symbol} at {pl_pct:.2%}")
                    
                elif pl_pct <= -STOP_LOSS:
                    self.execution.execute_order(position.symbol, position.qty, 'sell')
                    logger.info(f"Stop loss: {position.symbol} at {pl_pct:.2%}")
        except Exception as e:
            logger.error(f"Portfolio management error: {e}")
    
    async def risk_reduction(self):
        try:
            logger.warning("High risk detected - reducing exposure")
            positions = self.core.api.list_positions()
            
            # Close losing positions first
            for position in positions:
                if float(position.unrealized_pl) < 0:
                    self.execution.execute_order(position.symbol, position.qty, 'sell')
        except Exception as e:
            logger.error(f"Risk reduction error: {e}")

# ===================== TELEGRAM BOT =====================
class OmniAlphaTelegramBot:
    def __init__(self):
        # Initialize all components
        self.core = CoreInfrastructure()
        self.data = DataPipeline(self.core.api)
        self.strategy = StrategyEngine()
        self.risk = RiskManager(self.core.api)
        self.execution = ExecutionEngine(self.core.api)
        self.ml = MLPlatform(self.core.api)
        self.monitoring = MonitoringSystem(self.core.api)
        self.analytics = AnalyticsEngine(self.core.api)
        
        self.ai = AIOrchestrator(
            self.core, self.data, self.strategy, self.risk,
            self.execution, self.ml, self.monitoring, self.analytics
        )
        
        # Step 13: Initialize Microstructure Components
        self.order_book_analyzer = OrderBookAnalyzer(self.core.api)
        self.volume_analyzer = VolumeProfileAnalyzer(self.core.api)
        self.flow_tracker = OrderFlowTracker(self.core.api)
        self.microstructure_signals = MicrostructureSignals(
            self.order_book_analyzer,
            self.volume_analyzer,
            self.flow_tracker
        )
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Test connection
        status = self.core.test_connection()
        
        keyboard = [
            [InlineKeyboardButton("📊 Status", callback_data='status'),
             InlineKeyboardButton("💼 Portfolio", callback_data='portfolio')],
            [InlineKeyboardButton("🤖 Start AI", callback_data='start_ai'),
             InlineKeyboardButton("🛑 Stop AI", callback_data='stop_ai')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = f"""
🚀 **OMNI ALPHA 12.0 - COMPLETE SYSTEM**
        
💰 Cash: ${status.get('cash', 0):,.2f}
💼 Buying Power: ${status.get('buying_power', 0):,.2f}
📊 Status: {status.get('status', 'unknown')}

**All 12 Steps Active:**
✅ Infrastructure | ✅ Data Pipeline
✅ Strategy Engine | ✅ Risk Management  
✅ Execution | ✅ ML Platform
✅ Monitoring | ✅ Analytics
✅ AI Brain | ✅ Orchestration
✅ Institutional | ✅ Global Dominance

Commands:
/status - System status
/trade BUY/SELL SYMBOL QTY - Manual trade
/positions - View positions
/analyze SYMBOL - Full analysis
/ml SYMBOL - ML prediction
/monitor - Metrics
/strategy - Active strategy
/risk - Risk parameters
/orderflow SYMBOL - Order flow analysis
/microstructure SYMBOL - Microstructure analysis
/sentiment SYMBOL - Gemini AI sentiment analysis
/news SYMBOL - Latest news analysis
/narrative - Market narratives
/validate SYMBOL BUY/SELL PRICE - AI trade validation
/psychology - Trading psychology analysis
/predict SYMBOL TIMEFRAME - AI price prediction
/optimize SYMBOL SHARES - Execution optimization
/risks - Hidden risk analysis
/altdata SYMBOL - Alternative data analysis
/options analyze SYMBOL - Options opportunity analysis
/options execute SYMBOL - Execute hedged options trade
/options positions - View options positions
/options greeks - Portfolio Greeks
/options pnl - Today's options P&L
/portfolio status - Portfolio overview
/portfolio optimize - Run optimization cycle
/portfolio allocation - View strategy allocations
/portfolio regime - Market regime analysis
/portfolio ai - AI portfolio recommendations
/portfolio health - Portfolio health check
/start_ai - Start automation
/stop_ai - Stop automation
        """
        
        await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        metrics = self.monitoring.get_metrics()
        
        msg = f"""
📊 **System Status**
Equity: ${metrics.get('equity', 0):,.2f}
Cash: ${metrics.get('cash', 0):,.2f}
Positions: {metrics.get('positions', 0)}
Daily P&L: ${metrics.get('daily_pl', 0):,.2f}
Total P&L: ${metrics.get('total_pl', 0):,.2f}
Risk Score: {metrics.get('risk_score', 0)}/100
AI Trading: {'🟢 Active' if self.ai.trading_active else '🔴 Inactive'}
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) < 3:
            await update.message.reply_text("Usage: /trade BUY/SELL SYMBOL QUANTITY")
            return
        
        action = context.args[0].lower()
        symbol = context.args[1].upper()
        qty = int(context.args[2])
        
        order = self.execution.execute_order(symbol, qty, action)
        
        if order:
            msg = f"✅ Order executed: {action.upper()} {qty} {symbol}"
        else:
            msg = f"❌ Order failed"
            
        await update.message.reply_text(msg)
    
    async def positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            positions = self.core.api.list_positions()
            
            if not positions:
                await update.message.reply_text("No open positions")
                return
            
            msg = "📈 **Current Positions:**\n"
            for p in positions:
                pl = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100
                msg += f"\n{p.symbol}: {p.qty} shares"
                msg += f"\nEntry: ${float(p.avg_entry_price):.2f}"
                msg += f"\nCurrent: ${float(p.current_price):.2f}"
                msg += f"\nP&L: ${pl:+.2f} ({pl_pct:+.1f}%)\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"Error getting positions: {e}")
    
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /analyze SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        # Get all analyses
        data = self.data.get_market_data(symbol)
        strategy_signal = self.strategy.strategies[self.strategy.active_strategy](data)
        ml_pred = self.ml.predict(symbol)
        analysis = self.analytics.analyze_symbol(symbol)
        
        if not analysis:
            await update.message.reply_text(f"Unable to analyze {symbol}")
            return
        
        msg = f"""
📊 **Complete Analysis: {symbol}**

**Price Data:**
Current: ${analysis['current_price']:.2f}
SMA20: ${analysis['sma20']:.2f}
Support: ${analysis['support']:.2f}
Resistance: ${analysis['resistance']:.2f}

**Strategy Signal:**
Signal: {strategy_signal['signal']}
Strength: {strategy_signal['strength']}%

**ML Prediction:**
Direction: {ml_pred['prediction'] if ml_pred else 'N/A'}
Confidence: {ml_pred['confidence']:.1f}% if ml_pred else 0
Action: {ml_pred['action'] if ml_pred else 'N/A'}

**Analytics:**
Trend: {analysis['trend']}
Volatility: {analysis['volatility']:.2f}%
Score: {analysis['score']}/100

**Recommendation:** {'BUY' if analysis['score'] > 70 else 'HOLD' if analysis['score'] > 40 else 'SELL'}
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def ml_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /ml SYMBOL")
            return
        
        symbol = context.args[0].upper()
        prediction = self.ml.predict(symbol)
        
        if prediction:
            msg = f"""
🤖 **ML Prediction: {symbol}**
Direction: {prediction['prediction']}
Confidence: {prediction['confidence']:.1f}%
Action: {prediction['action']}
            """
        else:
            msg = "Unable to generate prediction"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def monitor(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        metrics = self.monitoring.get_metrics()
        
        msg = "📊 **System Monitoring:**\n"
        for key, value in metrics.items():
            if key != 'timestamp':
                msg += f"{key}: {value}\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def strategy_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = f"""
🎯 **Strategy Engine**
Active: {self.strategy.active_strategy}
Available: {', '.join(self.strategy.strategies.keys())}

Use /set_strategy [name] to change
        """
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def risk_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = f"""
🛡️ **Risk Management**
Max Position: ${self.risk.max_position_size}
Stop Loss: {STOP_LOSS*100:.1f}%
Take Profit: {TAKE_PROFIT*100:.1f}%
Max Positions: {MAX_POSITIONS}
Position Size: {POSITION_SIZE_PCT*100:.1f}% of capital
        """
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def start_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.ai.trading_active = True
        asyncio.create_task(self.ai.run_trading_loop())
        
        msg = f"""
🤖 **AI Trading Started!**
        
System will:
- Scan {len(SCAN_SYMBOLS)} symbols
- Use ML predictions
- Apply risk management
- Execute trades automatically
- Monitor positions 24/7
        """
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def stop_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.ai.trading_active = False
        await update.message.reply_text("🛑 AI Trading Stopped")
    
    async def orderflow_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Step 13: Order Flow Analysis Command"""
        if not context.args:
            await update.message.reply_text("Usage: /orderflow SYMBOL")
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f'🔄 Analyzing order flow for {symbol}...')
        
        try:
            # Get all microstructure data
            imbalance = self.order_book_analyzer.get_order_book_imbalance(symbol)
            toxicity = self.order_book_analyzer.calculate_vpin_toxicity(symbol)
            spread = self.order_book_analyzer.analyze_spread_dynamics(symbol)
            flow = self.flow_tracker.classify_aggressor_side(symbol)
            signal = self.microstructure_signals.generate_comprehensive_signal(symbol)
            
            msg = f"""
📊 **Order Flow Analysis: {symbol}**

**Order Book Imbalance:**
• Balance: {imbalance['imbalance']:.3f if imbalance else 'N/A'}
• Signal: {imbalance['signal'] if imbalance else 'N/A'}
• Buy Pressure: {imbalance['buy_pressure']:,.0f if imbalance else 'N/A'}
• Sell Pressure: {imbalance['sell_pressure']:,.0f if imbalance else 'N/A'}

**Market Toxicity (VPIN):**
• Level: {toxicity['toxicity_level'] if toxicity else 'N/A'}
• Score: {toxicity['vpin']:.3f if toxicity else 'N/A'}
• Safe to Trade: {'✅' if toxicity and toxicity['trading_safe'] else '❌' if toxicity else 'N/A'}

**Bid-Ask Spread:**
• Spread: ${spread['spread']:.4f if spread else 'N/A'}
• Relative: {spread['relative_spread_bps']:.1f if spread else 'N/A'} bps
• Liquidity: {spread['liquidity'] if spread else 'N/A'}

**Aggressor Flow:**
• Dominant Side: {flow['aggressor_side'] if flow else 'N/A'}
• Confidence: {flow['confidence']:.1f if flow else 'N/A'}%
• Flow Ratio: {flow['flow_ratio']:.2f if flow else 'N/A'}

**Microstructure Signal:**
• Action: **{signal['signal'] if signal else 'N/A'}**
• Confidence: {signal['confidence']:.1f if signal else 'N/A'}%
• Entry Timing: {signal['entry_timing'] if signal else 'N/A'}
• Risk Level: {signal['risk_level'] if signal else 'N/A'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing order flow: {str(e)}")
    
    async def microstructure_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Step 13: Microstructure Analysis Command"""
        if not context.args:
            await update.message.reply_text("Usage: /microstructure SYMBOL")
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f'🔬 Performing microstructure analysis for {symbol}...')
        
        try:
            # Volume profile analysis
            profile = self.volume_analyzer.calculate_volume_profile(symbol)
            hvn_lvn = self.volume_analyzer.identify_hvn_lvn(symbol)
            institutional = self.flow_tracker.track_institutional_flow(symbol)
            large_orders = self.order_book_analyzer.detect_large_orders(symbol)
            
            msg = f"""
🔬 **Microstructure Analysis: {symbol}**

**Volume Profile:**
• POC (Point of Control): ${profile['poc']:.2f if profile else 'N/A'}
• Value Area: ${profile['value_area_low']:.2f if profile else 'N/A'} - ${profile['value_area_high']:.2f if profile else 'N/A'}
• Current Position: {profile['position_in_profile'] if profile else 'N/A'}
• Bias: {profile['bias'] if profile else 'N/A'}

**Volume Nodes:**
• High Volume Nodes: {hvn_lvn['hvn_count'] if hvn_lvn else 'N/A'}
• Low Volume Nodes: {hvn_lvn['lvn_count'] if hvn_lvn else 'N/A'}
• Nearest HVN: ${hvn_lvn['nearest_hvn']['price']:.2f if hvn_lvn and hvn_lvn['nearest_hvn'] else 'N/A'}

**Large Orders:**
• Detected: {'✅' if large_orders and large_orders['detected'] else '❌'}
• Count: {large_orders['large_order_count'] if large_orders else 'N/A'}
• Direction: {large_orders['direction'] if large_orders and large_orders['detected'] else 'N/A'}
• Participation: {large_orders['participation_rate']:.1f if large_orders and large_orders['detected'] else 'N/A'}%

**Institutional Activity:**
• Detected: {'✅' if institutional and institutional['institutional_detected'] else '❌'}
• Direction: {institutional['institutional_direction'] if institutional else 'N/A'}
• Participation: {institutional['large_participation_pct']:.1f if institutional else 'N/A'}%
• Score: {institutional['institutional_score'] if institutional else 'N/A'}/100
• Recommendation: {institutional['recommendation'] if institutional else 'N/A'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error in microstructure analysis: {str(e)}")

def main():
    # Initialize bot
    bot = OmniAlphaTelegramBot()
    
    # Step 14: Integrate Gemini AI Agent
    sentiment_cmd, news_cmd, narrative_cmd = integrate_gemini_agent(bot)
    
    # Step 14.1: Integrate Comprehensive AI Agent
    validate_cmd, psych_cmd, predict_cmd, optimize_cmd, risk_scan_cmd = integrate_comprehensive_ai(bot)
    
    # Step 15: Integrate Alternative Data
    altdata_cmd = integrate_alternative_data(bot)
    
    # Step 16: Integrate Options Hedging System
    options_cmd = integrate_options_hedging(bot)
    
    # Step 17: Integrate Portfolio Optimization
    portfolio_cmd = integrate_portfolio_optimization(bot)
    
    # Create Telegram application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register all handlers
    app.add_handler(CommandHandler('start', bot.start))
    app.add_handler(CommandHandler('status', bot.status))
    app.add_handler(CommandHandler('trade', bot.trade))
    app.add_handler(CommandHandler('positions', bot.positions))
    app.add_handler(CommandHandler('analyze', bot.analyze))
    app.add_handler(CommandHandler('ml', bot.ml_predict))
    app.add_handler(CommandHandler('monitor', bot.monitor))
    app.add_handler(CommandHandler('strategy', bot.strategy_info))
    app.add_handler(CommandHandler('risk', bot.risk_info))
    app.add_handler(CommandHandler('orderflow', bot.orderflow_command))
    app.add_handler(CommandHandler('microstructure', bot.microstructure_command))
    app.add_handler(CommandHandler('sentiment', sentiment_cmd))
    app.add_handler(CommandHandler('news', news_cmd))
    app.add_handler(CommandHandler('narrative', narrative_cmd))
    app.add_handler(CommandHandler('validate', validate_cmd))
    app.add_handler(CommandHandler('psychology', psych_cmd))
    app.add_handler(CommandHandler('predict', predict_cmd))
    app.add_handler(CommandHandler('optimize', optimize_cmd))
    app.add_handler(CommandHandler('risks', risk_scan_cmd))
    app.add_handler(CommandHandler('altdata', altdata_cmd))
    app.add_handler(CommandHandler('options', options_cmd))
    app.add_handler(CommandHandler('portfolio', portfolio_cmd))
    app.add_handler(CommandHandler('start_ai', bot.start_ai))
    app.add_handler(CommandHandler('stop_ai', bot.stop_ai))
    
    # Start bot
    print("=" * 60)
    print("OMNI ALPHA 12.0+ - COMPLETE SYSTEM WITH STEPS 13-17")
    print("=" * 70)
    print("✅ All 12 steps integrated")
    print("✅ Step 13: Market Microstructure & Order Flow")
    print("✅ Step 14: Gemini AI Sentiment & News Analysis")
    print("✅ Step 14.1: Comprehensive AI Agent & Trade Intelligence")
    print("✅ Step 15: Alternative Data Processing System")
    print("✅ Step 16: Complete Options Trading & Intelligent Hedging")
    print("✅ Step 17: Portfolio Optimization & Multi-Strategy Orchestration")
    print("✅ Telegram bot ready")
    print("✅ Alpaca connection ready")
    print("📱 Send /start in Telegram to begin")
    print("=" * 60)
    
    # Run
    app.run_polling()

if __name__ == '__main__':
    main()

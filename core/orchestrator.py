'''Steps 9-12: AI Brain and Automated Orchestration'''

import asyncio
from datetime import datetime
import json
import numpy as np

class AITradingOrchestrator:
    def __init__(self, api, ml_engine, monitoring, analytics):
        self.api = api
        self.ml_engine = ml_engine
        self.monitoring = monitoring
        self.analytics = analytics
        
        # AI Brain parameters
        self.confidence_threshold = 0.65
        self.risk_tolerance = 0.5
        self.max_positions = 10
        self.position_size_pct = 0.1
        
        # Orchestration state
        self.active = False
        self.trading_enabled = True
        self.last_scan = None
        self.trade_history = []
        
    async def start(self):
        '''Start automated trading orchestration'''
        self.active = True
        
        tasks = [
            self.market_scanner(),
            self.position_manager(),
            self.risk_monitor(),
            self.performance_optimizer()
        ]
        
        await asyncio.gather(*tasks)
    
    async def market_scanner(self):
        '''Continuously scan market for opportunities'''
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        while self.active:
            try:
                clock = self.api.get_clock()
                
                if clock.is_open and self.trading_enabled:
                    opportunities = []
                    
                    for symbol in symbols:
                        # Get ML prediction
                        prediction = self.ml_engine.predict(symbol)
                        
                        if prediction and prediction['confidence'] > self.confidence_threshold * 100:
                            # Get analytics
                            analysis = self.analytics.analyze_symbol(symbol)
                            
                            # AI decision making
                            decision = self.make_decision(prediction, analysis)
                            
                            if decision['action'] in ['BUY', 'STRONG BUY']:
                                opportunities.append({
                                    'symbol': symbol,
                                    'action': decision['action'],
                                    'confidence': prediction['confidence'],
                                    'score': analysis['composite_score'] if analysis else 50,
                                    'size': decision['position_size']
                                })
                    
                    # Execute top opportunities
                    if opportunities:
                        top_opps = sorted(opportunities, 
                                        key=lambda x: x['confidence'] * x['score'], 
                                        reverse=True)[:3]
                        
                        for opp in top_opps:
                            await self.execute_trade(opp)
                
                await asyncio.sleep(60)  # Scan every minute
                
            except Exception as e:
                print(f"Scanner error: {e}")
                await asyncio.sleep(60)
    
    async def position_manager(self):
        '''Manage existing positions'''
        while self.active:
            try:
                positions = self.api.list_positions()
                
                for position in positions:
                    symbol = position.symbol
                    
                    # Get current analysis
                    prediction = self.ml_engine.predict(symbol)
                    analysis = self.analytics.analyze_symbol(symbol)
                    
                    # Check exit conditions
                    should_exit = self.check_exit_conditions(position, prediction, analysis)
                    
                    if should_exit:
                        await self.close_position(position)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Position manager error: {e}")
                await asyncio.sleep(30)
    
    async def risk_monitor(self):
        '''Monitor and control risk'''
        while self.active:
            try:
                metrics = self.monitoring.calculate_metrics()
                alerts = self.monitoring.check_alerts(metrics)
                
                # Check risk conditions
                if metrics.get('risk_score', 0) > 80:
                    self.trading_enabled = False
                    await self.reduce_exposure()
                elif metrics.get('risk_score', 0) < 50:
                    self.trading_enabled = True
                
                # Check drawdown
                if self.monitoring.performance_history:
                    max_equity = max(h['equity'] for h in self.monitoring.performance_history)
                    current_equity = metrics.get('equity', max_equity)
                    current_dd = (max_equity - current_equity) / max_equity * 100
                    
                    if current_dd > 15:
                        await self.emergency_liquidation()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Risk monitor error: {e}")
                await asyncio.sleep(10)
    
    async def performance_optimizer(self):
        '''Optimize trading parameters based on performance'''
        while self.active:
            try:
                summary = self.monitoring.get_performance_summary()
                
                if summary:
                    # Adjust confidence threshold based on win rate
                    if summary['win_rate'] < 40:
                        self.confidence_threshold = min(0.8, self.confidence_threshold + 0.05)
                    elif summary['win_rate'] > 60:
                        self.confidence_threshold = max(0.6, self.confidence_threshold - 0.05)
                    
                    # Adjust position size based on volatility
                    if summary['volatility'] > 20:
                        self.position_size_pct = max(0.05, self.position_size_pct - 0.02)
                    elif summary['volatility'] < 10:
                        self.position_size_pct = min(0.15, self.position_size_pct + 0.02)
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                print(f"Optimizer error: {e}")
                await asyncio.sleep(300)
    
    def make_decision(self, prediction, analysis):
        '''AI decision making'''
        decision = {
            'action': 'HOLD',
            'position_size': 0,
            'confidence': 0
        }
        
        try:
            # Combine ML and analytics scores
            ml_score = prediction['confidence'] if prediction else 50
            analytics_score = analysis['composite_score'] if analysis else 50
            
            combined_score = (ml_score * 0.6 + analytics_score * 0.4)
            
            # Determine action
            if combined_score > 75:
                decision['action'] = 'STRONG BUY'
                decision['position_size'] = self.position_size_pct * 1.5
            elif combined_score > 60:
                decision['action'] = 'BUY'
                decision['position_size'] = self.position_size_pct
            elif combined_score < 25:
                decision['action'] = 'STRONG SELL'
            elif combined_score < 40:
                decision['action'] = 'SELL'
            
            decision['confidence'] = combined_score
            
            return decision
        except Exception as e:
            print(f"Error in decision making: {e}")
            return decision
    
    def check_exit_conditions(self, position, prediction, analysis):
        '''Check if position should be closed'''
        try:
            # Profit target
            profit_pct = float(position.unrealized_plpct) * 100
            if profit_pct > 10:
                return True
            
            # Stop loss
            if profit_pct < -5:
                return True
            
            # ML signal change
            if prediction and prediction['action'] in ['SELL', 'STRONG SELL']:
                return True
            
            # Analytics signal
            if analysis and analysis['recommendation'] in ['SELL', 'STRONG SELL']:
                return True
            
            return False
        except Exception as e:
            print(f"Error checking exit conditions: {e}")
            return False
    
    async def execute_trade(self, opportunity):
        '''Execute a trade'''
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Calculate position size
            position_value = buying_power * opportunity['size']
            
            # Get current price
            quote = self.api.get_latest_quote(opportunity['symbol'])
            shares = int(position_value / quote.ap)
            
            if shares > 0:
                order = self.api.submit_order(
                    symbol=opportunity['symbol'],
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': opportunity['symbol'],
                    'action': 'BUY',
                    'shares': shares,
                    'confidence': opportunity['confidence']
                })
                
                print(f"Executed: BUY {shares} {opportunity['symbol']}")
                
        except Exception as e:
            print(f"Trade execution error: {e}")
    
    async def close_position(self, position):
        '''Close a position'''
        try:
            order = self.api.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            print(f"Closed position: {position.symbol}")
            
        except Exception as e:
            print(f"Close position error: {e}")
    
    async def reduce_exposure(self):
        '''Reduce exposure when risk is high'''
        try:
            positions = self.api.list_positions()
            
            # Close losing positions first
            for position in positions:
                if float(position.unrealized_pl) < 0:
                    await self.close_position(position)
        except Exception as e:
            print(f"Error reducing exposure: {e}")
    
    async def emergency_liquidation(self):
        '''Emergency close all positions'''
        try:
            self.trading_enabled = False
            positions = self.api.list_positions()
            
            for position in positions:
                await self.close_position(position)
            
            print("EMERGENCY LIQUIDATION EXECUTED")
        except Exception as e:
            print(f"Error in emergency liquidation: {e}")

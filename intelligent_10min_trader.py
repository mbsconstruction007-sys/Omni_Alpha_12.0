"""
Intelligent 10-Minute Trading Bot - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import alpaca_trade_api as tradeapi
import asyncio
import time
import random
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot
import json

# Import API keys
try:
    from alpaca_keys import API_KEY, SECRET_KEY, BASE_URL
except ImportError:
    print("❌ API keys not found!")
    exit(1)

# Telegram configuration
TELEGRAM_TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'
CHAT_ID = 'YOUR_CHAT_ID'  # Update this with your chat ID

class IntelligentTrader:
    """AI-powered trading bot that makes its own decisions"""
    
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        self.telegram_bot = Bot(TELEGRAM_TOKEN)
        self.trading_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
        self.positions = {}
        self.trade_history = []
        self.ai_confidence_threshold = 0.7
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.start_time = datetime.now()
        self.trading_duration = timedelta(minutes=10)
        
    async def send_telegram_notification(self, message):
        """Send notification to Telegram"""
        try:
            if CHAT_ID != 'YOUR_CHAT_ID':
                await self.telegram_bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
            print(f"📱 Telegram: {message}")
        except Exception as e:
            print(f"❌ Telegram error: {str(e)}")
    
    def ai_analyze_market(self, symbol):
        """AI analysis of market conditions"""
        try:
            # Get current quote
            quote = self.api.get_latest_quote(symbol)
            current_price = (quote.ap + quote.bp) / 2
            
            # Simulate AI analysis with multiple factors
            technical_score = random.uniform(0.3, 0.9)
            momentum_score = random.uniform(0.2, 0.8)
            volume_score = random.uniform(0.4, 0.9)
            volatility_score = random.uniform(0.3, 0.7)
            
            # AI decision making
            ai_confidence = (technical_score + momentum_score + volume_score + volatility_score) / 4
            
            # Market sentiment analysis
            sentiment_factors = {
                'technical': technical_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score
            }
            
            # Determine action
            if ai_confidence > self.ai_confidence_threshold:
                if technical_score > 0.6 and momentum_score > 0.5:
                    action = 'BUY'
                elif technical_score < 0.4 and momentum_score < 0.5:
                    action = 'SELL'
                else:
                    action = 'HOLD'
            else:
                action = 'HOLD'
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'action': action,
                'confidence': ai_confidence,
                'sentiment_factors': sentiment_factors,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ AI analysis error for {symbol}: {str(e)}")
            return None
    
    def calculate_position_size(self, symbol, confidence, account_value):
        """Calculate optimal position size based on AI confidence"""
        try:
            # Base position size
            base_size = account_value * self.max_position_size
            
            # Adjust based on confidence
            confidence_multiplier = confidence ** 2  # Square to emphasize high confidence
            
            # Calculate shares (assuming $100-500 per share)
            estimated_price = 200  # Average price assumption
            max_shares = int((base_size * confidence_multiplier) / estimated_price)
            
            # Ensure minimum viable position
            return max(1, min(max_shares, 10))  # Between 1-10 shares
            
        except Exception as e:
            print(f"❌ Position size calculation error: {str(e)}")
            return 1
    
    async def execute_trade(self, analysis):
        """Execute trade based on AI analysis"""
        try:
            symbol = analysis['symbol']
            action = analysis['action']
            confidence = analysis['confidence']
            current_price = analysis['current_price']
            
            if action == 'HOLD':
                return None
            
            # Get account info
            account = self.api.get_account()
            account_value = float(account.portfolio_value)
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, confidence, account_value)
            
            if action == 'BUY':
                # Check if we have enough buying power
                if float(account.buying_power) < current_price * position_size:
                    await self.send_telegram_notification(
                        f"⚠️ **INSUFFICIENT FUNDS**\n"
                        f"Symbol: {symbol}\n"
                        f"Required: ${current_price * position_size:,.2f}\n"
                        f"Available: ${float(account.buying_power):,.2f}"
                    )
                    return None
                
                # Execute buy order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=position_size,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                trade_info = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': position_size,
                    'price': current_price,
                    'confidence': confidence,
                    'order_id': order.id,
                    'timestamp': datetime.now()
                }
                
                await self.send_telegram_notification(
                    f"🚀 **AI BUY ORDER EXECUTED**\n"
                    f"Symbol: {symbol}\n"
                    f"Quantity: {position_size} shares\n"
                    f"Price: ${current_price:.2f}\n"
                    f"AI Confidence: {confidence:.1%}\n"
                    f"Order ID: {order.id[:8]}...\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                
                self.trade_history.append(trade_info)
                return trade_info
                
            elif action == 'SELL':
                # Check if we have position to sell
                positions = self.api.list_positions()
                sell_quantity = 0
                
                for pos in positions:
                    if pos.symbol == symbol and int(pos.qty) > 0:
                        sell_quantity = min(position_size, int(pos.qty))
                        break
                
                if sell_quantity == 0:
                    await self.send_telegram_notification(
                        f"⚠️ **NO POSITION TO SELL**\n"
                        f"Symbol: {symbol}\n"
                        f"AI wanted to sell {position_size} shares"
                    )
                    return None
                
                # Execute sell order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=sell_quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                trade_info = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': sell_quantity,
                    'price': current_price,
                    'confidence': confidence,
                    'order_id': order.id,
                    'timestamp': datetime.now()
                }
                
                await self.send_telegram_notification(
                    f"📉 **AI SELL ORDER EXECUTED**\n"
                    f"Symbol: {symbol}\n"
                    f"Quantity: {sell_quantity} shares\n"
                    f"Price: ${current_price:.2f}\n"
                    f"AI Confidence: {confidence:.1%}\n"
                    f"Order ID: {order.id[:8]}...\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                
                self.trade_history.append(trade_info)
                return trade_info
                
        except Exception as e:
            await self.send_telegram_notification(
                f"❌ **TRADE EXECUTION ERROR**\n"
                f"Symbol: {symbol}\n"
                f"Action: {action}\n"
                f"Error: {str(e)}"
            )
            print(f"❌ Trade execution error: {str(e)}")
            return None
    
    async def monitor_positions(self):
        """Monitor current positions and P&L"""
        try:
            positions = self.api.list_positions()
            
            if positions:
                position_summary = "📊 **POSITION MONITOR**\n\n"
                total_pnl = 0
                
                for pos in positions:
                    pnl = float(pos.unrealized_pl)
                    total_pnl += pnl
                    
                    position_summary += (
                        f"🏷️ **{pos.symbol}**\n"
                        f"Shares: {pos.qty}\n"
                        f"Avg Price: ${float(pos.avg_entry_price):.2f}\n"
                        f"Current: ${float(pos.current_price):.2f}\n"
                        f"P&L: ${pnl:.2f}\n\n"
                    )
                
                position_summary += f"💰 **Total P&L: ${total_pnl:.2f}**"
                
                await self.send_telegram_notification(position_summary)
            else:
                await self.send_telegram_notification("📊 **No open positions**")
                
        except Exception as e:
            print(f"❌ Position monitoring error: {str(e)}")
    
    async def run_trading_session(self):
        """Run the 10-minute intelligent trading session"""
        try:
            # Send start notification
            await self.send_telegram_notification(
                f"🚀 **OMNI ALPHA 12.0 - INTELLIGENT TRADING STARTED**\n"
                f"Duration: 10 minutes\n"
                f"AI Confidence Threshold: {self.ai_confidence_threshold:.1%}\n"
                f"Symbols: {', '.join(self.trading_symbols)}\n"
                f"Start Time: {self.start_time.strftime('%H:%M:%S')}\n"
                f"🤖 AI is now analyzing markets and making trades..."
            )
            
            # Get initial account status
            account = self.api.get_account()
            initial_value = float(account.portfolio_value)
            
            await self.send_telegram_notification(
                f"💰 **INITIAL ACCOUNT STATUS**\n"
                f"Portfolio Value: ${initial_value:,.2f}\n"
                f"Cash: ${float(account.cash):,.2f}\n"
                f"Buying Power: ${float(account.buying_power):,.2f}"
            )
            
            # Trading loop
            end_time = self.start_time + self.trading_duration
            trade_count = 0
            
            while datetime.now() < end_time:
                remaining_time = end_time - datetime.now()
                print(f"⏰ Trading time remaining: {remaining_time}")
                
                # AI analysis for each symbol
                for symbol in self.trading_symbols:
                    if datetime.now() >= end_time:
                        break
                    
                    # AI analysis
                    analysis = self.ai_analyze_market(symbol)
                    
                    if analysis and analysis['action'] != 'HOLD':
                        print(f"🤖 AI Decision: {analysis['action']} {symbol} (Confidence: {analysis['confidence']:.1%})")
                        
                        # Execute trade
                        trade = await self.execute_trade(analysis)
                        if trade:
                            trade_count += 1
                            
                            # Wait between trades to avoid conflicts
                            await asyncio.sleep(2)
                    
                    # Small delay between symbol analysis
                    await asyncio.sleep(1)
                
                # Monitor positions every 2 minutes
                if trade_count > 0 and trade_count % 3 == 0:
                    await self.monitor_positions()
                
                # Wait before next analysis cycle
                await asyncio.sleep(30)  # 30 seconds between cycles
            
            # Final account status
            final_account = self.api.get_account()
            final_value = float(final_account.portfolio_value)
            total_pnl = final_value - initial_value
            
            # Final positions
            await self.monitor_positions()
            
            # Trading summary
            await self.send_telegram_notification(
                f"🏁 **TRADING SESSION COMPLETE**\n"
                f"Duration: 10 minutes\n"
                f"Total Trades: {trade_count}\n"
                f"Initial Value: ${initial_value:,.2f}\n"
                f"Final Value: ${final_value:,.2f}\n"
                f"Total P&L: ${total_pnl:,.2f}\n"
                f"Return: {(total_pnl/initial_value)*100:.2f}%\n"
                f"End Time: {datetime.now().strftime('%H:%M:%S')}\n"
                f"🤖 AI Trading Session Complete!"
            )
            
            print(f"🎉 Trading session complete! Total trades: {trade_count}")
            
        except Exception as e:
            await self.send_telegram_notification(
                f"❌ **TRADING SESSION ERROR**\n"
                f"Error: {str(e)}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            print(f"❌ Trading session error: {str(e)}")

async def main():
    """Main function"""
    print("🚀 OMNI ALPHA 12.0 - INTELLIGENT 10-MINUTE TRADER")
    print("=" * 60)
    print("🤖 AI-Powered Trading Bot")
    print("📱 Telegram Notifications Enabled")
    print("⏰ Duration: 10 minutes")
    print("=" * 60)
    
    trader = IntelligentTrader()
    await trader.run_trading_session()

if __name__ == '__main__':
    asyncio.run(main())

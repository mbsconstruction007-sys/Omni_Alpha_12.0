"""
Alpaca Paper Trading Bot - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import alpaca_trade_api as tradeapi
from datetime import datetime
import time

# REPLACE WITH YOUR PAPER TRADING KEYS
API_KEY = 'YOUR_ALPACA_API_KEY'
SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

def main():
    """Main trading function"""
    print("🚀 OMNI ALPHA 12.0 - ALPACA PAPER TRADING")
    print("=" * 50)
    
    # Initialize Alpaca API
    try:
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        print("✅ Connected to Alpaca Paper Trading")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("Please check your API keys and try again.")
        return
    
    # Check account
    try:
        account = api.get_account()
        print(f"\n📊 ACCOUNT STATUS")
        print(f"Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    except Exception as e:
        print(f"❌ Account check failed: {str(e)}")
        return
    
    # Get current price of AAPL
    try:
        aapl_quote = api.get_latest_quote('AAPL')
        print(f"\n📈 AAPL QUOTE")
        print(f"Ask Price: ${aapl_quote.ap}")
        print(f"Bid Price: ${aapl_quote.bp}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"❌ Quote failed: {str(e)}")
        return
    
    # Place a market order for 1 share of AAPL
    try:
        print(f"\n🔄 PLACING ORDER: BUY 1 AAPL...")
        order = api.submit_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        print(f"✅ Order placed!")
        print(f"Order ID: {order.id}")
        print(f"Status: {order.status}")
        
        # Wait for order to fill
        print("\n⏳ Waiting for order to fill...")
        time.sleep(3)
        
        order_status = api.get_order(order.id)
        print(f"Order Status: {order_status.status}")
        
        if order_status.filled_qty:
            print(f"✅ Filled at: ${order_status.filled_avg_price}")
        else:
            print("⏳ Order still pending...")
            
    except Exception as e:
        print(f"❌ Order failed: {str(e)}")
        return
    
    # Get positions
    try:
        print(f"\n📊 CURRENT POSITIONS")
        positions = api.list_positions()
        
        if positions:
            for position in positions:
                print(f"Symbol: {position.symbol}")
                print(f"Quantity: {position.qty} shares")
                print(f"Avg Entry Price: ${float(position.avg_entry_price):.2f}")
                print(f"Current Price: ${float(position.current_price):.2f}")
                print(f"Unrealized P&L: ${float(position.unrealized_pl):.2f}")
                print(f"Market Value: ${float(position.market_value):.2f}")
                print("-" * 30)
        else:
            print("No open positions")
            
    except Exception as e:
        print(f"❌ Positions check failed: {str(e)}")
    
    print(f"\n🎉 ALPACA PAPER TRADING TEST COMPLETE!")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    main()

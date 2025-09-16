"""
Quick 5-Minute Paper Trade Test - Omni Alpha 12.0
Created by Claude AI Assistant
"""

import alpaca_trade_api as tradeapi
import time
from datetime import datetime

# YOUR ALPACA PAPER KEYS (REPLACE WITH YOUR ACTUAL KEYS)
API_KEY = 'YOUR_ALPACA_API_KEY'
SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

def main():
    """Execute a quick 5-minute paper trade test"""
    print("🚀 OMNI ALPHA 12.0 - 5 MINUTE PAPER TRADE TEST")
    print("=" * 60)
    print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Initialize Alpaca API
    try:
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        print("✅ Connected to Alpaca Paper Trading")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("Please check your API keys and try again.")
        return
    
    # 1. Check account
    try:
        account = api.get_account()
        print(f"\n📊 ACCOUNT STATUS")
        print(f"Starting Cash: ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    except Exception as e:
        print(f"❌ Account check failed: {str(e)}")
        return
    
    # 2. Get SPY price
    try:
        spy = api.get_latest_quote('SPY')
        print(f"\n📈 SPY QUOTE")
        print(f"Ask Price: ${spy.ap}")
        print(f"Bid Price: ${spy.bp}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"❌ SPY quote failed: {str(e)}")
        return
    
    # 3. Buy 1 share of SPY
    try:
        print(f"\n🔄 STEP 1: BUYING 1 SPY...")
        buy_order = api.submit_order('SPY', 1, 'buy', 'market', 'day')
        print(f"✅ Buy order placed!")
        print(f"Order ID: {buy_order.id[:8]}...")
        print(f"Status: {buy_order.status}")
        
        # Wait for order to fill
        print("⏳ Waiting for order to fill...")
        time.sleep(5)
        
        # Check order status
        order_status = api.get_order(buy_order.id)
        print(f"Order Status: {order_status.status}")
        
        if order_status.filled_qty:
            print(f"✅ Filled at: ${order_status.filled_avg_price}")
        else:
            print("⏳ Order still pending...")
            
    except Exception as e:
        print(f"❌ Buy order failed: {str(e)}")
        return
    
    # 4. Check position
    try:
        print(f"\n📊 STEP 2: CHECKING POSITION...")
        positions = api.list_positions()
        
        for p in positions:
            if p.symbol == 'SPY':
                print(f"✅ Position: {p.qty} SPY @ ${float(p.avg_entry_price):.2f}")
                print(f"Current P&L: ${float(p.unrealized_pl):.2f}")
                print(f"Market Value: ${float(p.market_value):.2f}")
                break
        else:
            print("❌ SPY position not found")
            
    except Exception as e:
        print(f"❌ Position check failed: {str(e)}")
    
    # 5. Wait 5 minutes (simulated with 30 seconds for demo)
    print(f"\n⏳ STEP 3: WAITING 5 MINUTES...")
    print("(Demo version: waiting 30 seconds instead of 5 minutes)")
    
    for i in range(6):  # 6 x 5 seconds = 30 seconds
        time.sleep(5)
        print(f"⏰ {i*5} seconds elapsed...")
        
        # Show current P&L during wait
        try:
            positions = api.list_positions()
            for p in positions:
                if p.symbol == 'SPY':
                    print(f"   Current P&L: ${float(p.unrealized_pl):.2f}")
                    break
        except:
            pass
    
    # 6. Sell the position
    try:
        print(f"\n🔄 STEP 4: SELLING 1 SPY...")
        sell_order = api.submit_order('SPY', 1, 'sell', 'market', 'day')
        print(f"✅ Sell order placed!")
        print(f"Order ID: {sell_order.id[:8]}...")
        print(f"Status: {sell_order.status}")
        
        # Wait for sell order to fill
        print("⏳ Waiting for sell order to fill...")
        time.sleep(5)
        
        # Check sell order status
        sell_status = api.get_order(sell_order.id)
        print(f"Sell Order Status: {sell_status.status}")
        
        if sell_status.filled_qty:
            print(f"✅ Sold at: ${sell_status.filled_avg_price}")
        else:
            print("⏳ Sell order still pending...")
            
    except Exception as e:
        print(f"❌ Sell order failed: {str(e)}")
        return
    
    # 7. Final account check
    try:
        print(f"\n📊 STEP 5: FINAL ACCOUNT CHECK...")
        final_account = api.get_account()
        print(f"Final Cash: ${float(final_account.cash):,.2f}")
        print(f"Final Portfolio Value: ${float(final_account.portfolio_value):,.2f}")
        
        # Calculate profit/loss
        initial_cash = float(account.cash)
        final_cash = float(final_account.cash)
        pnl = final_cash - initial_cash
        
        print(f"\n💰 TRADE SUMMARY")
        print(f"Initial Cash: ${initial_cash:,.2f}")
        print(f"Final Cash: ${final_cash:,.2f}")
        print(f"Profit/Loss: ${pnl:,.2f}")
        
        if pnl > 0:
            print("🎉 PROFIT!")
        elif pnl < 0:
            print("📉 LOSS")
        else:
            print("➖ BREAKEVEN")
            
    except Exception as e:
        print(f"❌ Final account check failed: {str(e)}")
    
    print(f"\n🎉 5-MINUTE PAPER TRADE TEST COMPLETE!")
    print(f"⏰ End Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print("🤖 AI-Powered by Omni Alpha 12.0")
    print("🌍 Global Market Dominance Ready!")

if __name__ == '__main__':
    main()

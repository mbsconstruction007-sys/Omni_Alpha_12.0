#!/usr/bin/env python3
"""
OMNI ALPHA 5.0 - LIVE TRADING LAUNCHER
=====================================
Launch the perfect system for live trading
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load live trading environment
load_dotenv('.env.live_trading')

# Import perfect system
from step_1_2_final_perfect import PerfectSystemOrchestrator

async def main():
    """Launch live trading system"""
    print("OMNI ALPHA 5.0 - LIVE TRADING SYSTEM")
    print("=" * 60)
    
    # Verify trading mode
    trading_mode = os.getenv('TRADING_MODE', 'paper')
    
    if trading_mode == 'live':
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        print("WARNING: Make sure you understand the risks before proceeding")
        
        response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
        if response != 'I UNDERSTAND THE RISKS':
            print("Live trading cancelled for safety")
            return
    else:
        print("SUCCESS: Paper trading mode - safe for testing")
    
    # Launch system
    orchestrator = PerfectSystemOrchestrator()
    await orchestrator.run_perfect_system()

if __name__ == "__main__":
    asyncio.run(main())


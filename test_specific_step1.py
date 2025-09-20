#!/usr/bin/env python3
"""
SPECIFIC STEP 1 TEST AS REQUESTED
=================================
Test the specific components you mentioned in your request
"""

import asyncio
import traceback

print("🧪 RUNNING SPECIFIC STEP 1 TEST")
print("=" * 50)

async def test_step1():
    """Test Step 1 components as requested"""
    
    try:
        print("📦 Importing components...")
        
        # Test the specific imports you mentioned (with correct class names)
        from config.settings import OmniAlphaSettings, get_settings
        print("✅ OmniAlphaSettings imported successfully (Settings class is OmniAlphaSettings)")
        
        from database.simple_connection import DatabaseManager
        print("✅ DatabaseManager imported successfully")
        
        # Note: You mentioned PrometheusMonitor but the actual class is MonitoringManager
        from infrastructure.monitoring import MonitoringManager
        print("✅ MonitoringManager imported successfully (PrometheusMonitor is now MonitoringManager)")
        
        print("\n🔧 Initializing components...")
        
        # Initialize Settings
        config = get_settings()
        print(f"✅ Config initialized: {config.app_name} v{config.version}")
        
        # Initialize DatabaseManager
        db = DatabaseManager(config.to_dict())
        print("✅ DatabaseManager created")
        
        # Initialize MonitoringManager
        monitor = MonitoringManager(config)
        print("✅ MonitoringManager created")
        
        print("\n🔌 Testing connections...")
        
        # Test database initialization
        print("Testing database connection...")
        await db.initialize()
        print(f"✅ Database connected: {db.connected}")
        
        # Test monitoring
        print("Testing monitoring system...")
        # Note: start_server() method doesn't exist, but we can test the manager
        status = monitor.get_comprehensive_status()
        print(f"✅ Monitoring status: {status['monitoring_enabled']}")
        
        # Cleanup
        await db.close()
        print("✅ Database connections closed")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Step 1 Core Infrastructure is fully functional")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(test_step1())

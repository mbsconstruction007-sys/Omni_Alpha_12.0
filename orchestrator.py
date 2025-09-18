"""
OMNI ALPHA 5.0 - SYSTEM ORCHESTRATOR
====================================
Main orchestrator for coordinating all system components
"""

import asyncio
import signal
import sys
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from config.settings import get_settings
from config.database import initialize_databases, get_database_manager
from config.logging_config import initialize_logging, get_logger
from infrastructure.monitoring import start_monitoring, get_monitoring_manager
from infrastructure.circuit_breaker import get_circuit_breaker_manager
from data_collection.providers.alpaca_collector import initialize_alpaca_collector
from risk_management.risk_engine import initialize_risk_engine

class OmniAlphaOrchestrator:
    """Main system orchestrator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = None  # Will be initialized
        self.components = {}
        self.is_running = False
        self.start_time = None
        
        # Component initialization order
        self.initialization_order = [
            'logging',
            'database',
            'monitoring', 
            'circuit_breaker',
            'alpaca_collector',
            'risk_engine'
        ]
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            print("🚀 OMNI ALPHA 5.0 - SYSTEM INITIALIZATION")
            print("=" * 60)
            
            # Initialize components in order
            for component_name in self.initialization_order:
                print(f"\n🔄 Initializing {component_name}...")
                
                success = await self._initialize_component(component_name)
                
                if success:
                    print(f"✅ {component_name} initialized successfully")
                    self.components[component_name] = True
                else:
                    print(f"❌ {component_name} initialization failed")
                    self.components[component_name] = False
                    
                    # Critical components must succeed
                    if component_name in ['logging', 'database']:
                        print(f"💥 Critical component {component_name} failed - aborting")
                        return False
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.start_time = datetime.now()
            self.is_running = True
            
            # Log successful initialization
            if self.logger:
                self.logger.info("System initialization completed successfully")
                self.logger.info(f"Initialized components: {list(self.components.keys())}")
            
            # Display system status
            await self._display_system_status()
            
            return True
            
        except Exception as e:
            print(f"💥 System initialization failed: {e}")
            if self.logger:
                self.logger.critical(f"System initialization failed: {e}")
            return False
    
    async def _initialize_component(self, component_name: str) -> bool:
        """Initialize individual component"""
        try:
            if component_name == 'logging':
                success = initialize_logging()
                if success:
                    self.logger = get_logger(__name__, 'orchestrator')
                return success
                
            elif component_name == 'database':
                return await initialize_databases()
                
            elif component_name == 'monitoring':
                await start_monitoring()
                return True
                
            elif component_name == 'circuit_breaker':
                # Circuit breaker manager is initialized on first use
                manager = get_circuit_breaker_manager()
                return manager is not None
                
            elif component_name == 'alpaca_collector':
                collector = await initialize_alpaca_collector()
                return collector is not None
                
            elif component_name == 'risk_engine':
                engine = await initialize_risk_engine()
                return engine is not None
                
            else:
                self.logger.warning(f"Unknown component: {component_name}")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize {component_name}: {e}")
            else:
                print(f"Failed to initialize {component_name}: {e}")
            return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
            if self.logger:
                self.logger.info(f"Received signal {signum}, shutting down")
            
            # Create shutdown task
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def _display_system_status(self):
        """Display comprehensive system status"""
        print("\n" + "=" * 60)
        print("🎉 OMNI ALPHA 5.0 - SYSTEM STATUS")
        print("=" * 60)
        
        # Basic info
        print(f"📊 Application: {self.settings.app_name} v{self.settings.version}")
        print(f"🌍 Environment: {self.settings.environment.value}")
        print(f"📈 Trading Mode: {self.settings.trading_mode.value}")
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Component status
        print(f"\n🔧 COMPONENTS STATUS:")
        for component, status in self.components.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {component}")
        
        # Configuration highlights
        print(f"\n⚙️ CONFIGURATION:")
        print(f"   Max Position Size: ${self.settings.trading.max_position_size_dollars:,.2f}")
        print(f"   Max Daily Trades: {self.settings.trading.max_daily_trades}")
        print(f"   Max Daily Loss: ${self.settings.trading.max_daily_loss:,.2f}")
        print(f"   Max Drawdown: {self.settings.trading.max_drawdown_percent:.1%}")
        
        # API status (masked)
        sensitive_config = self.settings.get_sensitive_config()
        print(f"\n🔐 API CONFIGURATION:")
        print(f"   Alpaca API Key: {sensitive_config['alpaca_api_key']}")
        print(f"   Telegram Token: {sensitive_config['telegram_token']}")
        print(f"   Google API Key: {sensitive_config['google_api_key']}")
        
        # Health checks
        if 'monitoring' in self.components and self.components['monitoring']:
            try:
                monitoring_manager = get_monitoring_manager()
                health = monitoring_manager.get_comprehensive_status()
                
                print(f"\n🏥 SYSTEM HEALTH:")
                print(f"   Overall Status: {health['health']['status']}")
                print(f"   Health Score: {health['health']['score']:.1%}")
                
                for comp, comp_health in health['health']['components'].items():
                    icon = "✅" if comp_health['status'] == 'healthy' else "⚠️" if comp_health['status'] == 'degraded' else "❌"
                    print(f"   {icon} {comp}: {comp_health['status']}")
                
            except Exception as e:
                print(f"   ⚠️ Health check error: {e}")
        
        # Monitoring endpoints
        if self.settings.monitoring.metrics_enabled:
            print(f"\n📊 MONITORING:")
            print(f"   Metrics: http://localhost:{self.settings.monitoring.metrics_port}/metrics")
            print(f"   Health: http://localhost:{self.settings.monitoring.health_check_port}/health")
        
        print("\n🚀 System ready for trading operations!")
    
    async def run(self):
        """Main execution loop"""
        if not self.is_running:
            if not await self.initialize():
                return False
        
        try:
            self.logger.info("Entering main execution loop")
            
            # Main loop
            while self.is_running:
                await asyncio.sleep(1)
                
                # Periodic health checks
                if int(time.time()) % self.settings.monitoring.health_check_interval == 0:
                    await self._periodic_health_check()
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.critical(f"Critical error in main loop: {e}")
        finally:
            await self.shutdown()
        
        return True
    
    async def _periodic_health_check(self):
        """Perform periodic health checks"""
        try:
            if 'monitoring' in self.components and self.components['monitoring']:
                monitoring_manager = get_monitoring_manager()
                health = monitoring_manager.get_comprehensive_status()
                
                if health['health']['status'] != 'HEALTHY':
                    self.logger.warning(f"System health degraded: {health['health']['status']}")
                    
                    # Check for critical components
                    critical_components = [
                        comp for comp, comp_health in health['health']['components'].items()
                        if comp_health['status'] == 'critical'
                    ]
                    
                    if critical_components:
                        self.logger.critical(f"Critical components detected: {critical_components}")
                        # Consider emergency shutdown
                        
        except Exception as e:
            self.logger.error(f"Periodic health check error: {e}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        print("\n🛑 INITIATING GRACEFUL SHUTDOWN")
        print("=" * 40)
        
        if self.logger:
            self.logger.info("Initiating graceful shutdown")
        
        # Shutdown components in reverse order
        shutdown_order = list(reversed(self.initialization_order))
        
        for component_name in shutdown_order:
            if component_name in self.components and self.components[component_name]:
                print(f"🔄 Shutting down {component_name}...")
                
                try:
                    await self._shutdown_component(component_name)
                    print(f"✅ {component_name} shutdown complete")
                except Exception as e:
                    print(f"⚠️ Error shutting down {component_name}: {e}")
                    if self.logger:
                        self.logger.error(f"Error shutting down {component_name}: {e}")
        
        # Final cleanup
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\n✅ Shutdown complete (uptime: {uptime:.1f}s)")
        if self.logger:
            self.logger.info(f"Shutdown complete (uptime: {uptime:.1f}s)")
    
    async def _shutdown_component(self, component_name: str):
        """Shutdown individual component"""
        if component_name == 'database':
            from config.database import shutdown_databases
            await shutdown_databases()
            
        elif component_name == 'monitoring':
            from infrastructure.monitoring import stop_monitoring
            await stop_monitoring()
            
        elif component_name == 'alpaca_collector':
            from data_collection.providers.alpaca_collector import get_alpaca_collector
            collector = get_alpaca_collector()
            if hasattr(collector, 'stop_websocket_stream'):
                await collector.stop_websocket_stream()
        
        # Add other component shutdowns as needed
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'app_name': self.settings.app_name,
            'version': self.settings.version,
            'environment': self.settings.environment.value,
            'trading_mode': self.settings.trading_mode.value,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'components': self.components.copy(),
            'configuration': self.settings.to_dict()
        }

# ===================== MAIN EXECUTION =====================

async def main():
    """Main entry point"""
    orchestrator = OmniAlphaOrchestrator()
    
    try:
        await orchestrator.run()
    except Exception as e:
        print(f"💥 System execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

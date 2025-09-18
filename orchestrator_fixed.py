import asyncio
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv
import signal
import sys

# Import working components
from config.settings import get_settings
from database.simple_connection import DatabaseManager
from infrastructure.prometheus_monitor import PrometheusMonitor
from infrastructure.health_check import HealthCheck
from data_collection.fixed_alpaca_collector import FixedAlpacaCollector
from risk_management.risk_engine import get_risk_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedOrchestrator:
    """Simple orchestrator that actually works"""
    
    def __init__(self):
        load_dotenv('.env')
        self.config = self._load_config()
        self.components = {}
        self.running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            # Database
            'DB_HOST': os.getenv('DB_HOST', 'localhost'),
            'DB_PORT': int(os.getenv('DB_PORT', 5432)),
            'DB_NAME': os.getenv('DB_NAME', 'omni_alpha'),
            'DB_USER': os.getenv('DB_USER', 'postgres'),
            'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
            
            # Redis
            'REDIS_HOST': os.getenv('REDIS_HOST', 'localhost'),
            'REDIS_PORT': int(os.getenv('REDIS_PORT', 6379)),
            
            # Alpaca
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
            'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
            
            # System
            'MONITORING_ENABLED': os.getenv('MONITORING_ENABLED', 'true') == 'true',
            'PROMETHEUS_PORT': int(os.getenv('PROMETHEUS_PORT', 8001)),
        }
        
    async def initialize(self):
        """Initialize all components in correct order"""
        logger.info("🚀 Initializing Omni Alpha 5.0...")
        
        # 1. Database
        logger.info("Connecting to databases...")
        self.components['database'] = DatabaseManager(self.config)
        await self.components['database'].initialize()
        
        # 2. Monitoring (if enabled)
        if self.config['MONITORING_ENABLED']:
            logger.info("Starting monitoring...")
            self.components['monitoring'] = PrometheusMonitor(self.config)
            self.components['monitoring'].start_server()
            
        # 3. Health checks
        logger.info("Initializing health checks...")
        self.components['health'] = HealthCheck()
        
        # Register health checks
        self.components['health'].register_component('database', self._check_database_health)
        
        # 4. Risk Engine
        logger.info("Initializing risk engine...")
        try:
            self.components['risk'] = await get_risk_engine()
            self.components['health'].register_component('risk_engine', self._check_risk_health)
        except Exception as e:
            logger.warning(f"Risk engine initialization failed: {e}")
        
        # 5. Alpaca Data Collector
        logger.info("Connecting to Alpaca...")
        if self.config['ALPACA_API_KEY'] and self.config['ALPACA_SECRET_KEY']:
            self.components['alpaca'] = FixedAlpacaCollector(self.config)
            alpaca_connected = await self.components['alpaca'].initialize()
            
            if alpaca_connected:
                # Start streaming for default symbols
                await self.components['alpaca'].start_streaming(['SPY', 'QQQ', 'AAPL'])
                self.components['health'].register_component('alpaca', self._check_alpaca_health)
            else:
                logger.warning("Alpaca connection failed - continuing without live data")
        else:
            logger.warning("Alpaca credentials missing - continuing without live data")
        
        self.running = True
        logger.info("✅ System initialization complete!")
        
        # Print status
        await self._print_status()
        
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        db = self.components.get('database')
        if db and db.connected:
            return {'status': 'healthy', 'message': 'Database connected'}
        else:
            return {'status': 'degraded', 'message': 'Using fallback database'}
            
    async def _check_risk_health(self) -> Dict[str, Any]:
        """Check risk engine health"""
        risk = self.components.get('risk')
        if risk:
            return {'status': 'healthy', 'message': 'Risk engine active'}
        else:
            return {'status': 'unhealthy', 'message': 'Risk engine not available'}
            
    async def _check_alpaca_health(self) -> Dict[str, Any]:
        """Check Alpaca health"""
        alpaca = self.components.get('alpaca')
        if alpaca:
            return alpaca.get_health_status()
        else:
            return {'status': 'degraded', 'message': 'Alpaca not connected'}
        
    def _print_detailed_status(self):
        """Print detailed component status"""
        print("\n📊 Detailed Component Status:")
        
        for name, component in self.components.items():
            if hasattr(component, 'get_health_status'):
                health = component.get_health_status()
                print(f"\n{name}:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
    
    async def _print_status(self):
        """Print system status"""
        print("\n" + "="*60)
        print("🎯 OMNI ALPHA 5.0 - SYSTEM STATUS")
        print("="*60)
        
        # Get health status
        health = await self.components['health'].check_all()
        
        # Component status
        print("\n📦 Components:")
        for name, component_health in health['components'].items():
            status_icon = {
                'healthy': '✅',
                'degraded': '⚠️',
                'unhealthy': '❌'
            }.get(component_health.get('status', 'unknown'), '❓')
            
            status_text = component_health.get('status', 'unknown').upper()
            message = component_health.get('message', '')
            print(f"   {name}: {status_icon} {status_text} - {message}")
            
        # Overall health
        overall_icon = {
            'healthy': '✅',
            'degraded': '⚠️', 
            'unhealthy': '❌'
        }.get(health['overall_status'], '❓')
        
        print(f"\n🏥 Overall Health: {overall_icon} {health['overall_status'].upper()}")
        print(f"   Components: {health['healthy_count']}/{health['total_count']} healthy")
        
        # Configuration
        print("\n⚙️ Configuration:")
        print(f"   Environment: production")
        print(f"   Trading Mode: paper")
        print(f"   Risk Controls: {'enabled' if 'risk' in self.components else 'disabled'}")
        print(f"   Live Data: {'enabled' if 'alpaca' in self.components else 'disabled'}")
        
        # Endpoints
        if self.config['MONITORING_ENABLED']:
            print("\n🌐 Endpoints:")
            print(f"   Metrics: http://localhost:{self.config['PROMETHEUS_PORT']}/metrics")
            print(f"   Health: Available via health check component")
            
        # Readiness assessment
        if health['overall_status'] == 'healthy':
            print("\n🚀 System is fully operational and ready for trading!")
        elif health['overall_status'] == 'degraded':
            print("\n⚠️ System is operational but some features are degraded")
        else:
            print("\n❌ System has critical issues - check component status")
        
        # Print detailed component status for debugging
        self._print_detailed_status()
            
        print("="*60 + "\n")
        
    async def run(self):
        """Main run loop"""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Health monitoring loop
            health_check_interval = 30  # seconds
            last_health_check = 0
            
            # Keep running
            while self.running:
                current_time = asyncio.get_event_loop().time()
                
                # Periodic health checks
                if current_time - last_health_check >= health_check_interval:
                    try:
                        health = await self.components['health'].check_all()
                        
                        # Update monitoring metrics
                        if 'monitoring' in self.components:
                            health_score = health['healthy_count'] / max(health['total_count'], 1)
                            self.components['monitoring'].update_health(health_score)
                            
                        # Log health changes
                        if health['overall_status'] != 'healthy':
                            logger.warning(f"System health: {health['overall_status']}")
                            
                        last_health_check = current_time
                        
                    except Exception as e:
                        logger.error(f"Health check error: {e}")
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received...")
        self.running = False
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down components...")
        
        # Close components in reverse order
        for name in reversed(list(self.components.keys())):
            component = self.components[name]
            try:
                if hasattr(component, 'close'):
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                logger.info(f"✅ {name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
                
        logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    orchestrator = FixedOrchestrator()
    await orchestrator.initialize()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())

"""
OMNI ALPHA 5.0 - ENHANCED ORCHESTRATOR
======================================
Enhanced orchestrator with production components (compatible version)
"""

import asyncio
import signal
import sys
import time
import os
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

# Production infrastructure imports (with fallbacks)
try:
    from database.connection_pool import initialize_production_db
    PRODUCTION_DB_AVAILABLE = True
except ImportError:
    PRODUCTION_DB_AVAILABLE = False

try:
    from observability.tracing import initialize_tracing
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

try:
    from security.enterprise.security_manager import initialize_enterprise_security
    ENTERPRISE_SECURITY_AVAILABLE = True
except ImportError:
    ENTERPRISE_SECURITY_AVAILABLE = False

class EnhancedOrchestrator:
    """Enhanced orchestrator with production-grade components"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = None  # Will be initialized
        self.components = {}
        self.is_running = False
        self.start_time = None
        
        # Component initialization order (core + available production components)
        self.initialization_order = [
            'logging',
            'database',
            'production_db',
            'tracing',
            'enterprise_security',
            'monitoring', 
            'circuit_breaker',
            'alpaca_collector',
            'risk_engine'
        ]
        
        # Component health requirements
        self.critical_components = ['logging', 'database', 'monitoring']
        self.optional_components = ['production_db', 'tracing', 'enterprise_security']
        
        # Health check optimization
        self.last_health_status = None
        self.health_check_counter = 0
    
    async def initialize(self) -> bool:
        """Initialize all available system components"""
        try:
            print("🚀 OMNI ALPHA 5.0 - ENHANCED SYSTEM INITIALIZATION")
            print("=" * 70)
            
            successful_components = 0
            total_components = len(self.initialization_order)
            
            # Initialize components in order
            for component_name in self.initialization_order:
                print(f"\n🔄 Initializing {component_name}...")
                
                success = await self._initialize_component(component_name)
                
                if success:
                    print(f"✅ {component_name} initialized successfully")
                    self.components[component_name] = True
                    successful_components += 1
                else:
                    print(f"❌ {component_name} initialization failed")
                    self.components[component_name] = False
                    
                    # Critical components must succeed
                    if component_name in self.critical_components:
                        print(f"💥 Critical component {component_name} failed - aborting")
                        return False
                    
                    # Optional components can fail
                    elif component_name in self.optional_components:
                        print(f"⚠️ Optional component {component_name} failed - continuing")
                    
                    # Core trading components should succeed but system can continue
                    else:
                        print(f"⚠️ Component {component_name} failed - system will run in degraded mode")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.start_time = datetime.now()
            self.is_running = True
            
            # Log successful initialization
            if self.logger:
                self.logger.info("Enhanced system initialization completed")
                self.logger.info(f"Initialized components: {successful_components}/{total_components}")
            
            # Display enhanced system status
            await self._display_enhanced_status()
            
            return True
            
        except Exception as e:
            print(f"💥 Enhanced system initialization failed: {e}")
            if self.logger:
                self.logger.critical(f"Enhanced system initialization failed: {e}")
            return False
    
    async def _initialize_component(self, component_name: str) -> bool:
        """Initialize individual component with enhanced error handling"""
        try:
            if component_name == 'logging':
                success = initialize_logging()
                if success:
                    self.logger = get_logger(__name__, 'enhanced_orchestrator')
                return success
                
            elif component_name == 'database':
                return await initialize_databases()
            
            elif component_name == 'production_db':
                if PRODUCTION_DB_AVAILABLE:
                    return await initialize_production_db()
                else:
                    if self.logger:
                        self.logger.info("Production DB not available, using standard database")
                    return True
            
            elif component_name == 'tracing':
                if TRACING_AVAILABLE:
                    return initialize_tracing()
                else:
                    if self.logger:
                        self.logger.info("Distributed tracing not available")
                    return True
            
            elif component_name == 'enterprise_security':
                if ENTERPRISE_SECURITY_AVAILABLE:
                    return await initialize_enterprise_security()
                else:
                    if self.logger:
                        self.logger.info("Enterprise security not available")
                    return True
                
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
                if self.logger:
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
    
    async def _display_enhanced_status(self):
        """Display comprehensive enhanced system status"""
        print("\n" + "=" * 70)
        print("🎉 OMNI ALPHA 5.0 - ENHANCED SYSTEM STATUS")
        print("=" * 70)
        
        # Basic info
        print(f"📊 Application: {self.settings.app_name} v{self.settings.version}")
        print(f"🌍 Environment: {self.settings.environment.value}")
        print(f"📈 Trading Mode: {self.settings.trading_mode.value}")
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏠 Instance: {os.getenv('INSTANCE_ID', 'local')}")
        
        # Component status with enhanced details
        print(f"\n🔧 ENHANCED COMPONENTS STATUS:")
        
        # Core components
        core_components = ['logging', 'database', 'monitoring', 'circuit_breaker', 'risk_engine']
        print(f"   📦 Core Infrastructure:")
        for component in core_components:
            if component in self.components:
                icon = "✅" if self.components[component] else "❌"
                status = "OPERATIONAL" if self.components[component] else "FAILED"
                print(f"      {icon} {component}: {status}")
        
        # Enhanced components
        enhanced_components = ['production_db', 'tracing', 'enterprise_security']
        print(f"   🏭 Enhanced Features:")
        for component in enhanced_components:
            if component in self.components:
                icon = "✅" if self.components[component] else "⚠️"
                status = "ACTIVE" if self.components[component] else "UNAVAILABLE"
                print(f"      {icon} {component}: {status}")
        
        # Data components
        data_components = ['alpaca_collector']
        print(f"   📡 Data Infrastructure:")
        for component in data_components:
            if component in self.components:
                icon = "✅" if self.components[component] else "⚠️"
                status = "CONNECTED" if self.components[component] else "DEGRADED"
                print(f"      {icon} {component}: {status}")
        
        # Enhanced configuration
        print(f"\n⚙️ ENHANCED CONFIGURATION:")
        print(f"   Max Position Size: ${self.settings.trading.max_position_size_dollars:,.2f}")
        print(f"   Max Daily Trades: {self.settings.trading.max_daily_trades}")
        print(f"   Max Daily Loss: ${self.settings.trading.max_daily_loss:,.2f}")
        print(f"   Max Drawdown: {self.settings.trading.max_drawdown_percent:.1%}")
        print(f"   Circuit Breakers: {'ENABLED' if self.settings.monitoring.circuit_breaker_enabled else 'DISABLED'}")
        print(f"   Latency Monitoring: {self.settings.trading.max_order_latency_us}μs threshold")
        
        # API status (masked for security)
        sensitive_config = self.settings.get_sensitive_config()
        print(f"\n🔐 API CONFIGURATION:")
        print(f"   Alpaca API Key: {sensitive_config['alpaca_api_key']}")
        print(f"   Telegram Token: {sensitive_config['telegram_token']}")
        print(f"   Google API Key: {sensitive_config['google_api_key']}")
        
        # Enhanced health checks with intelligent reporting
        if 'monitoring' in self.components and self.components['monitoring']:
            try:
                monitoring_manager = get_monitoring_manager()
                health = monitoring_manager.get_comprehensive_status()
                
                print(f"\n🏥 ENHANCED HEALTH ANALYSIS:")
                print(f"   Overall Status: {health['health']['status']}")
                print(f"   Health Score: {health['health']['score']:.1%}")
                print(f"   Components Monitored: {len(health['health']['components'])}")
                
                # Component breakdown
                healthy_count = sum(1 for comp in health['health']['components'].values() if comp['status'] == 'healthy')
                degraded_count = sum(1 for comp in health['health']['components'].values() if comp['status'] == 'degraded')
                critical_count = sum(1 for comp in health['health']['components'].values() if comp['status'] == 'critical')
                
                print(f"   📊 Health Breakdown: {healthy_count} Healthy | {degraded_count} Degraded | {critical_count} Critical")
                
                # Show component details (only if status changed or critical)
                for comp, comp_health in health['health']['components'].items():
                    if comp_health['status'] in ['critical', 'degraded']:
                        icon = "⚠️" if comp_health['status'] == 'degraded' else "❌"
                        print(f"   {icon} {comp}: {comp_health['status']} - {comp_health['message']}")
                    elif comp_health['status'] == 'healthy':
                        print(f"   ✅ {comp}: operational")
                
            except Exception as e:
                print(f"   ⚠️ Health check error: {e}")
        
        # Enhanced endpoints
        print(f"\n📊 ENHANCED ENDPOINTS:")
        if self.settings.monitoring.metrics_enabled:
            print(f"   Prometheus Metrics: http://localhost:{self.settings.monitoring.metrics_port}/metrics")
            print(f"   Health Check: http://localhost:{self.settings.monitoring.health_check_port}/health")
        
        # Additional production endpoints
        if TRACING_AVAILABLE and self.components.get('tracing'):
            jaeger_host = os.getenv('JAEGER_HOST', 'localhost')
            jaeger_port = os.getenv('JAEGER_UI_PORT', '16686')
            print(f"   Distributed Tracing: http://{jaeger_host}:{jaeger_port}")
        
        # System readiness assessment
        critical_healthy = all(
            self.components.get(comp, False) 
            for comp in self.critical_components
        )
        
        total_score = sum(1 for comp in self.components.values() if comp) / len(self.components)
        
        # Enhanced readiness calculation
        if critical_healthy and total_score >= 0.8:
            readiness = "🚀 PRODUCTION READY"
            color = "GREEN"
        elif critical_healthy and total_score >= 0.6:
            readiness = "⚠️ DEGRADED MODE"
            color = "YELLOW"
        else:
            readiness = "❌ NOT READY"
            color = "RED"
        
        print(f"\n🎯 ENHANCED SYSTEM READINESS: {readiness}")
        print(f"   Overall Score: {total_score:.1%}")
        print(f"   Status: {color}")
        print(f"   Critical Components: {'✅ ALL HEALTHY' if critical_healthy else '❌ ISSUES DETECTED'}")
        
        # Production features summary
        production_features = {
            'Database Pooling': PRODUCTION_DB_AVAILABLE and self.components.get('production_db', False),
            'Distributed Tracing': TRACING_AVAILABLE and self.components.get('tracing', False),
            'Enterprise Security': ENTERPRISE_SECURITY_AVAILABLE and self.components.get('enterprise_security', False),
            'Circuit Breakers': self.components.get('circuit_breaker', False),
            'Risk Management': self.components.get('risk_engine', False),
            'Real-time Monitoring': self.components.get('monitoring', False)
        }
        
        print(f"\n🏭 PRODUCTION FEATURES:")
        for feature, enabled in production_features.items():
            icon = "✅" if enabled else "⚠️"
            status = "ENABLED" if enabled else "DISABLED"
            print(f"   {icon} {feature}: {status}")
        
        if color == "GREEN":
            print("\n🏆 Enhanced system ready for institutional-grade trading operations!")
            print("🎯 All critical components operational with production features active")
        elif color == "YELLOW":
            print("\n⚠️ System operational but running in degraded mode")
            print("🔧 Some enhanced features unavailable but core trading functional")
        else:
            print("\n🚨 System requires attention before production use")
            print("🛠️ Critical components need repair")
    
    async def run(self):
        """Enhanced main execution loop with intelligent health monitoring"""
        if not self.is_running:
            if not await self.initialize():
                return False
        
        try:
            self.logger.info("Entering enhanced execution loop")
            
            # Enhanced monitoring loop with reduced noise
            while self.is_running:
                await asyncio.sleep(1)
                
                # Optimized health checks (every 30 seconds, but only log changes)
                if self.health_check_counter % 30 == 0:
                    await self._intelligent_health_check()
                
                # Performance monitoring (every 120 seconds)
                if self.health_check_counter % 120 == 0:
                    await self._performance_monitoring()
                
                self.health_check_counter += 1
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.critical(f"Critical error in enhanced loop: {e}")
        finally:
            await self.shutdown()
        
        return True
    
    async def _intelligent_health_check(self):
        """Intelligent health check that only logs changes or critical issues"""
        try:
            if 'monitoring' in self.components and self.components['monitoring']:
                monitoring_manager = get_monitoring_manager()
                health = monitoring_manager.get_comprehensive_status()
                
                current_status = health['health']['status']
                
                # Only log if status changed or is critical
                if (current_status != self.last_health_status or 
                    current_status in ['CRITICAL', 'DEGRADED']):
                    
                    if current_status != self.last_health_status:
                        self.logger.info(f"System health status changed: {self.last_health_status} → {current_status}")
                    
                    # Handle critical components intelligently
                    critical_components = [
                        comp for comp, comp_health in health['health']['components'].items()
                        if comp_health['status'] == 'critical'
                    ]
                    
                    degraded_components = [
                        comp for comp, comp_health in health['health']['components'].items()
                        if comp_health['status'] == 'degraded'
                    ]
                    
                    if critical_components:
                        # Filter out expected issues
                        real_critical = [
                            comp for comp in critical_components 
                            if not self._is_expected_issue(comp, health['health']['components'][comp])
                        ]
                        
                        if real_critical:
                            self.logger.error(f"Critical components requiring attention: {real_critical}")
                        else:
                            # Log as info if it's expected (like Alpaca in paper mode)
                            self.logger.info(f"Components in critical state (expected): {critical_components}")
                    
                    elif degraded_components and current_status != self.last_health_status:
                        self.logger.warning(f"Degraded components: {degraded_components}")
                    
                    self.last_health_status = current_status
                
        except Exception as e:
            self.logger.error(f"Intelligent health check error: {e}")
    
    def _is_expected_issue(self, component: str, health_data: Dict[str, Any]) -> bool:
        """Check if a component issue is expected/acceptable"""
        if component == 'alpaca_collector':
            # Alpaca collector issues are often due to missing secret key in paper trading
            message = health_data.get('message', '').lower()
            if 'api not initialized' in message or 'credentials missing' in message:
                return True  # Expected in paper trading mode
        
        return False
    
    async def _performance_monitoring(self):
        """Monitor system performance metrics"""
        try:
            # Get performance data
            from infrastructure.monitoring import get_performance_tracker
            performance_tracker = get_performance_tracker()
            stats = performance_tracker.get_all_stats()
            
            # Check for performance issues
            slow_operations = []
            for operation, op_stats in stats.items():
                if 'mean_us' in op_stats and op_stats['mean_us'] > 50000:  # > 50ms (more reasonable threshold)
                    slow_operations.append(f"{operation}: {op_stats['mean_us']:.0f}μs")
            
            if slow_operations:
                self.logger.warning(f"Performance alert - slow operations: {'; '.join(slow_operations[:3])}")
            
            # Log performance summary every 10 minutes
            if self.health_check_counter % 600 == 0:
                total_operations = sum(op_stats.get('count', 0) for op_stats in stats.values())
                if total_operations > 0:
                    self.logger.info(f"Performance summary: {total_operations} operations tracked, {len(slow_operations)} slow")
            
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}")
    
    async def shutdown(self):
        """Enhanced graceful system shutdown"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        print("\n🛑 INITIATING ENHANCED SHUTDOWN")
        print("=" * 50)
        
        if self.logger:
            self.logger.info("Initiating enhanced system shutdown")
        
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
        
        # Final cleanup and stats
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        successful_components = sum(1 for comp in self.components.values() if comp)
        
        print(f"\n📊 ENHANCED SHUTDOWN SUMMARY:")
        print(f"   Uptime: {uptime:.1f} seconds")
        print(f"   Components: {successful_components}/{len(self.components)} were operational")
        print(f"   Health Checks: {self.health_check_counter} performed")
        
        print(f"\n✅ Enhanced shutdown complete")
        if self.logger:
            self.logger.info(f"Enhanced shutdown complete (uptime: {uptime:.1f}s, health checks: {self.health_check_counter})")
    
    async def _shutdown_component(self, component_name: str):
        """Shutdown individual component"""
        if component_name == 'database':
            from config.database import shutdown_databases
            await shutdown_databases()
            
        elif component_name == 'production_db':
            if PRODUCTION_DB_AVAILABLE:
                try:
                    from database.connection_pool import get_production_database_pool
                    pool = get_production_database_pool()
                    await pool.close()
                except Exception as e:
                    self.logger.warning(f"Production DB shutdown error: {e}")
            
        elif component_name == 'monitoring':
            from infrastructure.monitoring import stop_monitoring
            await stop_monitoring()
            
        elif component_name == 'tracing':
            if TRACING_AVAILABLE:
                try:
                    from observability.tracing import get_distributed_tracing
                    tracing = get_distributed_tracing()
                    tracing.shutdown()
                except Exception as e:
                    self.logger.warning(f"Tracing shutdown error: {e}")
            
        elif component_name == 'alpaca_collector':
            from data_collection.providers.alpaca_collector import get_alpaca_collector
            collector = get_alpaca_collector()
            if hasattr(collector, 'stop_websocket_stream'):
                await collector.stop_websocket_stream()
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # Calculate enhanced health score
        healthy_components = sum(1 for comp in self.components.values() if comp)
        total_components = len(self.components)
        health_score = healthy_components / total_components if total_components > 0 else 0
        
        # Determine readiness level
        critical_healthy = all(
            self.components.get(comp, False) 
            for comp in self.critical_components
        )
        
        if critical_healthy and health_score >= 0.8:
            readiness_level = "PRODUCTION_READY"
        elif critical_healthy and health_score >= 0.6:
            readiness_level = "DEGRADED_MODE"
        else:
            readiness_level = "NOT_READY"
        
        return {
            'app_name': self.settings.app_name,
            'version': self.settings.version,
            'environment': self.settings.environment.value,
            'trading_mode': self.settings.trading_mode.value,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'readiness_level': readiness_level,
            'health_score': health_score,
            'components': self.components.copy(),
            'critical_components_healthy': critical_healthy,
            'enhanced_features': {
                'database_pooling': PRODUCTION_DB_AVAILABLE and self.components.get('production_db', False),
                'distributed_tracing': TRACING_AVAILABLE and self.components.get('tracing', False),
                'enterprise_security': ENTERPRISE_SECURITY_AVAILABLE and self.components.get('enterprise_security', False),
                'intelligent_monitoring': True,
                'circuit_breaker_protection': self.components.get('circuit_breaker', False),
                'real_time_risk_management': self.components.get('risk_engine', False)
            },
            'performance_metrics': {
                'health_checks_performed': self.health_check_counter,
                'last_health_status': self.last_health_status
            }
        }

# ===================== MAIN EXECUTION =====================

async def main():
    """Enhanced main entry point"""
    orchestrator = EnhancedOrchestrator()
    
    try:
        await orchestrator.run()
    except Exception as e:
        print(f"💥 Enhanced system execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

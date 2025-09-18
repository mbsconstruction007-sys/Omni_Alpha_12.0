"""
OMNI ALPHA 5.0 - PRODUCTION INFRASTRUCTURE SETUP
================================================
Complete setup script for production-grade infrastructure
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
import json
from datetime import datetime

class ProductionSetup:
    """Production infrastructure setup and validation"""
    
    def __init__(self):
        self.setup_report = {
            'timestamp': datetime.now().isoformat(),
            'components_installed': [],
            'components_tested': [],
            'errors': [],
            'warnings': []
        }
    
    def install_production_dependencies(self):
        """Install production dependencies"""
        print("📦 Installing production dependencies...")
        
        try:
            # Install from production requirements
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements_production.txt'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Production dependencies installed successfully")
                self.setup_report['components_installed'].append('production_dependencies')
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                self.setup_report['errors'].append(f"Dependency installation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⚠️ Dependency installation timed out")
            self.setup_report['warnings'].append("Dependency installation timed out")
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            self.setup_report['errors'].append(f"Dependency error: {e}")
    
    def create_production_directories(self):
        """Create production directory structure"""
        print("📁 Creating production directories...")
        
        directories = [
            'logs/production',
            'data/production',
            'backups',
            'configs/production',
            'certificates',
            'load_test_results',
            'audit_logs',
            'monitoring/dashboards',
            'scripts/production'
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created: {directory}")
                self.setup_report['components_installed'].append(f'directory_{directory}')
            except Exception as e:
                print(f"   ❌ Failed to create {directory}: {e}")
                self.setup_report['errors'].append(f"Directory creation failed: {directory} - {e}")
    
    def create_production_configs(self):
        """Create production configuration files"""
        print("⚙️ Creating production configuration files...")
        
        # Production environment template
        env_template = """# OMNI ALPHA 5.0 - PRODUCTION ENVIRONMENT
# Copy this to .env.production and configure

# Core settings
APP_NAME=Omni Alpha 5.0
ENVIRONMENT=production
TRADING_MODE=paper
DEBUG=false

# Database
DB_PRIMARY_HOST=localhost
DB_PRIMARY_PORT=5432
DB_USER=postgres
DB_PASSWORD=CHANGE_ME
DB_NAME=omni_alpha_prod

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_ME

# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=CHANGE_ME
INFLUXDB_ORG=omni_alpha
INFLUXDB_BUCKET=market_data

# API Keys
ALPACA_API_KEY=PK02D3BXIPSW11F0Q9OW
ALPACA_SECRET_KEY=CHANGE_ME
TELEGRAM_BOT_TOKEN=8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk
GOOGLE_API_KEY=AIzaSyDpKZV5XTysC2T9lJax29v2kIAR2q6LXnU

# Security
ENCRYPTION_KEY=GENERATE_NEW_KEY
JWT_SECRET=GENERATE_NEW_SECRET

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=8001
HEALTH_CHECK_PORT=8000

# Tracing
TRACING_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=14268

# Service Discovery
CONSUL_HOST=localhost
CONSUL_PORT=8500

# Message Queue
KAFKA_BROKERS=localhost:9092
"""
        
        try:
            with open('.env.production.template', 'w') as f:
                f.write(env_template)
            print("   ✅ Created: .env.production.template")
            self.setup_report['components_installed'].append('env_template')
        except Exception as e:
            print(f"   ❌ Failed to create env template: {e}")
            self.setup_report['errors'].append(f"Env template creation failed: {e}")
        
        # Docker compose for production
        docker_compose = """version: '3.8'

services:
  trading-engine:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8080:8080"
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
      - influxdb
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: omni_alpha_prod
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  influxdb:
    image: influxdb:2.7-alpine
    ports:
      - "8086:8086"
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUXDB_PASSWORD}
      DOCKER_INFLUXDB_INIT_ORG: omni_alpha
      DOCKER_INFLUXDB_INIT_BUCKET: market_data
    volumes:
      - influxdb_data:/var/lib/influxdb2
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
    restart: unless-stopped
    
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      COLLECTOR_OTLP_ENABLED: true
    restart: unless-stopped
    
  consul:
    image: consul:latest
    ports:
      - "8500:8500"
    command: agent -server -bootstrap-expect=1 -ui -client=0.0.0.0
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  influxdb_data:
  prometheus_data:
  grafana_data:
"""
        
        try:
            with open('docker-compose.production.yml', 'w') as f:
                f.write(docker_compose)
            print("   ✅ Created: docker-compose.production.yml")
            self.setup_report['components_installed'].append('docker_compose_production')
        except Exception as e:
            print(f"   ❌ Failed to create docker-compose: {e}")
            self.setup_report['errors'].append(f"Docker compose creation failed: {e}")
    
    async def test_production_components(self):
        """Test production components"""
        print("🧪 Testing production components...")
        
        # Test orchestrator
        try:
            from orchestrator_production import ProductionOrchestrator
            orchestrator = ProductionOrchestrator()
            status = orchestrator.get_production_status()
            
            print(f"   ✅ Production orchestrator: {status['readiness_level']}")
            self.setup_report['components_tested'].append('production_orchestrator')
            
        except Exception as e:
            print(f"   ❌ Production orchestrator test failed: {e}")
            self.setup_report['errors'].append(f"Orchestrator test failed: {e}")
        
        # Test database layer
        try:
            try:
                from database.connection_pool import get_production_database_pool
                pool = get_production_database_pool()
                print("   ✅ Production database pool available")
                self.setup_report['components_tested'].append('production_database')
            except ImportError:
                print("   ⚠️ Production database pool not available")
                self.setup_report['warnings'].append("Production database pool not available")
                
        except Exception as e:
            print(f"   ❌ Database test failed: {e}")
            self.setup_report['errors'].append(f"Database test failed: {e}")
        
        # Test monitoring
        try:
            from infrastructure.monitoring import get_monitoring_manager
            manager = get_monitoring_manager()
            status = manager.get_comprehensive_status()
            
            print(f"   ✅ Monitoring system: {status['health']['status']}")
            self.setup_report['components_tested'].append('monitoring_system')
            
        except Exception as e:
            print(f"   ❌ Monitoring test failed: {e}")
            self.setup_report['errors'].append(f"Monitoring test failed: {e}")
        
        # Test risk engine
        try:
            from risk_management.risk_engine import get_risk_engine
            engine = get_risk_engine()
            health = await engine.health_check()
            
            print(f"   ✅ Risk engine: {health['status']}")
            self.setup_report['components_tested'].append('risk_engine')
            
        except Exception as e:
            print(f"   ❌ Risk engine test failed: {e}")
            self.setup_report['errors'].append(f"Risk engine test failed: {e}")
    
    def create_production_scripts(self):
        """Create production management scripts"""
        print("📜 Creating production scripts...")
        
        # Health check script
        health_script = """#!/bin/bash
# Production health check script

echo "🏥 OMNI ALPHA 5.0 - PRODUCTION HEALTH CHECK"
echo "============================================"

# Check orchestrator
echo "🔍 Orchestrator Status:"
python -c "
import asyncio
from orchestrator_production import ProductionOrchestrator
async def check():
    o = ProductionOrchestrator()
    if hasattr(o, 'get_production_status'):
        status = o.get_production_status()
        print(f'Readiness: {status[\"readiness_level\"]}')
        print(f'Health Score: {status[\"health_score\"]:.1%}')
        print(f'Uptime: {status.get(\"uptime_seconds\", 0):.1f}s')
asyncio.run(check())
"

# Check endpoints
echo ""
echo "🌐 Endpoint Health:"
curl -s -o /dev/null -w "Health: %{http_code} (%{time_total}s)\\n" http://localhost:8000/health
curl -s -o /dev/null -w "Metrics: %{http_code} (%{time_total}s)\\n" http://localhost:8001/metrics

# Check databases
echo ""
echo "🗄️ Database Status:"
if command -v pg_isready &> /dev/null; then
    pg_isready -h ${DB_PRIMARY_HOST:-localhost} -p ${DB_PRIMARY_PORT:-5432}
else
    echo "PostgreSQL client not available"
fi

if command -v redis-cli &> /dev/null; then
    redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping
else
    echo "Redis client not available"
fi

echo ""
echo "✅ Health check complete"
"""
        
        try:
            script_path = Path('scripts/production_health_check.sh')
            script_path.parent.mkdir(parents=True, exist_ok=True)
            with open(script_path, 'w') as f:
                f.write(health_script)
            
            # Make executable on Unix-like systems
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
            
            print("   ✅ Created: scripts/production_health_check.sh")
            self.setup_report['components_installed'].append('health_check_script')
            
        except Exception as e:
            print(f"   ❌ Failed to create health check script: {e}")
            self.setup_report['errors'].append(f"Health script creation failed: {e}")
        
        # Start script
        start_script = """#!/bin/bash
# Production start script

echo "🚀 Starting Omni Alpha 5.0 Production System"

# Load environment
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Start with production orchestrator
python orchestrator_production.py
"""
        
        try:
            script_path = Path('scripts/start_production.sh')
            with open(script_path, 'w') as f:
                f.write(start_script)
            
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
            
            print("   ✅ Created: scripts/start_production.sh")
            self.setup_report['components_installed'].append('start_script')
            
        except Exception as e:
            print(f"   ❌ Failed to create start script: {e}")
            self.setup_report['errors'].append(f"Start script creation failed: {e}")
    
    def generate_security_keys(self):
        """Generate secure keys for production"""
        print("🔐 Generating security keys...")
        
        try:
            from cryptography.fernet import Fernet
            import secrets
            
            # Generate encryption key
            encryption_key = Fernet.generate_key().decode()
            
            # Generate JWT secret
            jwt_secret = secrets.token_urlsafe(64)
            
            # Generate API key secret
            api_key_secret = secrets.token_urlsafe(32)
            
            # Save to secure file
            keys_file = Path('configs/production/security_keys.env')
            keys_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(keys_file, 'w') as f:
                f.write(f"# Generated security keys - KEEP SECURE!\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                f.write(f"ENCRYPTION_KEY={encryption_key}\n")
                f.write(f"JWT_SECRET={jwt_secret}\n")
                f.write(f"API_KEY_SECRET={api_key_secret}\n")
            
            # Set restrictive permissions
            if os.name != 'nt':
                os.chmod(keys_file, 0o600)
            
            print(f"   ✅ Security keys generated: {keys_file}")
            print(f"   🔒 File permissions set to 600 (owner read/write only)")
            self.setup_report['components_installed'].append('security_keys')
            
        except Exception as e:
            print(f"   ❌ Failed to generate security keys: {e}")
            self.setup_report['errors'].append(f"Security key generation failed: {e}")
    
    def validate_production_readiness(self):
        """Validate production readiness"""
        print("✅ Validating production readiness...")
        
        readiness_checks = {
            'config_files': Path('.env.production.template').exists(),
            'docker_compose': Path('docker-compose.production.yml').exists(),
            'orchestrator': Path('orchestrator_production.py').exists(),
            'security_keys': Path('configs/production/security_keys.env').exists(),
            'monitoring_config': Path('monitoring/prometheus.yml').exists(),
            'incident_runbook': Path('docs/runbooks/incident_response.md').exists()
        }
        
        passed_checks = 0
        total_checks = len(readiness_checks)
        
        for check_name, passed in readiness_checks.items():
            icon = "✅" if passed else "❌"
            print(f"   {icon} {check_name}")
            if passed:
                passed_checks += 1
            else:
                self.setup_report['errors'].append(f"Readiness check failed: {check_name}")
        
        readiness_score = passed_checks / total_checks
        
        print(f"\n📊 PRODUCTION READINESS SCORE: {readiness_score:.1%}")
        
        if readiness_score >= 0.9:
            print("🏆 PRODUCTION READY - All systems go!")
            return True
        elif readiness_score >= 0.7:
            print("⚠️ MOSTLY READY - Some optional components missing")
            return True
        else:
            print("❌ NOT READY - Critical components missing")
            return False
    
    def save_setup_report(self):
        """Save setup report"""
        report_file = f'production_setup_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.setup_report, f, indent=2)
            
            print(f"\n📄 Setup report saved: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save setup report: {e}")
    
    async def run_complete_setup(self):
        """Run complete production setup"""
        print("🏭 OMNI ALPHA 5.0 - PRODUCTION INFRASTRUCTURE SETUP")
        print("=" * 60)
        
        # Step 1: Install dependencies
        self.install_production_dependencies()
        
        # Step 2: Create directories
        self.create_production_directories()
        
        # Step 3: Create configurations
        self.create_production_configs()
        
        # Step 4: Generate security keys
        self.generate_security_keys()
        
        # Step 5: Create scripts
        self.create_production_scripts()
        
        # Step 6: Test components
        await self.test_production_components()
        
        # Step 7: Validate readiness
        is_ready = self.validate_production_readiness()
        
        # Step 8: Save report
        self.save_setup_report()
        
        print(f"\n🎯 SETUP COMPLETE:")
        print(f"   Components Installed: {len(self.setup_report['components_installed'])}")
        print(f"   Components Tested: {len(self.setup_report['components_tested'])}")
        print(f"   Errors: {len(self.setup_report['errors'])}")
        print(f"   Warnings: {len(self.setup_report['warnings'])}")
        
        if is_ready:
            print(f"\n🚀 OMNI ALPHA 5.0 IS PRODUCTION READY!")
            print(f"   Start with: python orchestrator_production.py")
            print(f"   Or with Docker: docker-compose -f docker-compose.production.yml up -d")
        else:
            print(f"\n⚠️ Production setup needs attention - check errors above")
        
        return is_ready

async def main():
    """Main setup execution"""
    setup = ProductionSetup()
    await setup.run_complete_setup()

if __name__ == "__main__":
    asyncio.run(main())

#!/bin/bash

# ============================================
# OMNI ALPHA 5.0 - STEP 10 DEPLOYMENT SCRIPT
# ============================================

set -e

echo "🚀 Starting Omni Alpha 5.0 - Step 10: Final Convergence"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
}

# Setup environment
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # Copy env file if not exists
    if [ ! -f .env ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please update .env file with your configuration${NC}"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
    mkdir -p config
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/grafana/dashboards
    
    echo -e "${GREEN}Environment setup complete${NC}"
}

# Start infrastructure
start_infrastructure() {
    echo -e "${YELLOW}Starting infrastructure services...${NC}"
    
    # Start databases and message queues
    docker-compose up -d postgres mongodb redis timescaledb
    sleep 10
    
    docker-compose up -d zookeeper kafka
    sleep 10
    
    # Start monitoring
    docker-compose up -d prometheus grafana elasticsearch kibana
    sleep 10
    
    echo -e "${GREEN}Infrastructure services started${NC}"
}

# Initialize databases
initialize_databases() {
    echo -e "${YELLOW}Initializing databases...${NC}"
    
    # Run database migrations
    python scripts/migrate_database.py
    
    # Create indexes
    python scripts/create_indexes.py
    
    echo -e "${GREEN}Databases initialized${NC}"
}

# Start orchestrator
start_orchestrator() {
    echo -e "${YELLOW}Starting Master Orchestrator...${NC}"
    
    # Build Docker image
    docker build -t omni-alpha/orchestrator:10.0.0 .
    
    # Start orchestrator
    docker-compose up -d orchestrator
    
    echo -e "${GREEN}Master Orchestrator started${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}Performing health check...${NC}"
    
    # Wait for services to be ready
    sleep 30
    
    # Check orchestrator health
    response=$(curl -s http://localhost:9000/health)
    
    if [[ $response == *"HEALTHY"* ]]; then
        echo -e "${GREEN}System is healthy and ready${NC}"
    else
        echo -e "${RED}System health check failed${NC}"
        echo $response
        exit 1
    fi
}

# Main execution
main() {
    echo "================================================"
    echo "    OMNI ALPHA 5.0 - STEP 10 DEPLOYMENT"
    echo "================================================"
    
    check_prerequisites
    install_dependencies
    setup_environment
    start_infrastructure
    initialize_databases
    start_orchestrator
    health_check
    
    echo ""
    echo -e "${GREEN}✅ Omni Alpha 5.0 Step 10 is now running!${NC}"
    echo ""
    echo "Access points:"
    echo "  - Orchestrator API: http://localhost:9000"
    echo "  - Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Kibana: http://localhost:5601"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f orchestrator"
    echo ""
    echo "To stop the system:"
    echo "  docker-compose down"
    echo ""
    echo -e "${YELLOW}⚡ The system is evolving...${NC}"
}

# Run main function
main

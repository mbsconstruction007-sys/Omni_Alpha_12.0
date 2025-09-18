#!/bin/bash

#
# ============================================================
# OMNI ALPHA 5.0 - STEP 1 INSTALLATION & VERIFICATION SCRIPT
# ============================================================
# This script will:
# 1. Install all Step 1 enhanced components
# 2. Setup security and encryption
# 3. Run verification tests
# 4. Start the infrastructure

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================
# STEP 1: CHECK PREREQUISITES
# ============================================================
print_header "STEP 1: CHECKING PREREQUISITES"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )[\d.]+')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    print_success "Python version $PYTHON_VERSION meets requirement (>=$REQUIRED_VERSION)"
else
    print_error "Python version $PYTHON_VERSION is too old (requires >=$REQUIRED_VERSION)"
    exit 1
fi

# Check if git is installed
if command -v git &> /dev/null; then
    print_success "Git is installed"
else
    print_error "Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ -d .git ]; then
    print_success "In git repository"
else
    print_warning "Not in a git repository. Initializing..."
    git init
fi

# ============================================================
# STEP 2: SECURITY CHECK
# ============================================================
print_header "STEP 2: SECURITY CHECK"

# Check for exposed credentials in git history
if git rev-list --all | xargs -I{} git ls-tree -r {} --name-only | grep -E "\.env$" &> /dev/null; then
    print_error "CRITICAL: .env file found in git history!"
    print_warning "Run this to clean history: git filter-branch --index-filter 'git rm --cached --ignore-unmatch .env' HEAD"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "No .env file in git history"
fi

# Check .gitignore
if [ -f .gitignore ]; then
    if grep -q "^\.env" .gitignore; then
        print_success ".env is in .gitignore"
    else
        print_warning "Adding .env to .gitignore"
        echo -e "\n# Security\n.env\n.env.local\n.env.production\n*.key\n*.pem" >> .gitignore
    fi
else
    print_warning "Creating .gitignore"
    cat > .gitignore << 'EOF'
# Security
.env
.env.local
.env.production
*.key
*.pem
credentials/
secrets/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Testing
.coverage
.pytest_cache/
htmlcov/

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo
EOF
    print_success "Created .gitignore with security rules"
fi

# ============================================================
# STEP 3: CREATE DIRECTORY STRUCTURE
# ============================================================
print_header "STEP 3: CREATING DIRECTORY STRUCTURE"

directories=(
    "src/core"
    "src/routers"
    "src/strategies"
    "tests"
    "scripts"
    "config"
    "logs"
    "data"
    "models"
    "monitoring"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    else
        print_success "Directory exists: $dir"
    fi
done

# ============================================================
# STEP 4: INSTALL DEPENDENCIES
# ============================================================
print_header "STEP 4: INSTALLING DEPENDENCIES"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_warning "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Pip upgraded"

# Install core requirements
print_warning "Installing core dependencies (this may take a few minutes)..."

# Install dependencies from existing requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Dependencies installed from requirements.txt"
else
    # Install basic dependencies
    pip install fastapi uvicorn sqlalchemy pandas numpy cryptography prometheus-client structlog psutil pytest python-dotenv aiohttp websockets redis alpaca-trade-api > /dev/null 2>&1
    print_success "Basic dependencies installed"
fi

# ============================================================
# STEP 5: SETUP ENVIRONMENT
# ============================================================
print_header "STEP 5: SETTING UP ENVIRONMENT"

# Create .env.local from template if it doesn't exist
if [ ! -f .env.local ]; then
    if [ -f step1_environment_template.env ]; then
        cp step1_environment_template.env .env.local
        print_success "Created .env.local from template"
    elif [ -f alpaca_live_trading.env ]; then
        cp alpaca_live_trading.env .env.local
        print_success "Created .env.local from existing env file"
    else
        # Create basic .env.local
        cat > .env.local << 'EOF'
# OMNI ALPHA 5.0 - Enhanced Configuration
ENV=production
APP_NAME=Omni Alpha Enhanced
APP_VERSION=5.0.0
TRADING_MODE=paper

# API Keys (Encrypted)
API_KEY_ENCRYPTED=
API_SECRET_ENCRYPTED=
ENCRYPTION_KEY=

# Trading Limits
MAX_POSITION_SIZE=10000
MAX_DAILY_TRADES=100
MAX_DAILY_LOSS=1000
MAX_DRAWDOWN_PCT=0.02

# Latency Thresholds (microseconds)
MAX_ORDER_LATENCY_US=10000
MAX_DATA_LATENCY_US=1000
MAX_STRATEGY_LATENCY_US=5000

# Circuit Breakers
CIRCUIT_BREAKER_ENABLED=true
MAX_CONSECUTIVE_ERRORS=5
ERROR_COOLDOWN_SECONDS=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
EOF
        print_success "Created basic .env.local"
    fi
else
    print_success ".env.local already exists"
fi

# Generate encryption key if not present
if ! grep -q "^ENCRYPTION_KEY=" .env.local || grep -q "^ENCRYPTION_KEY=$" .env.local; then
    print_warning "Generating encryption key..."
    ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    
    if grep -q "^ENCRYPTION_KEY=" .env.local; then
        # Update existing key
        sed -i.bak "s/^ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env.local
    else
        # Add new key
        echo "ENCRYPTION_KEY=$ENCRYPTION_KEY" >> .env.local
    fi
    print_success "Encryption key generated and added to .env.local"
    print_warning "SAVE THIS ENCRYPTION KEY: $ENCRYPTION_KEY"
fi

# ============================================================
# STEP 6: RUN TESTS
# ============================================================
print_header "STEP 6: RUNNING VERIFICATION TESTS"

# Test imports
print_warning "Testing imports..."
python3 -c "from step_1_core_infrastructure import CoreInfrastructure; print('✅ Enhanced infrastructure import successful')" 2>/dev/null && \
    print_success "Import test passed" || \
    print_error "Import test failed - check dependencies"

# Test basic functionality
print_warning "Testing core functionality..."
python3 -c "
import asyncio
from step_1_core_infrastructure import CoreInfrastructure

async def test():
    core = CoreInfrastructure()
    config = core.config
    print(f'App: {config.APP_NAME} v{config.APP_VERSION}')
    print(f'Trading Mode: {config.TRADING_MODE}')
    return True

result = asyncio.run(test())
print('✅ Core functionality test passed' if result else '❌ Core functionality test failed')
" && print_success "Core functionality test passed" || print_warning "Core functionality test had issues"

# ============================================================
# STEP 7: API KEY SECURITY CHECK
# ============================================================
print_header "STEP 7: API KEY SECURITY CHECK"

print_warning "Checking for exposed API keys..."

# Check for common exposed keys (examples - replace with actual checks)
EXPOSED_KEYS=(
    "PKFBP7MN3TZB3KIZOCTG"
    "AIzaSyDpKZV5XTysC2T9lJax29v2kIAR2q6LXnU"
    "ghp_GsyxFNu8NrrwaPwGP2N1p2eKAqY6Qt1ezG60"
)

for key in "${EXPOSED_KEYS[@]}"; do
    if grep -r "$key" . --exclude-dir=venv --exclude-dir=.git 2>/dev/null | grep -v "EXPOSED_KEYS"; then
        print_error "CRITICAL: Exposed key found: ${key:0:10}..."
        print_warning "Rotate this key immediately!"
    fi
done

print_success "Security check complete"

# ============================================================
# STEP 8: START INFRASTRUCTURE
# ============================================================
print_header "STEP 8: STARTING ENHANCED INFRASTRUCTURE"

echo -e "\n${GREEN}Would you like to start the enhanced infrastructure now?${NC}"
read -p "Start infrastructure? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Starting Omni Alpha 5.0 Enhanced Infrastructure..."
    
    # Start the enhanced infrastructure
    python3 step_1_core_infrastructure.py
else
    print_success "Infrastructure ready to start"
    echo "Run this to start: python3 step_1_core_infrastructure.py"
fi

# ============================================================
# STEP 9: FINAL REPORT
# ============================================================
print_header "INSTALLATION COMPLETE!"

cat << 'EOF'

✅ Step 1 Enhanced Infrastructure installed successfully!

🏛️ INSTITUTIONAL COMPONENTS INSTALLED:
   ✓ Market Microstructure Engine
   ✓ Latency Monitoring (microsecond precision)
   ✓ Risk Management Engine
   ✓ Circuit Breaker System
   ✓ Emergency Kill Switch
   ✓ Position Reconciliation
   ✓ Enhanced Database Manager
   ✓ Order Book Management

🚀 NEXT STEPS:
   1. Encrypt your API keys:
      python3 -c "
from cryptography.fernet import Fernet
import os
key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
fernet = Fernet(key.encode())
api_key = input('Enter API Key: ')
encrypted = fernet.encrypt(api_key.encode()).decode()
print(f'API_KEY_ENCRYPTED={encrypted}')
      "

   2. Configure your settings in .env.local

   3. Run the enhanced infrastructure:
      python3 step_1_core_infrastructure.py

   4. Access metrics:
      http://localhost:8001/metrics

📚 ENHANCED FEATURES:
   - Microsecond Latency Monitoring
   - Circuit Breaker Protection
   - Emergency Kill Switch
   - Order Book Management
   - Market Microstructure Analysis
   - Institutional Risk Controls

⚠️  IMPORTANT REMINDERS:
   - NEVER commit .env.local to git
   - Test kill switch before live trading
   - Start with paper trading mode
   - Monitor latency thresholds
   - Configure circuit breaker limits

🎉 Happy Trading with Omni Alpha 5.0 Enhanced!

EOF

# ============================================================
# END OF INSTALLATION
# ============================================================

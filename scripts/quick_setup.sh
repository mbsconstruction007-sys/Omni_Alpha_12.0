#!/bin/bash
# Quick setup script for Omni Alpha 12.0

set -e

echo "🚀 Quick Setup for Omni Alpha 12.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
GITHUB_USER="mbsconstruction007-sys"
GITHUB_REPO="Omni_Alpha_12.0"

echo -e "${BLUE}Step 1: Clone repository${NC}"
if [ ! -d "$GITHUB_REPO" ]; then
    git clone "https://github.com/$GITHUB_USER/$GITHUB_REPO.git"
    cd "$GITHUB_REPO"
else
    echo -e "${YELLOW}Repository already exists, updating...${NC}"
    cd "$GITHUB_REPO"
    git pull origin main
fi

echo -e "${BLUE}Step 2: Set up Python environment${NC}"
python -m venv venv

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo -e "${BLUE}Step 3: Install dependencies${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${BLUE}Step 4: Set up environment configuration${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env file. Please edit it with your configuration.${NC}"
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

echo -e "${BLUE}Step 5: Create necessary directories${NC}"
mkdir -p data logs models

echo -e "${BLUE}Step 6: Set up git configuration${NC}"
git config user.name "$GITHUB_USER"
git config user.email "mbsconstruction007-sys@users.noreply.github.com"

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit .env file with your configuration"
echo "2. Run tests: python -m pytest tests/"
echo "3. Start the system: python backend/main.py"
echo "4. Or use Docker: docker-compose up -d"
echo ""
echo -e "${BLUE}Repository URL: https://github.com/$GITHUB_USER/$GITHUB_REPO${NC}"

#!/bin/bash
# GitHub CLI setup and commands for Omni Alpha 12.0

set -e

echo "🔧 Setting up GitHub CLI for Omni Alpha 12.0"

# Configuration
GITHUB_USER="mbsconstruction007-sys"
GITHUB_REPO="Omni_Alpha_12.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}GitHub CLI is not installed. Please install it first:${NC}"
    echo "  macOS: brew install gh"
    echo "  Windows: winget install GitHub.cli"
    echo "  Linux: https://cli.github.com/"
    exit 1
fi

echo -e "${BLUE}Step 1: Authenticate with GitHub${NC}"
echo -e "${YELLOW}Please authenticate with your GitHub token when prompted...${NC}"
gh auth login

echo -e "${BLUE}Step 2: Create repository (if not exists)${NC}"
if ! gh repo view "$GITHUB_USER/$GITHUB_REPO" &> /dev/null; then
    gh repo create "$GITHUB_REPO" --public --description "Global Financial Ecosystem - Omni Alpha 12.0"
    echo -e "${GREEN}Repository created successfully!${NC}"
else
    echo -e "${GREEN}Repository already exists.${NC}"
fi

echo -e "${BLUE}Step 3: Set repository topics${NC}"
gh repo edit "$GITHUB_USER/$GITHUB_REPO" --add-topic trading,ai,machine-learning,finance,algorithmic-trading,cryptocurrency,forex,quantum-computing,fintech,blockchain

echo -e "${BLUE}Step 4: Create a release${NC}"
gh release create v12.0.0 \
  --title "Omni Alpha 12.0 - Global Market Dominance" \
  --notes "Complete implementation of all 12 steps including global financial ecosystem with AI superintelligence, market making, and institutional operations." \
  --target main

echo -e "${BLUE}Step 5: Create project board${NC}"
gh project create --title "Omni Alpha Development" --body "Development tracking for Omni Alpha 12.0" --public

echo -e "${BLUE}Step 6: Set default repository${NC}"
gh repo set-default "$GITHUB_USER/$GITHUB_REPO"

echo -e "${BLUE}Step 7: Create initial issues${NC}"
gh issue create --title "Set up CI/CD pipeline" --body "Implement GitHub Actions for continuous integration and deployment" --label "enhancement,infrastructure"
gh issue create --title "Implement core trading engine" --body "Build the foundation trading system" --label "enhancement,strategy"
gh issue create --title "Add risk management system" --body "Implement comprehensive risk controls" --label "enhancement,risk"
gh issue create --title "Create AI brain integration" --body "Integrate AI/ML capabilities" --label "enhancement,ai"
gh issue create --title "Build monitoring dashboard" --body "Create Grafana dashboards for monitoring" --label "enhancement,infrastructure"

echo -e "${BLUE}Step 8: Set up repository secrets (manual step)${NC}"
echo -e "${YELLOW}Please add these secrets to your repository manually:${NC}"
echo "  - AWS_ACCESS_KEY_ID"
echo "  - AWS_SECRET_ACCESS_KEY"
echo "  - DOCKER_USERNAME"
echo "  - DOCKER_PASSWORD"
echo "  - ALPACA_API_KEY"
echo "  - ALPACA_SECRET_KEY"
echo "  - SLACK_WEBHOOK"
echo ""
echo "Use: gh secret set SECRET_NAME --body 'SECRET_VALUE'"

echo -e "${GREEN}✅ GitHub CLI setup complete!${NC}"
echo ""
echo -e "${YELLOW}Repository URL: https://github.com/$GITHUB_USER/$GITHUB_REPO${NC}"
echo -e "${YELLOW}Release URL: https://github.com/$GITHUB_USER/$GITHUB_REPO/releases/tag/v12.0.0${NC}"

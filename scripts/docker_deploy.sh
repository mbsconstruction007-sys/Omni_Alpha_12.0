#!/bin/bash
# Docker deployment to GitHub Container Registry

set -e

echo "🐳 Deploying Omni Alpha 12.0 to GitHub Container Registry"

# Configuration
GITHUB_USER="mbsconstruction007-sys"
GITHUB_REPO="Omni_Alpha_12.0"
GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_NEW_TOKEN_HERE}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if token is set
if [ "$GITHUB_TOKEN" == "YOUR_NEW_TOKEN_HERE" ]; then
    echo -e "${RED}Error: Please set your GitHub token in the script or environment${NC}"
    exit 1
fi

# Create Dockerfile
echo -e "${YELLOW}Creating Dockerfile...${NC}"
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 omni && chown -R omni:omni /app
USER omni

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create .dockerignore
echo -e "${YELLOW}Creating .dockerignore...${NC}"
cat > .dockerignore << 'EOF'
# Git
.git
.gitignore

# Documentation
README.md
docs/
*.md

# Development
.vscode/
.idea/
*.swp
*.swo

# Testing
tests/
pytest.ini

# Data
data/
*.csv
*.db

# Logs
logs/
*.log

# Environment
.env
.env.*

# OS
.DS_Store
Thumbs.db
EOF

# Login to GitHub Container Registry
echo -e "${YELLOW}Logging in to GitHub Container Registry...${NC}"
echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

# Build and tag image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t "ghcr.io/$GITHUB_USER/omni-alpha-12:latest" .
docker build -t "ghcr.io/$GITHUB_USER/omni-alpha-12:v12.0.0" .

# Push images
echo -e "${YELLOW}Pushing images to GitHub Container Registry...${NC}"
docker push "ghcr.io/$GITHUB_USER/omni-alpha-12:latest"
docker push "ghcr.io/$GITHUB_USER/omni-alpha-12:v12.0.0"

# Create docker-compose for GitHub Container Registry
echo -e "${YELLOW}Creating docker-compose for GitHub Container Registry...${NC}"
cat > docker-compose.ghcr.yml << EOF
version: '3.8'

services:
  omni-alpha:
    image: ghcr.io/$GITHUB_USER/omni-alpha-12:latest
    container_name: omni-alpha-12
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Add other services as needed
  redis:
    image: redis:7-alpine
    container_name: omni-alpha-redis
    ports:
      - "6379:6379"
    restart: unless-stopped

  postgres:
    image: postgres:14
    container_name: omni-alpha-postgres
    environment:
      POSTGRES_DB: omni_alpha
      POSTGRES_USER: omni
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
EOF

echo -e "${GREEN}✅ Docker deployment complete!${NC}"
echo -e "${YELLOW}Images pushed to:${NC}"
echo "  - ghcr.io/$GITHUB_USER/omni-alpha-12:latest"
echo "  - ghcr.io/$GITHUB_USER/omni-alpha-12:v12.0.0"
echo ""
echo -e "${YELLOW}To run locally:${NC}"
echo "  docker-compose -f docker-compose.ghcr.yml up -d"

#!/bin/bash
# Production deployment script

set -e

echo "🚀 Starting production deployment of Omni Alpha 5.0..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
REQUIRED_VERSION="3.12.10"

if [ "$PYTHON_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION required, found $PYTHON_VERSION"
    exit 1
fi

# Backup existing deployment
if [ -d "/opt/omni-alpha" ]; then
    echo "📦 Backing up existing deployment..."
    sudo cp -r /opt/omni-alpha /opt/omni-alpha-backup-$(date +%Y%m%d-%H%M%S)
fi

# Create deployment directory
sudo mkdir -p /opt/omni-alpha
sudo chown $USER:$USER /opt/omni-alpha

# Copy application files
echo "📂 Copying application files..."
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='logs/*' --exclude='data/*' --exclude='.env.local' \
    ./ /opt/omni-alpha/

# Setup environment
cd /opt/omni-alpha

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v --tb=short

# Check application startup
echo "✅ Checking application startup..."
timeout 10 python -c "from src.app import app; print('App imports successfully')"

# Setup systemd service
echo "⚙️ Setting up systemd service..."
sudo cp deployment/omni-alpha.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable omni-alpha

# Setup nginx (if needed)
if [ -f "deployment/nginx.conf" ]; then
    echo "🌐 Setting up nginx..."
    sudo cp deployment/nginx.conf /etc/nginx/sites-available/omni-alpha
    sudo ln -sf /etc/nginx/sites-available/omni-alpha /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl reload nginx
fi

# Start services
echo "🚀 Starting services..."
sudo systemctl start omni-alpha

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Service is ready!"
        break
    fi
    sleep 1
done

# Verify deployment
echo "🔍 Verifying deployment..."
curl -s http://localhost:8000/health | python -m json.tool

# Run smoke tests
echo "🔥 Running smoke tests..."
python scripts/smoke_tests.py

echo "✅ Production deployment completed successfully!"
echo "📊 Monitoring available at:"
echo "   - Application: http://localhost:8000"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana: http://localhost:3000"

# Display service status
sudo systemctl status omni-alpha --no-pager

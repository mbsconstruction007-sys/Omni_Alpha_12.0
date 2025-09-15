# ============================================
# OMNI ALPHA 5.0 - STEP 10 DEPLOYMENT SCRIPT (PowerShell)
# ============================================

param(
    [switch]$SkipPrerequisites,
    [switch]$SkipInfrastructure,
    [switch]$SkipOrchestrator
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"

Write-Host "🚀 Starting Omni Alpha 5.0 - Step 10: Final Convergence" -ForegroundColor $Green

# Check prerequisites
function Check-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor $Yellow
    
    # Check Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker is not installed" -ForegroundColor $Red
        exit 1
    }
    
    # Check Docker Compose
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Host "Docker Compose is not installed" -ForegroundColor $Red
        exit 1
    }
    
    # Check Python
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "Python is not installed" -ForegroundColor $Red
        exit 1
    }
    
    Write-Host "Prerequisites check passed" -ForegroundColor $Green
}

# Install dependencies
function Install-Dependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor $Yellow
    pip install -r requirements.txt
    Write-Host "Dependencies installed" -ForegroundColor $Green
}

# Setup environment
function Setup-Environment {
    Write-Host "Setting up environment..." -ForegroundColor $Yellow
    
    # Copy env file if not exists
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Host "Please update .env file with your configuration" -ForegroundColor $Yellow
        } else {
            Write-Host "No .env.example file found" -ForegroundColor $Red
            exit 1
        }
    }
    
    # Create necessary directories
    $directories = @("logs", "data", "config", "monitoring/prometheus", "monitoring/grafana/dashboards")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Host "Environment setup complete" -ForegroundColor $Green
}

# Start infrastructure
function Start-Infrastructure {
    Write-Host "Starting infrastructure services..." -ForegroundColor $Yellow
    
    # Start databases and message queues
    docker-compose up -d postgres mongodb redis timescaledb
    Start-Sleep -Seconds 10
    
    docker-compose up -d zookeeper kafka
    Start-Sleep -Seconds 10
    
    # Start monitoring
    docker-compose up -d prometheus grafana elasticsearch kibana
    Start-Sleep -Seconds 10
    
    Write-Host "Infrastructure services started" -ForegroundColor $Green
}

# Initialize databases
function Initialize-Databases {
    Write-Host "Initializing databases..." -ForegroundColor $Yellow
    
    # Run database migrations (if script exists)
    if (Test-Path "scripts/migrate_database.py") {
        python scripts/migrate_database.py
    }
    
    # Create indexes (if script exists)
    if (Test-Path "scripts/create_indexes.py") {
        python scripts/create_indexes.py
    }
    
    Write-Host "Databases initialized" -ForegroundColor $Green
}

# Start orchestrator
function Start-Orchestrator {
    Write-Host "Starting Master Orchestrator..." -ForegroundColor $Yellow
    
    # Build Docker image
    docker build -t omni-alpha/orchestrator:10.0.0 .
    
    # Start orchestrator
    docker-compose up -d orchestrator
    
    Write-Host "Master Orchestrator started" -ForegroundColor $Green
}

# Health check
function Test-Health {
    Write-Host "Performing health check..." -ForegroundColor $Yellow
    
    # Wait for services to be ready
    Start-Sleep -Seconds 30
    
    # Check orchestrator health
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:9000/health" -Method Get
        if ($response.status -eq "active") {
            Write-Host "System is healthy and ready" -ForegroundColor $Green
        } else {
            Write-Host "System health check failed" -ForegroundColor $Red
            Write-Host $response
            exit 1
        }
    } catch {
        Write-Host "Health check failed: $($_.Exception.Message)" -ForegroundColor $Red
        Write-Host "This is expected if the orchestrator is not fully started yet" -ForegroundColor $Yellow
    }
}

# Main execution
function Main {
    Write-Host "================================================" -ForegroundColor $Green
    Write-Host "    OMNI ALPHA 5.0 - STEP 10 DEPLOYMENT" -ForegroundColor $Green
    Write-Host "================================================" -ForegroundColor $Green
    
    if (-not $SkipPrerequisites) {
        Check-Prerequisites
    }
    
    Install-Dependencies
    Setup-Environment
    
    if (-not $SkipInfrastructure) {
        Start-Infrastructure
        Initialize-Databases
    }
    
    if (-not $SkipOrchestrator) {
        Start-Orchestrator
    }
    
    Test-Health
    
    Write-Host ""
    Write-Host "✅ Omni Alpha 5.0 Step 10 is now running!" -ForegroundColor $Green
    Write-Host ""
    Write-Host "Access points:" -ForegroundColor $Yellow
    Write-Host "  - Orchestrator API: http://localhost:9000" -ForegroundColor $Yellow
    Write-Host "  - Grafana Dashboard: http://localhost:3000 (admin/admin)" -ForegroundColor $Yellow
    Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor $Yellow
    Write-Host "  - Kibana: http://localhost:5601" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "To view logs:" -ForegroundColor $Yellow
    Write-Host "  docker-compose logs -f orchestrator" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "To stop the system:" -ForegroundColor $Yellow
    Write-Host "  docker-compose down" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "⚡ The system is evolving..." -ForegroundColor $Yellow
}

# Run main function
Main

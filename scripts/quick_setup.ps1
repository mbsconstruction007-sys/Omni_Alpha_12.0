# Quick setup script for Omni Alpha 12.0 - PowerShell Version

param(
    [string]$GitHubUser = "mbsconstruction007-sys",
    [string]$GitHubRepo = "Omni_Alpha_12.0"
)

Write-Host "🚀 Quick Setup for Omni Alpha 12.0" -ForegroundColor Green

Write-Host "📥 Step 1: Clone repository" -ForegroundColor Blue
if (-not (Test-Path $GitHubRepo)) {
    git clone "https://github.com/$GitHubUser/$GitHubRepo.git"
    Set-Location $GitHubRepo
} else {
    Write-Host "Repository already exists, updating..." -ForegroundColor Yellow
    Set-Location $GitHubRepo
    git pull origin main
}

Write-Host "🐍 Step 2: Set up Python environment" -ForegroundColor Blue
python -m venv venv

Write-Host "🔧 Step 3: Activate virtual environment" -ForegroundColor Blue
& ".\venv\Scripts\Activate.ps1"

Write-Host "📦 Step 4: Install dependencies" -ForegroundColor Blue
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "⚙️ Step 5: Set up environment configuration" -ForegroundColor Blue
if (-not (Test-Path ".env")) {
    Copy-Item "env.example" ".env"
    Write-Host "Created .env file. Please edit it with your configuration." -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists." -ForegroundColor Green
}

Write-Host "📁 Step 6: Create necessary directories" -ForegroundColor Blue
$directories = @("data", "logs", "models")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "🔧 Step 7: Set up git configuration" -ForegroundColor Blue
git config user.name $GitHubUser
git config user.email "$GitHubUser@users.noreply.github.com"

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env file with your configuration"
Write-Host "2. Run tests: python -m pytest tests/"
Write-Host "3. Start the system: python backend/main.py"
Write-Host "4. Or use Docker: docker-compose up -d"
Write-Host ""
Write-Host "Repository URL: https://github.com/$GitHubUser/$GitHubRepo" -ForegroundColor Cyan

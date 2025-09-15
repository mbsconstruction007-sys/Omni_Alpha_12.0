# =============================================================================
# OMNI ALPHA 5.0 - SECRETS SETUP SCRIPT (PowerShell)
# =============================================================================
# Secure secrets generation and management for production deployment
# =============================================================================

param(
    [string]$Environment = "production",
    [string]$SecretsDir = "./secrets",
    [string]$EnvFile = ".env"
)

# Configuration
$ErrorActionPreference = "Stop"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to generate secure random string
function New-SecureSecret {
    param(
        [int]$Length = 32,
        [string]$Charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
    )
    
    $secret = ""
    for ($i = 0; $i -lt $Length; $i++) {
        $secret += $Charset[(Get-Random -Maximum $Charset.Length)]
    }
    return $secret
}

# Function to generate API key
function New-ApiKey {
    param(
        [string]$Prefix = "omni_alpha"
    )
    
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $random = New-SecureSecret -Length 16
    return "${Prefix}_${timestamp}_${random}"
}

# Function to create secrets directory
function New-SecretsDirectory {
    Write-Status "Creating secrets directory..."
    
    if (-not (Test-Path $SecretsDir)) {
        New-Item -ItemType Directory -Path $SecretsDir -Force | Out-Null
        # Set directory permissions (Windows equivalent of chmod 700)
        $acl = Get-Acl $SecretsDir
        $acl.SetAccessRuleProtection($true, $false)
        $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")))
        Set-Acl -Path $SecretsDir -AclObject $acl
        Write-Success "Secrets directory created: $SecretsDir"
    } else {
        Write-Warning "Secrets directory already exists: $SecretsDir"
    }
}

# Function to generate database password
function New-DatabasePassword {
    $passwordFile = Join-Path $SecretsDir "database_password.txt"
    
    if (Test-Path $passwordFile) {
        Write-Warning "Database password already exists. Skipping generation."
        return
    }
    
    Write-Status "Generating database password..."
    $password = New-SecureSecret -Length 32
    $password | Out-File -FilePath $passwordFile -Encoding UTF8 -NoNewline
    # Set file permissions (Windows equivalent of chmod 600)
    $acl = Get-Acl $passwordFile
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")))
    Set-Acl -Path $passwordFile -AclObject $acl
    Write-Success "Database password generated and saved to: $passwordFile"
}

# Function to generate Redis password
function New-RedisPassword {
    $passwordFile = Join-Path $SecretsDir "redis_password.txt"
    
    if (Test-Path $passwordFile) {
        Write-Warning "Redis password already exists. Skipping generation."
        return
    }
    
    Write-Status "Generating Redis password..."
    $password = New-SecureSecret -Length 24
    $password | Out-File -FilePath $passwordFile -Encoding UTF8 -NoNewline
    # Set file permissions
    $acl = Get-Acl $passwordFile
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")))
    Set-Acl -Path $passwordFile -AclObject $acl
    Write-Success "Redis password generated and saved to: $passwordFile"
}

# Function to generate Grafana password
function New-GrafanaPassword {
    $passwordFile = Join-Path $SecretsDir "grafana_password.txt"
    
    if (Test-Path $passwordFile) {
        Write-Warning "Grafana password already exists. Skipping generation."
        return
    }
    
    Write-Status "Generating Grafana admin password..."
    $password = New-SecureSecret -Length 16
    $password | Out-File -FilePath $passwordFile -Encoding UTF8 -NoNewline
    # Set file permissions
    $acl = Get-Acl $passwordFile
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")))
    Set-Acl -Path $passwordFile -AclObject $acl
    Write-Success "Grafana password generated and saved to: $passwordFile"
}

# Function to generate API keys
function New-ApiKeys {
    $keysFile = Join-Path $SecretsDir "api_keys.txt"
    
    if (Test-Path $keysFile) {
        Write-Warning "API keys file already exists. Skipping generation."
        return
    }
    
    Write-Status "Generating API keys..."
    
    $content = @"
# =============================================================================
# OMNI ALPHA 5.0 - API KEYS
# =============================================================================
# Generated on: $(Get-Date)
# Environment: $Environment
# =============================================================================

# Application Keys
SECRET_KEY=$(New-SecureSecret -Length 32)
API_KEY=$(New-ApiKey -Prefix "omni_alpha")
JWT_SECRET=$(New-SecureSecret -Length 32)
ENCRYPTION_KEY=$(New-SecureSecret -Length 32)

# Exchange API Keys (CHANGE THESE TO YOUR ACTUAL KEYS)
BINANCE_API_KEY=your-binance-api-key-here
BINANCE_SECRET_KEY=your-binance-secret-key-here
COINBASE_API_KEY=your-coinbase-api-key-here
COINBASE_SECRET_KEY=your-coinbase-secret-key-here
COINBASE_PASSPHRASE=your-coinbase-passphrase-here
KRAKEN_API_KEY=your-kraken-api-key-here
KRAKEN_SECRET_KEY=your-kraken-secret-key-here
KRAKEN_OTP_SECRET=your-kraken-otp-secret-here

# Market Data API Keys (CHANGE THESE TO YOUR ACTUAL KEYS)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key-here
POLYGON_API_KEY=your-polygon-key-here
QUANDL_API_KEY=your-quandl-key-here

# Notification API Keys (CHANGE THESE TO YOUR ACTUAL KEYS)
TWILIO_ACCOUNT_SID=your-twilio-account-sid-here
TWILIO_AUTH_TOKEN=your-twilio-auth-token-here
FCM_SERVER_KEY=your-fcm-server-key-here
ALERT_TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here

# SMTP Credentials (CHANGE THESE TO YOUR ACTUAL CREDENTIALS)
SMTP_USERNAME=your-smtp-username-here
SMTP_PASSWORD=your-smtp-password-here
"@
    
    $content | Out-File -FilePath $keysFile -Encoding UTF8
    # Set file permissions
    $acl = Get-Acl $keysFile
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")))
    Set-Acl -Path $keysFile -AclObject $acl
    Write-Success "API keys generated and saved to: $keysFile"
}

# Function to create environment file
function New-EnvironmentFile {
    $envTemplate = "env.$Environment"
    
    if (-not (Test-Path $envTemplate)) {
        Write-Error "Environment template not found: $envTemplate"
        exit 1
    }
    
    if (Test-Path $EnvFile) {
        Write-Warning "Environment file already exists: $EnvFile"
        $overwrite = Read-Host "Do you want to overwrite it? (y/N)"
        if ($overwrite -notmatch "^[Yy]$") {
            Write-Status "Skipping environment file creation."
            return
        }
    }
    
    Write-Status "Creating environment file from template: $envTemplate"
    Copy-Item $envTemplate $EnvFile
    
    # Update database password
    $dbPasswordFile = Join-Path $SecretsDir "database_password.txt"
    if (Test-Path $dbPasswordFile) {
        $dbPassword = Get-Content $dbPasswordFile -Raw
        (Get-Content $EnvFile) -replace "CHANGE_THIS_PASSWORD", $dbPassword | Set-Content $EnvFile
        Write-Success "Database password updated in environment file"
    }
    
    # Update Redis password
    $redisPasswordFile = Join-Path $SecretsDir "redis_password.txt"
    if (Test-Path $redisPasswordFile) {
        $redisPassword = Get-Content $redisPasswordFile -Raw
        (Get-Content $EnvFile) -replace "CHANGE_THIS_REDIS_PASSWORD", $redisPassword | Set-Content $EnvFile
        Write-Success "Redis password updated in environment file"
    }
    
    Write-Success "Environment file created: $EnvFile"
}

# Function to validate secrets
function Test-Secrets {
    Write-Status "Validating secrets..."
    
    $errors = 0
    
    # Check required secret files
    $requiredFiles = @(
        (Join-Path $SecretsDir "database_password.txt"),
        (Join-Path $SecretsDir "redis_password.txt"),
        (Join-Path $SecretsDir "grafana_password.txt"),
        (Join-Path $SecretsDir "api_keys.txt")
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Error "Required secret file missing: $file"
            $errors++
        } elseif ((Get-Item $file).Length -eq 0) {
            Write-Error "Secret file is empty: $file"
            $errors++
        }
    }
    
    # Check environment file
    if (-not (Test-Path $EnvFile)) {
        Write-Error "Environment file missing: $EnvFile"
        $errors++
    }
    
    if ($errors -eq 0) {
        Write-Success "All secrets validated successfully"
    } else {
        Write-Error "Validation failed with $errors errors"
        exit 1
    }
}

# Function to display security recommendations
function Show-SecurityRecommendations {
    Write-Status "Security Recommendations:"
    Write-Host ""
    Write-Host "1. 🔐 Change all default API keys in $SecretsDir\api_keys.txt"
    Write-Host "2. 🔒 Update exchange API credentials with your actual keys"
    Write-Host "3. 📧 Configure SMTP credentials for email notifications"
    Write-Host "4. 📱 Set up notification webhooks (Slack, Discord, Telegram)"
    Write-Host "5. 🛡️ Enable two-factor authentication on all exchange accounts"
    Write-Host "6. 🔄 Implement regular key rotation schedule"
    Write-Host "7. 📊 Set up monitoring and alerting systems"
    Write-Host "8. 🗄️ Configure secure database backups"
    Write-Host "9. 🌐 Use HTTPS/TLS for all external communications"
    Write-Host "10. 🔍 Regular security audits and penetration testing"
    Write-Host ""
    Write-Warning "NEVER commit secrets to version control!"
    Write-Warning "Keep secrets files secure and backed up safely!"
}

# Function to display next steps
function Show-NextSteps {
    Write-Status "Next Steps:"
    Write-Host ""
    Write-Host "1. 📝 Edit $SecretsDir\api_keys.txt with your actual API keys"
    Write-Host "2. 🔧 Update $EnvFile with your specific configuration"
    Write-Host "3. 🐳 Run: docker-compose -f docker-compose.production.yml up -d"
    Write-Host "4. 📊 Access Grafana at: http://localhost:3000"
    Write-Host "5. 📈 Access Prometheus at: http://localhost:9090"
    Write-Host "6. 🔍 Check logs: docker-compose logs -f omni-alpha-app"
    Write-Host ""
    Write-Success "Setup completed successfully!"
}

# Main execution
function Main {
    Write-Host "============================================================================="
    Write-Host "🚀 OMNI ALPHA 5.0 - SECRETS SETUP SCRIPT (PowerShell)"
    Write-Host "============================================================================="
    Write-Host "Environment: $Environment"
    Write-Host "Secrets Directory: $SecretsDir"
    Write-Host "Environment File: $EnvFile"
    Write-Host "============================================================================="
    Write-Host ""
    
    # Create secrets directory
    New-SecretsDirectory
    
    # Generate secrets
    New-DatabasePassword
    New-RedisPassword
    New-GrafanaPassword
    New-ApiKeys
    
    # Create environment file
    New-EnvironmentFile
    
    # Validate secrets
    Test-Secrets
    
    # Show recommendations and next steps
    Write-Host ""
    Show-SecurityRecommendations
    Write-Host ""
    Show-NextSteps
}

# Run main function
Main

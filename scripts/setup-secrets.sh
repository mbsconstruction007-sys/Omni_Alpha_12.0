#!/bin/bash
# =============================================================================
# OMNI ALPHA 5.0 - SECRETS SETUP SCRIPT
# =============================================================================
# Secure secrets generation and management for production deployment
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SECRETS_DIR="./secrets"
ENV_FILE=".env"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to generate secure random string
generate_secret() {
    local length=${1:-32}
    local charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
    local secret=""
    
    for ((i=0; i<length; i++)); do
        secret="${secret}${charset:$((RANDOM % ${#charset})):1}"
    done
    
    echo "$secret"
}

# Function to generate API key
generate_api_key() {
    local prefix=${1:-"omni_alpha"}
    local timestamp=$(date +%s)
    local random=$(generate_secret 16)
    echo "${prefix}_${timestamp}_${random}"
}

# Function to create secrets directory
create_secrets_directory() {
    print_status "Creating secrets directory..."
    
    if [ ! -d "$SECRETS_DIR" ]; then
        mkdir -p "$SECRETS_DIR"
        chmod 700 "$SECRETS_DIR"
        print_success "Secrets directory created: $SECRETS_DIR"
    else
        print_warning "Secrets directory already exists: $SECRETS_DIR"
    fi
}

# Function to generate database password
generate_database_password() {
    local password_file="$SECRETS_DIR/database_password.txt"
    
    if [ -f "$password_file" ]; then
        print_warning "Database password already exists. Skipping generation."
        return
    fi
    
    print_status "Generating database password..."
    local password=$(generate_secret 32)
    echo "$password" > "$password_file"
    chmod 600 "$password_file"
    print_success "Database password generated and saved to: $password_file"
}

# Function to generate Redis password
generate_redis_password() {
    local password_file="$SECRETS_DIR/redis_password.txt"
    
    if [ -f "$password_file" ]; then
        print_warning "Redis password already exists. Skipping generation."
        return
    fi
    
    print_status "Generating Redis password..."
    local password=$(generate_secret 24)
    echo "$password" > "$password_file"
    chmod 600 "$password_file"
    print_success "Redis password generated and saved to: $password_file"
}

# Function to generate Grafana password
generate_grafana_password() {
    local password_file="$SECRETS_DIR/grafana_password.txt"
    
    if [ -f "$password_file" ]; then
        print_warning "Grafana password already exists. Skipping generation."
        return
    fi
    
    print_status "Generating Grafana admin password..."
    local password=$(generate_secret 16)
    echo "$password" > "$password_file"
    chmod 600 "$password_file"
    print_success "Grafana password generated and saved to: $password_file"
}

# Function to generate API keys
generate_api_keys() {
    local keys_file="$SECRETS_DIR/api_keys.txt"
    
    if [ -f "$keys_file" ]; then
        print_warning "API keys file already exists. Skipping generation."
        return
    fi
    
    print_status "Generating API keys..."
    
    cat > "$keys_file" << EOF
# =============================================================================
# OMNI ALPHA 5.0 - API KEYS
# =============================================================================
# Generated on: $(date)
# Environment: $ENVIRONMENT
# =============================================================================

# Application Keys
SECRET_KEY=$(generate_secret 32)
API_KEY=$(generate_api_key "omni_alpha")
JWT_SECRET=$(generate_secret 32)
ENCRYPTION_KEY=$(generate_secret 32)

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
EOF
    
    chmod 600 "$keys_file"
    print_success "API keys generated and saved to: $keys_file"
}

# Function to create environment file
create_environment_file() {
    local env_template="env.$ENVIRONMENT"
    
    if [ ! -f "$env_template" ]; then
        print_error "Environment template not found: $env_template"
        exit 1
    fi
    
    if [ -f "$ENV_FILE" ]; then
        print_warning "Environment file already exists: $ENV_FILE"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping environment file creation."
            return
        fi
    fi
    
    print_status "Creating environment file from template: $env_template"
    cp "$env_template" "$ENV_FILE"
    
    # Update database password
    if [ -f "$SECRETS_DIR/database_password.txt" ]; then
        local db_password=$(cat "$SECRETS_DIR/database_password.txt")
        sed -i "s/CHANGE_THIS_PASSWORD/$db_password/g" "$ENV_FILE"
        print_success "Database password updated in environment file"
    fi
    
    # Update Redis password
    if [ -f "$SECRETS_DIR/redis_password.txt" ]; then
        local redis_password=$(cat "$SECRETS_DIR/redis_password.txt")
        sed -i "s/CHANGE_THIS_REDIS_PASSWORD/$redis_password/g" "$ENV_FILE"
        print_success "Redis password updated in environment file"
    fi
    
    print_success "Environment file created: $ENV_FILE"
}

# Function to validate secrets
validate_secrets() {
    print_status "Validating secrets..."
    
    local errors=0
    
    # Check required secret files
    local required_files=(
        "$SECRETS_DIR/database_password.txt"
        "$SECRETS_DIR/redis_password.txt"
        "$SECRETS_DIR/grafana_password.txt"
        "$SECRETS_DIR/api_keys.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required secret file missing: $file"
            ((errors++))
        elif [ ! -s "$file" ]; then
            print_error "Secret file is empty: $file"
            ((errors++))
        fi
    done
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file missing: $ENV_FILE"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "All secrets validated successfully"
    else
        print_error "Validation failed with $errors errors"
        exit 1
    fi
}

# Function to display security recommendations
show_security_recommendations() {
    print_status "Security Recommendations:"
    echo
    echo "1. 🔐 Change all default API keys in $SECRETS_DIR/api_keys.txt"
    echo "2. 🔒 Update exchange API credentials with your actual keys"
    echo "3. 📧 Configure SMTP credentials for email notifications"
    echo "4. 📱 Set up notification webhooks (Slack, Discord, Telegram)"
    echo "5. 🛡️ Enable two-factor authentication on all exchange accounts"
    echo "6. 🔄 Implement regular key rotation schedule"
    echo "7. 📊 Set up monitoring and alerting systems"
    echo "8. 🗄️ Configure secure database backups"
    echo "9. 🌐 Use HTTPS/TLS for all external communications"
    echo "10. 🔍 Regular security audits and penetration testing"
    echo
    print_warning "NEVER commit secrets to version control!"
    print_warning "Keep secrets files secure and backed up safely!"
}

# Function to display next steps
show_next_steps() {
    print_status "Next Steps:"
    echo
    echo "1. 📝 Edit $SECRETS_DIR/api_keys.txt with your actual API keys"
    echo "2. 🔧 Update $ENV_FILE with your specific configuration"
    echo "3. 🐳 Run: docker-compose -f docker-compose.production.yml up -d"
    echo "4. 📊 Access Grafana at: http://localhost:3000"
    echo "5. 📈 Access Prometheus at: http://localhost:9090"
    echo "6. 🔍 Check logs: docker-compose logs -f omni-alpha-app"
    echo
    print_success "Setup completed successfully!"
}

# Main execution
main() {
    echo "============================================================================="
    echo "🚀 OMNI ALPHA 5.0 - SECRETS SETUP SCRIPT"
    echo "============================================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Secrets Directory: $SECRETS_DIR"
    echo "Environment File: $ENV_FILE"
    echo "============================================================================="
    echo
    
    # Create secrets directory
    create_secrets_directory
    
    # Generate secrets
    generate_database_password
    generate_redis_password
    generate_grafana_password
    generate_api_keys
    
    # Create environment file
    create_environment_file
    
    # Validate secrets
    validate_secrets
    
    # Show recommendations and next steps
    echo
    show_security_recommendations
    echo
    show_next_steps
}

# Run main function
main "$@"

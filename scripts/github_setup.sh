#!/bin/bash
# Complete GitHub Setup for Omni Alpha 12.0

set -e

echo "🚀 Setting up Omni Alpha 12.0 GitHub Integration"

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

# Function to make GitHub API calls
github_api() {
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github.v3+json" \
         "$@"
}

# 1. Update repository settings
echo -e "${YELLOW}Configuring repository settings...${NC}"
github_api -X PATCH \
    "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO" \
    -d '{
        "has_issues": true,
        "has_projects": true,
        "has_wiki": true,
        "has_downloads": true,
        "allow_squash_merge": true,
        "allow_merge_commit": true,
        "allow_rebase_merge": true,
        "delete_branch_on_merge": true,
        "has_discussions": true
    }'

# 2. Create labels
echo -e "${YELLOW}Creating labels...${NC}"
labels=(
    '{"name":"bug","color":"d73a4a","description":"Something isnt working"}'
    '{"name":"enhancement","color":"a2eeef","description":"New feature or request"}'
    '{"name":"documentation","color":"0075ca","description":"Improvements or additions to documentation"}'
    '{"name":"good first issue","color":"7057ff","description":"Good for newcomers"}'
    '{"name":"help wanted","color":"008672","description":"Extra attention is needed"}'
    '{"name":"strategy","color":"fbca04","description":"Trading strategy related"}'
    '{"name":"risk","color":"e11d21","description":"Risk management related"}'
    '{"name":"ai","color":"c5def5","description":"AI/ML related"}'
    '{"name":"infrastructure","color":"bfd4f2","description":"Infrastructure related"}'
    '{"name":"security","color":"d93f0b","description":"Security related"}'
    '{"name":"performance","color":"0e8a16","description":"Performance optimization"}'
    '{"name":"urgent","color":"e11d21","description":"Urgent issue"}'
    '{"name":"wontfix","color":"ffffff","description":"This will not be worked on"}'
)

for label in "${labels[@]}"; do
    github_api -X POST \
        "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/labels" \
        -d "$label"
done

# 3. Create milestones
echo -e "${YELLOW}Creating milestones...${NC}"
milestones=(
    '{"title":"v1.0.0 - Foundation","description":"Core infrastructure and basic functionality","due_on":"2024-02-01T00:00:00Z"}'
    '{"title":"v2.0.0 - Intelligence","description":"AI and ML integration","due_on":"2024-03-01T00:00:00Z"}'
    '{"title":"v3.0.0 - Scale","description":"Institutional features and scaling","due_on":"2024-04-01T00:00:00Z"}'
    '{"title":"v12.0.0 - Dominance","description":"Global market dominance features","due_on":"2024-06-01T00:00:00Z"}'
)

for milestone in "${milestones[@]}"; do
    github_api -X POST \
        "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/milestones" \
        -d "$milestone"
done

# 4. Set up branch protection
echo -e "${YELLOW}Setting up branch protection...${NC}"
github_api -X PUT \
    "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/branches/main/protection" \
    -d '{
        "required_status_checks": {
            "strict": true,
            "contexts": ["continuous-integration/ci", "security/snyk"]
        },
        "enforce_admins": false,
        "required_pull_request_reviews": {
            "dismissal_restrictions": {},
            "dismiss_stale_reviews": true,
            "require_code_owner_reviews": true,
            "required_approving_review_count": 1
        },
        "restrictions": null,
        "allow_force_pushes": false,
        "allow_deletions": false
    }'

# 5. Create initial issues
echo -e "${YELLOW}Creating initial issues...${NC}"
issues=(
    '{"title":"Set up CI/CD pipeline","body":"Implement GitHub Actions for continuous integration and deployment","labels":["enhancement","infrastructure"],"milestone":1}'
    '{"title":"Implement core trading engine","body":"Build the foundation trading system","labels":["enhancement","strategy"],"milestone":1}'
    '{"title":"Add risk management system","body":"Implement comprehensive risk controls","labels":["enhancement","risk"],"milestone":1}'
    '{"title":"Create AI brain integration","body":"Integrate AI/ML capabilities","labels":["enhancement","ai"],"milestone":2}'
    '{"title":"Build monitoring dashboard","body":"Create Grafana dashboards for monitoring","labels":["enhancement","infrastructure"],"milestone":1}'
)

for issue in "${issues[@]}"; do
    github_api -X POST \
        "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/issues" \
        -d "$issue"
done

# 6. Create GitHub Actions secrets placeholder
echo -e "${YELLOW}Note: Add these secrets to your repository:${NC}"
echo "  - AWS_ACCESS_KEY_ID"
echo "  - AWS_SECRET_ACCESS_KEY"
echo "  - DOCKER_USERNAME"
echo "  - DOCKER_PASSWORD"
echo "  - ALPACA_API_KEY"
echo "  - ALPACA_SECRET_KEY"
echo "  - SLACK_WEBHOOK"

echo -e "${GREEN}✅ GitHub setup complete!${NC}"

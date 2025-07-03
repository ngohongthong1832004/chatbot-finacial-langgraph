#!/bin/bash

# Make all scripts executable
chmod +x scripts/deploy-staging.sh
chmod +x scripts/deploy-production.sh
chmod +x scripts/health-monitor.sh
chmod +x scripts/backup.sh

echo "All scripts are now executable"

# Create necessary directories
mkdir -p logs
mkdir -p nginx/ssl

echo "Directory structure created"

# Set up Git hooks (optional)
if [ -d .git ]; then
    echo "Setting up Git hooks..."
    # You can add pre-commit hooks here
fi

echo "Setup complete!"

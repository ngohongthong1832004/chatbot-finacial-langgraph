#!/bin/bash

set -e

PROJECT_NAME="chatbot-langgraph"
PRODUCTION_PORT_BACKEND="8000"
PRODUCTION_PORT_FRONTEND="3000"

echo "🚀 Starting production deployment..."

# Function to wait for service to be healthy
wait_for_health() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    echo "🔍 Waiting for $service_name to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f $url >/dev/null 2>&1; then
            echo "✅ $service_name is healthy!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name health check failed after $max_attempts attempts"
    return 1
}

# Backup function
backup_current() {
    echo "💾 Creating backup of current production..."
    
    # Backup current production images
    docker tag ${PROJECT_NAME}-backend:production ${PROJECT_NAME}-backend:backup-$(date +%Y%m%d-%H%M%S) || true
    docker tag ${PROJECT_NAME}-frontend:production ${PROJECT_NAME}-frontend:backup-$(date +%Y%m%d-%H%M%S) || true
    docker tag ${PROJECT_NAME}-backend:production ${PROJECT_NAME}-backend:backup || true
    docker tag ${PROJECT_NAME}-frontend:production ${PROJECT_NAME}-frontend:backup || true
    
    # Backup database
    if docker ps | grep -q postgres-production; then
        echo "📊 Backing up production database..."
        docker exec postgres-production pg_dump -U postgres chatbot_production > /opt/backups/db-backup-$(date +%Y%m%d-%H%M%S).sql
        echo "✅ Database backup completed"
    fi
}

# Rollback function
rollback() {
    echo "🔄 Rolling back to previous version..."
    
    # Stop current containers
    docker-compose -f docker-compose.production.yml down
    
    # Restore backup images
    if docker images | grep -q "${PROJECT_NAME}-backend:backup"; then
        docker tag ${PROJECT_NAME}-backend:backup ${PROJECT_NAME}-backend:production
        docker tag ${PROJECT_NAME}-frontend:backup ${PROJECT_NAME}-frontend:production
        
        # Start with backup version
        docker-compose -f docker-compose.production.yml up -d
        
        # Wait and check health
        sleep 30
        if wait_for_health "http://localhost:$PRODUCTION_PORT_BACKEND/health" "Backend" && \
           wait_for_health "http://localhost:$PRODUCTION_PORT_FRONTEND" "Frontend"; then
            echo "✅ Rollback completed successfully"
            echo "$(date): Production rollback successful" >> /var/log/deployment.log
            return 0
        else
            echo "❌ Rollback health check failed"
            return 1
        fi
    else
        echo "❌ No backup images found for rollback"
        return 1
    fi
}

# Main deployment function
main() {
    echo "📋 Pre-deployment checks..."
    
    # Check if required files exist
    if [ ! -f "docker-compose.production.yml" ]; then
        echo "❌ docker-compose.production.yml not found!"
        exit 1
    fi
    
    # Load environment variables
    if [ -f ".env.production" ]; then
        source .env.production
        echo "✅ Loaded production environment variables"
    fi
    
    # Verify staging is healthy before promoting
    echo "🔍 Verifying staging environment..."
    if ! curl -f http://localhost:8001/health >/dev/null 2>&1; then
        echo "❌ Staging environment is not healthy! Aborting production deployment."
        exit 1
    fi
    echo "✅ Staging environment verified"
    
    # Create backup
    backup_current
    
    # Promote staging images to production
    echo "📦 Promoting staging images to production..."
    docker tag ${PROJECT_NAME}-backend:staging ${PROJECT_NAME}-backend:production
    docker tag ${PROJECT_NAME}-frontend:staging ${PROJECT_NAME}-frontend:production
    
    # Blue-green deployment: start new production containers
    echo "🔄 Starting new production containers..."
    docker-compose -f docker-compose.production.yml up -d --force-recreate
    
    # Wait for services to start
    echo "⏳ Waiting for services to initialize..."
    sleep 30
    
    # Health checks
    echo "🩺 Performing production health checks..."
    
    if wait_for_health "http://localhost:$PRODUCTION_PORT_BACKEND/health" "Backend"; then
        echo "✅ Production backend health check passed"
    else
        echo "❌ Production backend health check failed - initiating rollback"
        rollback
        exit 1
    fi
    
    if wait_for_health "http://localhost:$PRODUCTION_PORT_FRONTEND" "Frontend"; then
        echo "✅ Production frontend health check passed"
    else
        echo "❌ Production frontend health check failed - initiating rollback"
        rollback
        exit 1
    fi
    
    # Additional production checks
    echo "🔍 Running additional production checks..."
    
    # Check database connectivity
    if docker-compose -f docker-compose.production.yml exec -T backend python -c "
import requests
try:
    response = requests.get('http://localhost:8000/health/db')
    if response.status_code == 200:
        print('Database connectivity: OK')
    else:
        print('Database connectivity: FAILED')
        exit(1)
except Exception as e:
    print(f'Database connectivity error: {e}')
    exit(1)
" 2>/dev/null; then
        echo "✅ Database connectivity check passed"
    else
        echo "❌ Database connectivity check failed - initiating rollback"
        rollback
        exit 1
    fi
    
    # Load testing (basic)
    echo "⚡ Running basic load test..."
    for i in {1..10}; do
        if ! curl -f http://localhost:$PRODUCTION_PORT_BACKEND/health >/dev/null 2>&1; then
            echo "❌ Load test failed - initiating rollback"
            rollback
            exit 1
        fi
    done
    echo "✅ Basic load test passed"
    
    # Cleanup old images (keep last 3 versions)
    echo "🧹 Cleaning up old images..."
    docker images | grep "${PROJECT_NAME}-backend" | grep "backup-" | tail -n +4 | awk '{print $3}' | xargs -r docker rmi || true
    docker images | grep "${PROJECT_NAME}-frontend" | grep "backup-" | tail -n +4 | awk '{print $3}' | xargs -r docker rmi || true
    
    echo "🎉 Production deployment completed successfully!"
    echo "🌐 Production URL: http://localhost:$PRODUCTION_PORT_FRONTEND"
    echo "🔧 Backend URL: http://localhost:$PRODUCTION_PORT_BACKEND"
    
    # Log successful deployment
    echo "$(date): Production deployment successful" >> /var/log/deployment.log
    
    # Optional: Send notification (Slack, email, etc.)
    # send_notification "Production deployment successful"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    *)
        echo "Usage: $0 [deploy|rollback]"
        exit 1
        ;;
esac

#!/bin/bash

set -e

PROJECT_NAME="chatbot-langgraph"
STAGING_PORT_BACKEND="8001"
STAGING_PORT_FRONTEND="3001"

echo "🚀 Starting staging deployment..."

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        echo "⚠️  Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

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

# Cleanup function
cleanup_staging() {
    echo "🧹 Cleaning up staging environment..."
    docker-compose -f docker-compose.staging.yml down || true
    docker system prune -f || true
}

# Main deployment logic
main() {
    echo "📋 Pre-deployment checks..."
    
    # Check if required files exist
    if [ ! -f "docker-compose.staging.yml" ]; then
        echo "❌ docker-compose.staging.yml not found!"
        exit 1
    fi
    
    # Load environment variables
    if [ -f ".env.staging" ]; then
        source .env.staging
        echo "✅ Loaded staging environment variables"
    fi
    
    # Stop any existing staging containers
    echo "🛑 Stopping existing staging containers..."
    docker-compose -f docker-compose.staging.yml down || true
    
    # Pull/load latest images
    echo "📦 Loading latest Docker images..."
    if [ -f "/tmp/backend-image.tar" ]; then
        docker load < /tmp/backend-image.tar
        docker tag ${PROJECT_NAME}-backend:* ${PROJECT_NAME}-backend:staging
    fi
    
    if [ -f "/tmp/frontend-image.tar" ]; then
        docker load < /tmp/frontend-image.tar
        docker tag ${PROJECT_NAME}-frontend:* ${PROJECT_NAME}-frontend:staging
    fi
    
    # Start staging environment
    echo "🚀 Starting staging environment..."
    docker-compose -f docker-compose.staging.yml up -d
    
    # Wait for services to start
    echo "⏳ Waiting for services to initialize..."
    sleep 30
    
    # Health checks
    echo "🩺 Performing health checks..."
    
    if wait_for_health "http://localhost:$STAGING_PORT_BACKEND/health" "Backend"; then
        echo "✅ Backend health check passed"
    else
        echo "❌ Backend health check failed"
        cleanup_staging
        exit 1
    fi
    
    if wait_for_health "http://localhost:$STAGING_PORT_FRONTEND" "Frontend"; then
        echo "✅ Frontend health check passed"
    else
        echo "❌ Frontend health check failed"
        cleanup_staging
        exit 1
    fi
    
    # Additional checks
    echo "🔍 Running additional checks..."
    
    # Check database connectivity (if applicable)
    if docker-compose -f docker-compose.staging.yml exec -T backend-staging python -c "
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
        echo "⚠️  Database connectivity check skipped or failed"
    fi
    
    # Performance test
    echo "⚡ Running basic performance test..."
    response_time=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:$STAGING_PORT_BACKEND/health)
    if (( $(echo "$response_time < 5.0" | bc -l) )); then
        echo "✅ Performance test passed (${response_time}s)"
    else
        echo "⚠️  Performance test warning: slow response (${response_time}s)"
    fi
    
    echo "🎉 Staging deployment completed successfully!"
    echo "🌐 Frontend URL: http://localhost:$STAGING_PORT_FRONTEND"
    echo "🔧 Backend URL: http://localhost:$STAGING_PORT_BACKEND"
    
    # Log deployment info
    echo "$(date): Staging deployment successful" >> /var/log/deployment.log
}

# Trap to cleanup on exit
trap cleanup_staging EXIT

# Run main function
main "$@"

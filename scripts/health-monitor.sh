#!/bin/bash

# Health monitoring script for production
# Run this script periodically via cron

LOG_FILE="/var/log/health-monitor.log"
PROJECT_NAME="chatbot-langgraph"
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
ALERT_EMAIL="admin@yourcompany.com"
SLACK_WEBHOOK_URL=""  # Add your Slack webhook URL

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# Function to send alerts
send_alert() {
    local message="$1"
    local severity="$2"
    
    log_message "ALERT [$severity]: $message"
    
    # Email alert (requires mailutils or sendmail)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "[$severity] Chatbot Service Alert" $ALERT_EMAIL
    fi
    
    # Slack alert
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 [$severity] Chatbot Service Alert\\n$message\"}" \
            $SLACK_WEBHOOK_URL
    fi
}

# Function to check service health
check_service_health() {
    local service_name="$1"
    local url="$2"
    local expected_status="$3"
    
    log_message "Checking $service_name health..."
    
    response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$url")
    
    if [ "$response" = "$expected_status" ]; then
        log_message "✅ $service_name is healthy (HTTP $response)"
        return 0
    else
        log_message "❌ $service_name health check failed (HTTP $response)"
        send_alert "$service_name is unhealthy. HTTP response: $response" "CRITICAL"
        return 1
    fi
}

# Function to check Docker containers
check_containers() {
    log_message "Checking Docker containers..."
    
    containers=("fastapi-backend-production" "react-frontend-production" "postgres-production")
    
    for container in "${containers[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log_message "✅ Container $container is running"
        else
            log_message "❌ Container $container is not running"
            send_alert "Container $container is not running" "CRITICAL"
            
            # Try to restart the container
            log_message "Attempting to restart $container..."
            docker restart "$container"
            sleep 10
            
            if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
                log_message "✅ Container $container restarted successfully"
                send_alert "Container $container was automatically restarted" "WARNING"
            else
                log_message "❌ Failed to restart container $container"
                send_alert "Failed to restart container $container" "CRITICAL"
            fi
        fi
    done
}

# Function to check disk space
check_disk_space() {
    log_message "Checking disk space..."
    
    disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        send_alert "Disk space usage is critical: ${disk_usage}%" "CRITICAL"
    elif [ "$disk_usage" -gt 80 ]; then
        send_alert "Disk space usage is high: ${disk_usage}%" "WARNING"
    else
        log_message "✅ Disk space usage is normal: ${disk_usage}%"
    fi
}

# Function to check memory usage
check_memory() {
    log_message "Checking memory usage..."
    
    memory_usage=$(free | awk 'FNR==2{printf "%.0f", $3/$2*100}')
    
    if [ "$memory_usage" -gt 90 ]; then
        send_alert "Memory usage is critical: ${memory_usage}%" "CRITICAL"
    elif [ "$memory_usage" -gt 80 ]; then
        send_alert "Memory usage is high: ${memory_usage}%" "WARNING"
    else
        log_message "✅ Memory usage is normal: ${memory_usage}%"
    fi
}

# Function to check database connectivity
check_database() {
    log_message "Checking database connectivity..."
    
    if docker exec postgres-production pg_isready -U postgres &> /dev/null; then
        log_message "✅ Database is accessible"
    else
        log_message "❌ Database is not accessible"
        send_alert "Database is not accessible" "CRITICAL"
    fi
    
    # Check database connections
    connections=$(docker exec postgres-production psql -U postgres -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs)
    
    if [ ! -z "$connections" ] && [ "$connections" -gt 0 ]; then
        log_message "✅ Database has $connections active connections"
        
        if [ "$connections" -gt 100 ]; then
            send_alert "Database has too many connections: $connections" "WARNING"
        fi
    fi
}

# Function to check application performance
check_performance() {
    log_message "Checking application performance..."
    
    # Measure response time
    backend_time=$(curl -w "%{time_total}" -o /dev/null -s --max-time 10 "$BACKEND_URL/health")
    frontend_time=$(curl -w "%{time_total}" -o /dev/null -s --max-time 10 "$FRONTEND_URL")
    
    # Check if response times are acceptable
    if (( $(echo "$backend_time > 5.0" | bc -l) )); then
        send_alert "Backend response time is slow: ${backend_time}s" "WARNING"
    else
        log_message "✅ Backend response time: ${backend_time}s"
    fi
    
    if (( $(echo "$frontend_time > 3.0" | bc -l) )); then
        send_alert "Frontend response time is slow: ${frontend_time}s" "WARNING"
    else
        log_message "✅ Frontend response time: ${frontend_time}s"
    fi
}

# Main monitoring function
main() {
    log_message "========== Starting Health Check =========="
    
    # Check services
    backend_healthy=false
    frontend_healthy=false
    
    if check_service_health "Backend" "$BACKEND_URL/health" "200"; then
        backend_healthy=true
    fi
    
    if check_service_health "Frontend" "$FRONTEND_URL" "200"; then
        frontend_healthy=true
    fi
    
    # Only proceed with other checks if services are running
    if [ "$backend_healthy" = true ] && [ "$frontend_healthy" = true ]; then
        check_containers
        check_database
        check_performance
    fi
    
    # Always check system resources
    check_disk_space
    check_memory
    
    log_message "========== Health Check Complete =========="
}

# Run the monitoring
main "$@"

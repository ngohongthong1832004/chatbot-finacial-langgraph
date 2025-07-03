#!/bin/bash

# Backup script for production database and application data
# Run daily via cron: 0 2 * * * /opt/chatbot-langgraph/scripts/backup.sh

PROJECT_NAME="chatbot-langgraph"
BACKUP_DIR="/opt/backups"
DATABASE_CONTAINER="postgres-production"
DATABASE_NAME="chatbot_production"
DATABASE_USER="postgres"
RETENTION_DAYS=7

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $BACKUP_DIR/backup.log
}

# Function to cleanup old backups
cleanup_old_backups() {
    log_message "Cleaning up backups older than $RETENTION_DAYS days..."
    find $BACKUP_DIR -name "*.sql" -type f -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
    log_message "Cleanup completed"
}

# Function to backup database
backup_database() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/db_backup_$timestamp.sql"
    
    log_message "Starting database backup..."
    
    if docker exec $DATABASE_CONTAINER pg_dump -U $DATABASE_USER $DATABASE_NAME > $backup_file; then
        log_message "✅ Database backup successful: $backup_file"
        
        # Compress the backup
        gzip $backup_file
        log_message "✅ Database backup compressed: $backup_file.gz"
        
        return 0
    else
        log_message "❌ Database backup failed"
        return 1
    fi
}

# Function to backup application data
backup_application_data() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/app_data_backup_$timestamp.tar.gz"
    
    log_message "Starting application data backup..."
    
    # Backup logs, uploads, and other important data
    tar -czf $backup_file \
        -C /opt/$PROJECT_NAME \
        --exclude='*.log' \
        --exclude='node_modules' \
        --exclude='.git' \
        --exclude='__pycache__' \
        API-chatbot-langgraph/logs \
        API-chatbot-langgraph/data \
        final-data \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_message "✅ Application data backup successful: $backup_file"
        return 0
    else
        log_message "❌ Application data backup failed"
        return 1
    fi
}

# Function to backup Docker images
backup_docker_images() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/docker_images_backup_$timestamp.tar"
    
    log_message "Starting Docker images backup..."
    
    # Save current production images
    docker save \
        $PROJECT_NAME-backend:production \
        $PROJECT_NAME-frontend:production \
        postgres:15-alpine \
        > $backup_file
    
    if [ $? -eq 0 ]; then
        gzip $backup_file
        log_message "✅ Docker images backup successful: $backup_file.gz"
        return 0
    else
        log_message "❌ Docker images backup failed"
        return 1
    fi
}

# Function to backup configuration files
backup_configurations() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/config_backup_$timestamp.tar.gz"
    
    log_message "Starting configuration backup..."
    
    tar -czf $backup_file \
        -C /opt/$PROJECT_NAME \
        docker-compose.production.yml \
        docker-compose.staging.yml \
        .env.production \
        .env.staging \
        nginx/nginx.conf \
        scripts/ \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_message "✅ Configuration backup successful: $backup_file"
        return 0
    else
        log_message "❌ Configuration backup failed"
        return 1
    fi
}

# Function to verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    if [ -f "$backup_file" ]; then
        local file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null)
        
        if [ "$file_size" -gt 0 ]; then
            log_message "✅ Backup verification passed: $backup_file ($file_size bytes)"
            return 0
        else
            log_message "❌ Backup verification failed: $backup_file is empty"
            return 1
        fi
    else
        log_message "❌ Backup verification failed: $backup_file not found"
        return 1
    fi
}

# Function to send backup report
send_backup_report() {
    local status="$1"
    local details="$2"
    
    # You can integrate with your monitoring system here
    log_message "Backup Status: $status"
    log_message "Details: $details"
    
    # Example: Send to monitoring endpoint
    # curl -X POST http://your-monitoring-system/backup-status \
    #      -H "Content-Type: application/json" \
    #      -d "{\"status\":\"$status\",\"details\":\"$details\",\"timestamp\":\"$(date -Iseconds)\"}"
}

# Main backup function
main() {
    log_message "========== Starting Backup Process =========="
    
    local backup_success=true
    local backup_details=""
    
    # Cleanup old backups first
    cleanup_old_backups
    
    # Database backup
    if backup_database; then
        backup_details="$backup_details Database: ✅"
    else
        backup_details="$backup_details Database: ❌"
        backup_success=false
    fi
    
    # Application data backup
    if backup_application_data; then
        backup_details="$backup_details AppData: ✅"
    else
        backup_details="$backup_details AppData: ❌"
        backup_success=false
    fi
    
    # Docker images backup (weekly on Sundays)
    if [ $(date +%u) -eq 7 ]; then
        if backup_docker_images; then
            backup_details="$backup_details Docker: ✅"
        else
            backup_details="$backup_details Docker: ❌"
            backup_success=false
        fi
    fi
    
    # Configuration backup
    if backup_configurations; then
        backup_details="$backup_details Config: ✅"
    else
        backup_details="$backup_details Config: ❌"
        backup_success=false
    fi
    
    # Send backup report
    if [ "$backup_success" = true ]; then
        send_backup_report "SUCCESS" "$backup_details"
        log_message "✅ All backups completed successfully"
    else
        send_backup_report "FAILURE" "$backup_details"
        log_message "❌ Some backups failed"
    fi
    
    log_message "========== Backup Process Complete =========="
    
    # Return appropriate exit code
    if [ "$backup_success" = true ]; then
        exit 0
    else
        exit 1
    fi
}

# Run the backup
main "$@"

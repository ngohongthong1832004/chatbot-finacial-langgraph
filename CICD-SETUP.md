# CI/CD Setup Guide

## Tổng quan

Hệ thống CI/CD này thực hiện quy trình tự động từ GitHub → GitLab → Server với staging và production environments.

## Kiến trúc CI/CD

```
GitHub (Push) → GitLab (Mirror) → Server Deployment
                                      ↓
                              Staging (Port 8001/3001)
                                      ↓
                              Health Check
                                   ↓    ↓
                              ✅ Pass  ❌ Fail
                                 ↓       ↓
                            Production  Rollback
                            (Port 8000/3000)
```

## Thiết lập từng bước

### 1. Cấu hình GitHub Repository

1. **Tạo GitHub Secrets:**
   ```
   GITLAB_REPO_URL=git@gitlab.com:username/repo.git
   GITLAB_SSH_PRIVATE_KEY=<your-gitlab-ssh-private-key>
   ```

2. **SSH Key Setup:**
   ```bash
   # Generate SSH key pair
   ssh-keygen -t rsa -b 4096 -C "github-to-gitlab@yourcompany.com"
   
   # Add public key to GitLab Deploy Keys
   # Add private key to GitHub Secrets
   ```

### 2. Cấu hình GitLab Repository

1. **Tạo GitLab Variables:**
   ```
   SSH_PRIVATE_KEY=<server-ssh-private-key>
   SERVER_HOST=<your-server-ip>
   SERVER_USER=<server-username>
   ```

2. **Tạo GitLab Runner (nếu cần):**
   ```bash
   # Trên server
   curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash
   sudo yum install gitlab-runner
   sudo gitlab-runner register
   ```

### 3. Cấu hình Server

1. **Cài đặt Docker và Docker Compose:**
   ```bash
   # CentOS/RHEL
   sudo yum install -y docker docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. **Tạo thư mục dự án:**
   ```bash
   sudo mkdir -p /opt/chatbot-langgraph
   sudo chown $USER:$USER /opt/chatbot-langgraph
   cd /opt/chatbot-langgraph
   ```

3. **Tạo thư mục logs và backups:**
   ```bash
   sudo mkdir -p /var/log/deployment
   sudo mkdir -p /opt/backups
   sudo chown $USER:$USER /var/log/deployment
   sudo chown $USER:$USER /opt/backups
   ```

4. **Cấu hình SSH:**
   ```bash
   # Add GitLab public key to authorized_keys
   mkdir -p ~/.ssh
   echo "ssh-rsa AAAAB3N... gitlab-ci" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   chmod 700 ~/.ssh
   ```

### 4. Environment Variables

1. **Copy và cấu hình environment files:**
   ```bash
   # Copy files từ repository
   cp .env.staging.example .env.staging
   cp .env.production.example .env.production
   
   # Cập nhật các giá trị thực tế
   nano .env.production
   nano .env.staging
   ```

2. **Cấu hình database passwords:**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

### 5. Database Setup

1. **Tạo database containers:**
   ```bash
   # Start database containers
   docker-compose -f docker-compose.staging.yml up -d db-staging
   docker-compose -f docker-compose.production.yml up -d db
   ```

2. **Initialize databases:**
   ```bash
   # Copy SQL initialization files
   docker cp ./DATABASE/init/ postgres-production:/docker-entrypoint-initdb.d/
   docker restart postgres-production
   ```

### 6. SSL/TLS Setup (Optional)

1. **Install Let's Encrypt certificates:**
   ```bash
   sudo apt install certbot
   sudo certbot certonly --standalone -d yourdomain.com
   ```

2. **Configure Nginx for HTTPS:**
   ```bash
   # Update nginx.conf with SSL configuration
   # Copy certificates to nginx/ssl/
   ```

## Quy trình Deployment

### Automated Deployment

1. **Push to GitHub main branch**
2. **GitHub Actions syncs to GitLab**
3. **GitLab CI/CD triggers:**
   - Build Docker images
   - Deploy to staging
   - Run health checks
   - Manual approval for production
   - Deploy to production

### Manual Operations

1. **Check staging:**
   ```bash
   curl http://your-server:8001/health
   curl http://your-server:3001
   ```

2. **Monitor deployment:**
   ```bash
   # Check logs
   docker-compose -f docker-compose.staging.yml logs -f
   
   # Check containers
   docker ps
   ```

3. **Manual rollback:**
   ```bash
   cd /opt/chatbot-langgraph
   ./scripts/deploy-production.sh rollback
   ```

## Monitoring và Maintenance

### 1. Health Monitoring

```bash
# Setup cron job for health monitoring
crontab -e

# Add line:
*/5 * * * * /opt/chatbot-langgraph/scripts/health-monitor.sh
```

### 2. Backup Schedule

```bash
# Setup daily backup
crontab -e

# Add line:
0 2 * * * /opt/chatbot-langgraph/scripts/backup.sh
```

### 3. Log Rotation

```bash
# Setup logrotate
sudo nano /etc/logrotate.d/chatbot-langgraph

# Content:
/var/log/deployment/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 user user
}
```

## Troubleshooting

### Common Issues

1. **Docker build fails:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Check disk space
   df -h
   ```

2. **Health checks fail:**
   ```bash
   # Check container logs
   docker logs fastapi-backend-staging
   docker logs react-frontend-staging
   
   # Check network connectivity
   docker network ls
   docker network inspect chatbot-langgraph_staging-network
   ```

3. **Database connection issues:**
   ```bash
   # Check database container
   docker exec postgres-production pg_isready -U postgres
   
   # Check connection from backend
   docker exec fastapi-backend-production python -c "from src.database.health import test_connection; print(test_connection())"
   ```

4. **Port conflicts:**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   lsof -i :8000
   
   # Kill process using port
   sudo kill -9 <PID>
   ```

## Security Considerations

1. **Firewall setup:**
   ```bash
   # UFW (Ubuntu)
   sudo ufw allow 22/tcp
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   
   # Restrict staging ports to internal network only
   sudo ufw allow from 10.0.0.0/8 to any port 8001
   sudo ufw allow from 10.0.0.0/8 to any port 3001
   ```

2. **Docker security:**
   ```bash
   # Run containers as non-root user
   # Use secrets for sensitive data
   # Regular security updates
   ```

3. **Database security:**
   ```bash
   # Strong passwords
   # Network isolation
   # Regular backups
   # Access logging
   ```

## Performance Optimization

1. **Docker optimizations:**
   - Use multi-stage builds
   - Optimize layer caching
   - Use .dockerignore

2. **Application optimizations:**
   - Enable gzip compression
   - Use CDN for static assets
   - Database connection pooling

3. **Server optimizations:**
   - Monitor resource usage
   - Scale containers as needed
   - Use SSD storage

## Monitoring Endpoints

- **Production Health:** `http://your-server:8000/health`
- **Staging Health:** `http://your-server:8001/health`
- **Database Health:** `http://your-server:8000/health/db`
- **Detailed Health:** `http://your-server:8000/health/detailed`

## Support

For issues or questions:
1. Check logs in `/var/log/deployment/`
2. Review container logs: `docker logs <container-name>`
3. Run health monitor: `./scripts/health-monitor.sh`
4. Check backup status: `tail -f /opt/backups/backup.log`

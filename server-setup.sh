#!/bin/bash

# Server setup script - Run this on your server (192.168.1.28)

set -e

echo "🚀 Setting up server for CI/CD deployment..."

# 1. Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# 3. Install Docker Compose
echo "🔧 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
else
    echo "✅ Docker Compose already installed"
fi

# 4. Create project directory
echo "📁 Creating project directories..."
sudo mkdir -p /opt/chatbot-langgraph
sudo chown $USER:$USER /opt/chatbot-langgraph
sudo mkdir -p /var/log/deployment
sudo chown $USER:$USER /var/log/deployment
sudo mkdir -p /opt/backups
sudo chown $USER:$USER /opt/backups

# 5. Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 8000/tcp # Production backend
sudo ufw allow 3000/tcp # Production frontend
sudo ufw allow 8001/tcp # Staging backend
sudo ufw allow 3001/tcp # Staging frontend
sudo ufw --force enable

# 6. Setup SSH for GitLab CI
echo "🔑 Setting up SSH for GitLab CI..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "📝 Add GitLab CI public key to ~/.ssh/authorized_keys manually"
echo "📝 Example: echo 'ssh-rsa AAAAB3N... gitlab-ci' >> ~/.ssh/authorized_keys"

# 7. Create logging directory
echo "📊 Setting up logging..."
sudo mkdir -p /var/log/nginx
sudo chown $USER:$USER /var/log/nginx

# 8. Install monitoring tools
echo "🔍 Installing monitoring tools..."
sudo apt install -y htop curl wget net-tools bc

# 9. Setup cron jobs (will be added later)
echo "⏰ Cron jobs setup ready..."

# 10. Test Docker
echo "🧪 Testing Docker installation..."
docker --version
docker-compose --version

echo "✅ Server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Add GitLab CI SSH public key to ~/.ssh/authorized_keys"
echo "2. Configure GitHub and GitLab secrets"
echo "3. Push code to trigger first deployment"
echo ""
echo "🔧 Server IP: $(hostname -I | awk '{print $1}')"
echo "🔧 User: $USER"

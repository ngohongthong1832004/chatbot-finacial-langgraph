# GitHub Secrets cần thiết

## Vào GitHub Repository Settings > Secrets and variables > Actions

### 1. GITLAB_REPO_URL
```
Ví dụ: git@gitlab.com:username/chatbot-langgraph.git
```

### 2. GITLAB_SSH_PRIVATE_KEY
```
Tạo SSH key pair:
ssh-keygen -t rsa -b 4096 -C "github-to-gitlab@yourcompany.com"

- Thêm public key vào GitLab Deploy Keys
- Thêm private key vào GitHub Secret này
```

## GitLab Variables cần thiết

### Vào GitLab Project Settings > CI/CD > Variables

1. **SSH_PRIVATE_KEY**: SSH private key để truy cập server
2. **SERVER_HOST**: 192.168.1.28
3. **SERVER_USER**: username để SSH vào server
4. **CI_COMMIT_SHORT_SHA**: (GitLab tự tạo)
5. **PROJECT_NAME**: chatbot-langgraph

### Tạo SSH key cho server access:
```bash
ssh-keygen -t rsa -b 4096 -C "gitlab-ci@yourcompany.com"
# Add public key to server: ~/.ssh/authorized_keys
# Add private key to GitLab Variables
```

# 安裝和部署指南

本文檔提供完整的安裝和部署步驟，符合 SSDLC 安全要求。

---

## 📋 目錄

1. [系統需求](#系統需求)
2. [開發環境安裝](#開發環境安裝)
3. [生產環境部署](#生產環境部署)
4. [安全配置](#安全配置)
5. [驗證安裝](#驗證安裝)
6. [常見問題](#常見問題)

---

## 🖥️ 系統需求

### 硬體需求

| 組件 | 最低要求 | 建議配置 |
|------|---------|---------|
| CPU | 4 核心 | 8 核心以上 |
| RAM | 8 GB | 16 GB 以上 |
| 硬碟 | 50 GB | 100 GB 以上 SSD |
| GPU | 無（CPU 模式） | NVIDIA GPU（CUDA 支援） |

### 軟體需求

- **作業系統**：
  - Linux (Ubuntu 20.04+, CentOS 8+)
  - Windows 10/11
  - macOS 11+

- **Python**：3.9 - 3.11

- **其他**：
  - FFmpeg 7.1.1+
  - Git
  - SMTP 郵件服務（Gmail, Outlook 等）

---

## 🚀 開發環境安裝

### 步驟 1：克隆專案

```bash
git clone <repository-url>
cd Speech-to-text
```

### 步驟 2：創建 Python 虛擬環境

#### Linux/macOS

```bash
# 創建虛擬環境
python3 -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate
```

#### Windows

```powershell
# 創建虛擬環境
python -m venv .venv

# 啟動虛擬環境
.venv\Scripts\activate
```

### 步驟 3：安裝依賴套件

```bash
cd remote_server
pip install -r requirements.txt
```

**注意**：首次安裝可能需要 10-30 分鐘，取決於網路速度。

### 步驟 4：配置環境變數

```bash
# 複製配置範例
cp .env.example .env

# 編輯 .env 文件
# Linux/macOS
nano .env

# Windows
notepad .env
```

**必須設置的環境變數**：

```bash
# 基礎配置
HUGGINGFACE_TOKEN=your_token_here
ADMIN_TOKEN=<生成 32 位隨機字符串>

# SMTP 配置
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com

# 安全配置
EMAIL_HASH_SALT=<生成 32 位隨機字符串>
ENCRYPTION_KEY=<首次運行自動生成>

# 開發環境設置
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
TRUSTED_HOSTS=localhost,127.0.0.1
ENABLE_DOCS=true
```

### 步驟 5：生成安全憑證

使用 Python 生成隨機字符串：

```python
import secrets

# 生成 ADMIN_TOKEN
print(f"ADMIN_TOKEN={secrets.token_hex(32)}")

# 生成 EMAIL_HASH_SALT
print(f"EMAIL_HASH_SALT={secrets.token_hex(32)}")
```

或使用命令行：

```bash
# Linux/macOS
python3 -c "import secrets; print('ADMIN_TOKEN=' + secrets.token_hex(32))"
python3 -c "import secrets; print('EMAIL_HASH_SALT=' + secrets.token_hex(32))"

# Windows
python -c "import secrets; print('ADMIN_TOKEN=' + secrets.token_hex(32))"
python -c "import secrets; print('EMAIL_HASH_SALT=' + secrets.token_hex(32))"
```

### 步驟 6：啟動後端服務

```bash
cd remote_server
python api.py
```

服務啟動後，訪問：
- API 文檔：http://localhost:8100/docs
- ReDoc 文檔：http://localhost:8100/redoc
- 健康檢查：http://localhost:8100/health

### 步驟 7：啟動前端（可選）

```bash
cd frontend
npm install
npm run dev
```

前端服務啟動後，訪問：http://localhost:5173

---

## 🏭 生產環境部署

### 方法 1：直接部署（使用 Uvicorn）

#### 1. 準備環境

```bash
# 創建部署目錄
sudo mkdir -p /opt/speech-to-text
sudo chown $USER:$USER /opt/speech-to-text

# 複製專案文件
cp -r remote_server /opt/speech-to-text/
cd /opt/speech-to-text/remote_server
```

#### 2. 配置生產環境變數

```bash
cp .env.example .env
nano .env
```

**生產環境重要設置**：

```bash
# 生產環境配置
ALLOWED_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com,www.yourdomain.com
ENABLE_DOCS=false  # 關閉 API 文檔

# 使用強密碼
ADMIN_TOKEN=<64位隨機字符串>
EMAIL_HASH_SALT=<64位隨機字符串>
```

#### 3. 使用 HTTPS 啟動

```bash
# 需要 SSL 憑證
uvicorn api:app \
  --host 0.0.0.0 \
  --port 8100 \
  --ssl-keyfile=/path/to/privkey.pem \
  --ssl-certfile=/path/to/fullchain.pem \
  --workers 4
```

#### 4. 使用 Systemd 服務（推薦）

創建服務文件：

```bash
sudo nano /etc/systemd/system/speech-to-text.service
```

內容：

```ini
[Unit]
Description=Speech-to-Text API Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/speech-to-text/remote_server
Environment="PATH=/opt/speech-to-text/.venv/bin"
ExecStart=/opt/speech-to-text/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8100
Restart=always
RestartSec=10

# 安全設置
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/speech-to-text/remote_server/logs /opt/speech-to-text/remote_server/uploads /opt/speech-to-text/remote_server/result

[Install]
WantedBy=multi-user.target
```

啟動服務：

```bash
sudo systemctl daemon-reload
sudo systemctl enable speech-to-text
sudo systemctl start speech-to-text
sudo systemctl status speech-to-text
```

### 方法 2：使用 Nginx 反向代理（推薦）

#### 1. 安裝 Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

#### 2. 配置 Nginx

創建配置文件：

```bash
sudo nano /etc/nginx/sites-available/speech-to-text
```

內容：

```nginx
upstream speech_to_text {
    server 127.0.0.1:8100;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL 憑證
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL 安全設置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # 安全標頭
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # 文件上傳大小限制
    client_max_body_size 500M;

    location / {
        proxy_pass http://speech_to_text;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 支援
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
```

啟用配置：

```bash
sudo ln -s /etc/nginx/sites-available/speech-to-text /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 3. 獲取 SSL 憑證（Let's Encrypt）

```bash
# 安裝 Certbot
sudo apt install certbot python3-certbot-nginx

# 獲取憑證
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 自動續期
sudo certbot renew --dry-run
```

### 方法 3：Docker 部署

#### 1. 創建 Dockerfile

```dockerfile
FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製依賴文件
COPY remote_server/requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式
COPY remote_server/ .

# 創建日誌目錄
RUN mkdir -p logs uploads result

# 暴露端口
EXPOSE 8100

# 啟動應用
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100"]
```

#### 2. 創建 docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8100:8100"
    env_file:
      - remote_server/.env
    volumes:
      - ./remote_server/logs:/app/logs
      - ./remote_server/uploads:/app/uploads
      - ./remote_server/result:/app/result
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

#### 3. 啟動容器

```bash
docker-compose up -d
```

---

## 🔒 安全配置

### 1. 防火牆設置

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 2. 文件權限設置

```bash
# 設置適當的文件權限
cd /opt/speech-to-text/remote_server

# 配置文件（僅擁有者可讀）
chmod 600 .env

# 日誌目錄
chmod 750 logs
chmod 640 logs/*.log

# 執行文件
chmod 644 *.py
```

### 3. 定期安全更新

```bash
# 更新系統套件
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
sudo yum update -y  # CentOS/RHEL

# 更新 Python 依賴
pip install --upgrade -r requirements.txt

# 檢查已知漏洞
pip-audit
```

### 4. 日誌監控設置

安裝日誌監控工具（可選）：

```bash
# 安裝 logrotate
sudo apt install logrotate

# 創建 logrotate 配置
sudo nano /etc/logrotate.d/speech-to-text
```

配置內容：

```
/opt/speech-to-text/remote_server/logs/*.log {
    daily
    rotate 365
    compress
    delaycompress
    notifempty
    create 640 www-data www-data
    sharedscripts
}
```

---

## ✅ 驗證安裝

### 1. 健康檢查

```bash
curl http://localhost:8100/health
```

預期輸出：

```json
{
  "status": "healthy",
  "queue_size": 0,
  "processing": false
}
```

### 2. API 測試

```bash
# 測試發送驗證碼
curl -X POST "http://localhost:8100/api/email/send-verification?email=test@example.com"
```

### 3. 安全標頭檢查

```bash
curl -I https://yourdomain.com/
```

檢查是否包含：
- `Strict-Transport-Security`
- `X-Frame-Options`
- `X-Content-Type-Options`
- `X-XSS-Protection`

### 4. SSL 檢查

使用 SSL Labs 測試：
https://www.ssllabs.com/ssltest/

目標：A+ 評級

---

## ❓ 常見問題

### Q1: 如何更換 ADMIN_TOKEN？

```bash
# 1. 生成新 Token
python -c "import secrets; print(secrets.token_hex(32))"

# 2. 更新 .env 文件
nano .env  # 更新 ADMIN_TOKEN

# 3. 重啟服務
sudo systemctl restart speech-to-text
```

### Q2: 如何查看日誌？

```bash
# 查看所有日誌
tail -f remote_server/logs/*.log

# 查看特定日誌
tail -f remote_server/logs/security.log
tail -f remote_server/logs/error.log
```

### Q3: 如何備份日誌？

```bash
# 手動備份
tar -czf logs-backup-$(date +%Y%m%d).tar.gz remote_server/logs/

# 自動備份（crontab）
# 每天凌晨 2 點備份
0 2 * * * tar -czf /backup/logs-$(date +\%Y\%m\%d).tar.gz /opt/speech-to-text/remote_server/logs/
```

### Q4: 如何升級到新版本？

```bash
# 1. 備份配置
cp remote_server/.env .env.backup

# 2. 拉取最新代碼
git pull origin main

# 3. 更新依賴
pip install --upgrade -r remote_server/requirements.txt

# 4. 重啟服務
sudo systemctl restart speech-to-text
```

### Q5: 如何檢查安全漏洞？

```bash
# 檢查 Python 依賴漏洞
cd remote_server
pip install pip-audit
pip-audit

# 檢查代碼安全
pip install bandit
bandit -r . -ll
```

### Q6: 忘記 ADMIN_TOKEN 怎麼辦？

```bash
# 查看當前 Token
grep ADMIN_TOKEN remote_server/.env

# 或重新生成（會使舊 Token 失效）
python -c "import secrets; print('ADMIN_TOKEN=' + secrets.token_hex(32))" >> remote_server/.env
```

---

## 📞 技術支援

如遇到安裝問題：

1. 檢查日誌文件：`remote_server/logs/error.log`
2. 查看 GitHub Issues
3. 參考 SECURITY.md 了解安全配置
4. 參考 SSDLC-COMPLIANCE.md 了解合規要求

---

**最後更新日期**：2025-01-10
**適用版本**：v2.1.0 (SSDLC Compliant)

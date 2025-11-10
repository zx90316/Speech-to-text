# 快速開始指南

5 分鐘快速設置 Speech-to-Text API 服務（符合 SSDLC 安全要求）

---

## 🚀 快速開始（開發環境）

### 前置要求

- Python 3.9-3.11
- Git
- Gmail 帳號（用於 SMTP）

### 步驟 1：克隆專案

```bash
git clone <repository-url>
cd Speech-to-text
```

### 步驟 2：創建虛擬環境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 步驟 3：安裝依賴

```bash
cd remote_server
pip install -r requirements.txt
```

⏱️ 預計時間：10-30 分鐘（取決於網路速度）

### 步驟 4：配置環境變數

```bash
# 複製配置範例
cp .env.example .env
```

編輯 `.env` 文件，最少需要設置：

```env
# 基礎配置
HUGGINGFACE_TOKEN=your_token_here
ADMIN_TOKEN=請使用下方命令生成

# SMTP 配置（Gmail 範例）
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
FROM_EMAIL=your_email@gmail.com

# 安全配置
EMAIL_HASH_SALT=請使用下方命令生成
```

**生成安全密鑰**：

```bash
python -c "import secrets; print('ADMIN_TOKEN=' + secrets.token_hex(32))"
python -c "import secrets; print('EMAIL_HASH_SALT=' + secrets.token_hex(32))"
```

**取得 Gmail 應用程式密碼**：

1. 訪問 https://myaccount.google.com/security
2. 啟用「兩步驟驗證」
3. 前往「應用程式密碼」
4. 生成「郵件」應用程式密碼
5. 複製 16 位密碼到 `SMTP_PASSWORD`

### 步驟 5：啟動服務

```bash
python api.py
```

服務啟動後訪問：
- **API 文檔**：http://localhost:8100/docs
- **健康檢查**：http://localhost:8100/health

### 步驟 6：測試 API

#### 測試 1：健康檢查

```bash
curl http://localhost:8100/health
```

預期輸出：
```json
{"status": "healthy", "queue_size": 0, "processing": false}
```

#### 測試 2：發送驗證碼

```bash
curl -X POST "http://localhost:8100/api/email/send-verification?email=your_email@example.com"
```

檢查您的郵箱，應該收到 6 位驗證碼。

#### 測試 3：驗證郵箱

```bash
curl -X POST "http://localhost:8100/api/email/verify-code?email=your_email@example.com&code=123456"
```

將 `123456` 替換為您收到的驗證碼。

---

## 🎯 快速測試完整流程

### 使用 API 文檔（推薦）

1. 訪問 http://localhost:8100/docs
2. 展開 **Email Verification** 區塊
3. 點擊 **POST /api/email/send-verification**
4. 點擊「Try it out」
5. 輸入您的郵箱，點擊「Execute」
6. 檢查郵箱，獲取驗證碼
7. 展開 **POST /api/email/verify-code**
8. 輸入郵箱和驗證碼，點擊「Execute」
9. 展開 **POST /api/tasks**
10. 上傳音頻文件，點擊「Execute」
11. 複製返回的 `task_id`
12. 使用 **GET /api/tasks/{task_id}** 查詢狀態
13. 等待郵件通知任務完成

### 使用 curl 命令

```bash
# 1. 發送驗證碼
curl -X POST "http://localhost:8100/api/email/send-verification?email=test@example.com"

# 2. 驗證郵箱（替換 123456 為實際驗證碼）
curl -X POST "http://localhost:8100/api/email/verify-code?email=test@example.com&code=123456"

# 3. 提交任務
curl -X POST "http://localhost:8100/api/tasks" \
  -H "Content-Type: multipart/form-data" \
  -F "email=test@example.com" \
  -F "file=@/path/to/audio.mp3" \
  -F "enable_diarization=true"

# 4. 查詢任務狀態（替換 TASK_ID）
curl "http://localhost:8100/api/tasks/TASK_ID"
```

---

## 📱 啟動前端（可選）

### 步驟 1：安裝 Node.js 依賴

```bash
cd frontend
npm install
```

### 步驟 2：啟動開發服務器

```bash
npm run dev
```

訪問：http://localhost:5173

### 使用前端

1. 輸入郵箱地址
2. 點擊「發送驗證碼」
3. 檢查郵箱，輸入 6 位驗證碼
4. 點擊「驗證」
5. 上傳音頻文件
6. 配置選項（語者分離、語言等）
7. 點擊「開始轉錄」
8. 等待處理完成，檢查郵箱接收結果

---

## 🔒 安全檢查清單

在開始使用前，請確認：

- ✅ `.env` 文件已配置且不在 Git 追蹤中
- ✅ `ADMIN_TOKEN` 至少 32 位隨機字符
- ✅ `EMAIL_HASH_SALT` 至少 32 位隨機字符
- ✅ SMTP 憑證正確（可發送郵件）
- ✅ 生產環境設置 `ENABLE_DOCS=false`
- ✅ 生產環境使用 HTTPS

---

## 📊 系統需求

### 最低配置

- **CPU**: 4 核心
- **RAM**: 8 GB
- **硬碟**: 50 GB
- **網路**: 穩定的網際網路連線

### 建議配置

- **CPU**: 8 核心以上
- **RAM**: 16 GB 以上
- **硬碟**: 100 GB SSD
- **GPU**: NVIDIA GPU（CUDA 支援，可選）

---

## 🐛 常見問題

### Q: 無法發送郵件？

**A**: 檢查：
1. SMTP 憑證是否正確
2. Gmail 是否啟用「兩步驟驗證」
3. 是否使用「應用程式密碼」而非帳號密碼
4. 防火牆是否阻擋 SMTP 端口 587

### Q: 模型下載太慢？

**A**:
1. 確認網路連線穩定
2. 設置 Hugging Face 鏡像站：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 首次啟動需要 10-30 分鐘下載模型

### Q: 出現 CUDA 錯誤？

**A**:
1. 確認是否有 NVIDIA GPU
2. 安裝 CUDA Toolkit 11.8+
3. 或使用 CPU 模式（系統會自動降級）

### Q: API 返回 429 Too Many Requests？

**A**: 您觸發了速率限制：
- 驗證碼：每小時最多 5 次
- 任務創建：每小時最多 10 個
- 等待一段時間後重試

### Q: 如何查看日誌？

**A**:
```bash
# 查看所有日誌
tail -f remote_server/logs/*.log

# 查看安全日誌
tail -f remote_server/logs/security.log

# 查看錯誤日誌
tail -f remote_server/logs/error.log
```

---

## 📚 下一步

- **完整文檔**：參考 [README.md](README.md)
- **安全設置**：參考 [SECURITY.md](SECURITY.md)
- **部署指南**：參考 [INSTALL.md](INSTALL.md)
- **架構說明**：參考 [CLAUDE.md](CLAUDE.md)
- **SSDLC 合規**：參考 [SSDLC-COMPLIANCE.md](SSDLC-COMPLIANCE.md)

---

## 💡 提示

1. **開發環境**：設置 `ENABLE_DOCS=true` 可訪問 API 文檔
2. **日誌監控**：定期檢查 `logs/security.log` 了解安全事件
3. **定期更新**：使用 `pip-audit` 檢查依賴漏洞
4. **備份重要資料**：定期備份 `.env` 和 `logs/` 目錄
5. **測試環境**：建議先在測試環境完整測試再部署到生產環境

---

**祝您使用愉快！**

如有問題，請參考其他文檔或提交 GitHub Issue。

---

**最後更新日期**：2025-01-10
**適用版本**：v2.1.0 (SSDLC Compliant)

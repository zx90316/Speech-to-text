# 安全文件 (SECURITY.md)

## 符合 SSDLC 安全要求

本專案已實施符合安全軟體開發生命週期 (SSDLC) 的各項安全措施。

---

## 📋 目錄

1. [安全功能概述](#安全功能概述)
2. [身份驗證與授權](#身份驗證與授權)
3. [輸入驗證與數據清理](#輸入驗證與數據清理)
4. [日誌與審計](#日誌與審計)
5. [加密與數據保護](#加密與數據保護)
6. [速率限制與防暴力破解](#速率限制與防暴力破解)
7. [安全標頭與CORS](#安全標頭與cors)
8. [錯誤處理](#錯誤處理)
9. [文件上傳安全](#文件上傳安全)
10. [個資保護](#個資保護)
11. [安全配置指南](#安全配置指南)
12. [安全最佳實踐](#安全最佳實踐)

---

## 🔒 安全功能概述

### 已實施的安全模組

1. **security_logger.py** - 安全日誌記錄系統
2. **input_validator.py** - 輸入驗證與數據清理
3. **rate_limiter.py** - 速率限制與防暴力破解
4. **crypto_utils.py** - 加密與密碼處理
5. **安全中介軟體** - HTTP 安全標頭、CORS、Trusted Host

---

## 🔐 身份驗證與授權

### 郵箱驗證機制

- **6 位數驗證碼**：隨機生成，5 分鐘有效期
- **驗證成功後**：延長至 24 小時有效期
- **防暴力破解**：
  - 15 分鐘內連續失敗 5 次，郵箱封禁 30 分鐘
  - IP 封禁 10 分鐘
  - 顯示剩餘嘗試次數

### 管理員 API

- **Token 驗證**：所有管理員 API 需要提供有效的 ADMIN_TOKEN
- **Token 要求**：至少 16 位，建議 32 位以上隨機字符串
- **環境變數存儲**：Token 存儲在 .env 文件中，不硬編碼

### 代碼位置

- 郵箱驗證：`remote_server/email_service.py`
- 管理員驗證：`remote_server/api.py` → `verify_admin_token()`

---

## ✅ 輸入驗證與數據清理

### 所有輸入都經過嚴格驗證

1. **郵箱地址**：
   - 格式驗證（RFC 5322 標準）
   - 長度限制（最多 255 字符）
   - 危險字符檢測

2. **驗證碼**：
   - 必須是 6 位數字
   - 長度和格式驗證

3. **文件名**：
   - 路徑遍歷攻擊防護（`../`、`\`）
   - Null 字節注入防護
   - 長度限制（255 字符）
   - 文件擴展名白名單

4. **文件上傳**：
   - 文件大小限制（500MB）
   - 魔術數字檢測（文件頭驗證）
   - MIME 類型檢測
   - 支持的格式：MP3, WAV, M4A, FLAC, OGG, OPUS

5. **任務參數**：
   - 時間範圍驗證（0-86400 秒）
   - 語言代碼白名單
   - 模型名稱驗證（防止路徑遍歷）
   - VAD 參數範圍檢查（0-1）
   - 語者數量範圍檢查（1-20）

### 代碼位置

- `remote_server/input_validator.py` - 所有驗證邏輯

---

## 📝 日誌與審計

### 多層級日誌系統

本專案實施符合 SSDLC 要求的完整日誌記錄：

#### 1. 身份驗證日誌 (`logs/auth.log`)

- **保存期限**：180 天（6 個月）
- **記錄內容**：
  - 郵箱驗證嘗試（成功/失敗）
  - 驗證碼發送記錄
  - 會話過期事件
  - 包含：時間戳、用戶郵箱、IP 地址、操作結果

#### 2. 操作日誌 (`logs/operation.log`)

- **保存期限**：180 天（6 個月）
- **記錄內容**：
  - 任務創建
  - 任務完成
  - 文件上傳
  - 文件刪除

#### 3. 安全事件日誌 (`logs/security.log`)

- **保存期限**：365 天（1 年）
- **記錄內容**：
  - 速率限制超出
  - 無效請求
  - 未授權訪問嘗試
  - 黑名單 IP/郵箱訪問

#### 4. 錯誤日誌 (`logs/error.log`)

- **保存期限**：180 天（6 個月）
- **記錄內容**：
  - 系統錯誤
  - 處理異常
  - 郵件發送失敗

#### 5. 審計日誌 (`logs/audit.log`)

- **保存期限**：1825 天（5 年，符合個資法要求）
- **記錄內容**：
  - 個人資料訪問
  - 數據刪除操作
  - 管理員操作

### 日誌格式

```
2025-01-10 14:30:45 | INFO | {"event_type":"AUTH_ATTEMPT","timestamp":"2025-01-10T14:30:45","user_id":"user@example.com","ip_address":"192.168.1.100","action":"email_verification","result":"success","details":{}}
```

### 日誌包含的關鍵資訊（符合 SSDLC 2.6 要求）

- ✅ 事件類型 (event_type)
- ✅ 發生時間 (timestamp)
- ✅ 發生位置 (ip_address)
- ✅ 使用者身分識別 (user_id)
- ✅ 操作結果 (result)
- ✅ 詳細資訊 (details)

### 代碼位置

- `remote_server/security_logger.py` - 日誌記錄系統

---

## 🔐 加密與數據保護

### 1. 密碼處理

- **算法**：PBKDF2-SHA256
- **迭代次數**：100,000 次
- **鹽值長度**：32 字節
- **常數時間比較**：防止時序攻擊

### 2. 數據加密

- **對稱加密**：Fernet (AES-128-CBC + HMAC-SHA256)
- **加密金鑰**：從環境變數讀取，自動生成
- **用途**：敏感數據加密存儲和傳輸

### 3. 郵箱地址保護

- **顯示遮罩**：`ab***@domain.com`
- **日誌遮罩**：管理員查看時自動遮罩
- **哈希存儲**：SHA-256 + 鹽值

### 4. 文件安全刪除

- **覆寫刪除**：刪除前覆寫 3 次隨機數據
- **自動清理**：任務完成或失敗後自動刪除臨時文件

### 代碼位置

- `remote_server/crypto_utils.py` - 加密工具

---

## 🚦 速率限制與防暴力破解

### API 端點速率限制

| 端點 | 限制 | 時間窗口 |
|------|------|----------|
| `/api/email/send-verification` | 5 次 | 1 小時 |
| `/api/email/verify-code` | 10 次 | 1 分鐘 |
| `/api/tasks` (創建任務) | 10 個 | 1 小時 |
| 管理員 API | 20 次 | 1 分鐘 |
| 一般端點 | 100 次 | 1 分鐘 |

### 黑名單機制

- **IP 黑名單**：超過限制後臨時封禁 10 分鐘
- **郵箱黑名單**：驗證失敗 5 次後封禁 30 分鐘
- **自動解封**：時間到期自動移除

### 防暴力破解措施

1. 顯示剩餘嘗試次數
2. 指數退避（連續失敗增加封禁時間）
3. 記錄所有異常訪問

### 代碼位置

- `remote_server/rate_limiter.py` - 速率限制器
- `remote_server/api.py` - `@limiter.limit()` 裝飾器

---

## 🛡️ 安全標頭與CORS

### HTTP 安全標頭

所有 API 響應都包含以下安全標頭：

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### CORS 配置

- **白名單模式**：只允許配置的源
- **默認允許源**：`http://localhost:5173`, `http://localhost:3000`
- **可配置**：通過環境變數 `ALLOWED_ORIGINS` 設置
- **限制方法**：GET, POST, DELETE
- **限制標頭**：Content-Type, Authorization

### Trusted Host

- **防止 Host Header 攻擊**
- **默認信任**：localhost, 127.0.0.1
- **可配置**：通過環境變數 `TRUSTED_HOSTS` 設置

### 代碼位置

- `remote_server/api.py` → `add_security_headers()` 中介軟體

---

## ❌ 錯誤處理

### 安全的錯誤響應

- **不洩露內部信息**：錯誤消息不包含堆棧追蹤、文件路徑
- **統一格式**：所有錯誤使用標準 HTTP 狀態碼
- **詳細日誌**：內部記錄詳細錯誤，外部只顯示安全消息

### 常見錯誤碼

- `400 Bad Request`：輸入驗證失敗
- `403 Forbidden`：未授權或已被封禁
- `404 Not Found`：資源不存在
- `429 Too Many Requests`：速率限制超出
- `500 Internal Server Error`：服務器內部錯誤

---

## 📁 文件上傳安全

### 1. 文件驗證

- ✅ 文件大小限制（500MB）
- ✅ 文件類型白名單（MP3, WAV, M4A, FLAC, OGG, OPUS）
- ✅ 魔術數字檢測（文件頭驗證）
- ✅ 文件名安全檢查（防止路徑遍歷）

### 2. 存儲安全

- ✅ 隔離存儲：每個任務獨立目錄 (`uploads/{task_id}/`)
- ✅ 臨時存儲：處理完成後自動刪除
- ✅ 權限控制：文件不可直接訪問

### 3. 處理安全

- ✅ 異步處理：不阻塞主線程
- ✅ 資源限制：防止 DoS 攻擊
- ✅ 錯誤隔離：單個任務失敗不影響其他任務

### 代碼位置

- `remote_server/input_validator.py` → `validate_upload_file()`
- `remote_server/memory_storage.py` → 文件管理

---

## 🔒 個資保護

### 符合個資法要求

#### 1. 數據收集

- **最小化原則**：只收集必要的個人資料（郵箱地址）
- **明確告知**：API 文檔說明數據用途

#### 2. 數據存儲

- **記憶體存儲**：任務數據存儲在記憶體中
- **臨時文件**：處理文件存儲在臨時目錄
- **加密保護**：敏感數據使用加密存儲

#### 3. 數據保存期限

- **任務數據**：服務器重啟後清除（記憶體模式）
- **日誌數據**：
  - 一般日誌：6 個月
  - 個資相關審計日誌：5 年

#### 4. 數據刪除

- **自動刪除**：任務完成/失敗後自動刪除臨時文件
- **安全刪除**：覆寫 3 次後刪除
- **刪除日誌**：記錄所有刪除操作

#### 5. 郵箱地址保護

- **遮罩顯示**：管理員查看時自動遮罩（`ab***@domain.com`）
- **日誌保護**：日誌中的郵箱地址適當遮罩
- **傳輸加密**：使用 HTTPS/TLS 傳輸

### 代碼位置

- `remote_server/memory_storage.py` → 數據管理
- `remote_server/crypto_utils.py` → `mask_email()`
- `remote_server/security_logger.py` → `log_data_deletion()`

---

## ⚙️ 安全配置指南

### 1. 環境變數設置

複製 `.env.example` 到 `.env` 並設置以下變數：

```bash
# 必須設置
ADMIN_TOKEN=<至少32位隨機字符串>
EMAIL_HASH_SALT=<至少32位隨機字符串>
ENCRYPTION_KEY=<程序自動生成>

# 建議設置
ALLOWED_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com
ENABLE_DOCS=false  # 生產環境

# SMTP 配置
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
```

### 2. 生成安全憑證

```python
# 生成 ADMIN_TOKEN
import secrets
print(secrets.token_hex(32))  # 64 字符

# 生成 EMAIL_HASH_SALT
print(secrets.token_hex(32))  # 64 字符
```

### 3. 生產環境部署

#### 使用 HTTPS

```bash
# 使用 Nginx 作為反向代理，配置 SSL/TLS
# 或使用 Uvicorn 的 SSL 選項
uvicorn api:app --host 0.0.0.0 --port 8100 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

#### 關閉 API 文檔

```bash
# .env
ENABLE_DOCS=false
```

#### 限制允許的源

```bash
# .env
ALLOWED_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com,www.yourdomain.com
```

---

## 🔐 安全最佳實踐

### 開發階段

- ✅ 不在代碼中硬編碼密鑰、密碼
- ✅ 使用環境變數管理敏感配置
- ✅ `.env` 文件加入 `.gitignore`
- ✅ 定期更新依賴套件
- ✅ 進行代碼安全審查

### 測試階段

- ✅ 進行輸入驗證測試
- ✅ 測試速率限制機制
- ✅ 測試錯誤處理
- ✅ 進行滲透測試（建議）
- ✅ 進行弱點掃描（建議）

### 部署階段

- ✅ 使用 HTTPS/TLS 1.2+
- ✅ 關閉 API 文檔（生產環境）
- ✅ 設置防火牆規則
- ✅ 配置適當的 CORS 白名單
- ✅ 定期備份日誌

### 維運階段

- ✅ 定期檢查日誌
- ✅ 監控異常訪問
- ✅ 定期更新憑證
- ✅ 定期清理舊日誌（符合保存期限）
- ✅ 定期審查安全配置

---

## 📞 安全問題回報

如發現安全漏洞，請勿公開揭露。請通過以下方式聯繫：

- 郵箱：[請填寫安全聯繫郵箱]
- 回報時請包含：
  - 漏洞描述
  - 重現步驟
  - 影響範圍
  - 建議的修復方案（可選）

---

## 📚 參考文檔

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [個人資料保護法](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=I0050021)

---

## 📋 SSDLC 檢核對照表

請參考 `WI-GA215-附件一、SSDLC檢核表.xlsx` 了解本專案如何滿足各項 SSDLC 要求。

---

**最後更新日期**：2025-01-10

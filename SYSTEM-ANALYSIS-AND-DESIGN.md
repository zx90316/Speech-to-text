# 系統分析及設計文件

Speech-to-Text API 服務 - SSDLC 合規版本

---

## 📋 文件資訊

- **專案名稱**：Speech-to-Text API Service
- **版本**：v2.1.0 (SSDLC Compliant)
- **文件版本**：1.0
- **最後更新**：2025-01-10
- **文件作者**：開發團隊
- **審核狀態**：待審核

---

## 目錄

1. [系統概述](#系統概述)
2. [需求分析](#需求分析)
3. [系統架構設計](#系統架構設計)
4. [安全設計](#安全設計)
5. [資料流程設計](#資料流程設計)
6. [組件設計](#組件設計)
7. [介面設計](#介面設計)
8. [部署架構](#部署架構)
9. [風險分析](#風險分析)
10. [設計追溯性](#設計追溯性)

---

## 1. 系統概述

### 1.1 系統目的

Speech-to-Text API 服務是一個基於 AI 的語音轉文字服務，提供：

- **語音轉錄**：使用 Whisper 模型將音頻文件轉換為文字
- **語者分離**：使用 Pyannote 模型識別不同說話者
- **信心分數分析**：提供詞級信心分數視覺化
- **郵件驗證**：基於郵件的身份驗證機制
- **即時進度追蹤**：使用 SSE 提供即時處理進度
- **結果郵件傳送**：處理完成後自動發送結果到使用者郵箱

### 1.2 系統特性

- **記憶體儲存**：無需資料庫，所有任務資料存於記憶體（OrderedDict）
- **自動清理**：任務完成或失敗後自動刪除暫存檔案
- **安全優先**：符合 45 項 SSDLC 安全要求（91.1% 合規率）
- **非同步處理**：支援多任務排隊，逐一處理
- **跨平台**：支援 Windows、Linux、macOS
- **GPU 加速**：自動檢測 CUDA，無 GPU 則降級到 CPU

### 1.3 技術堆疊

**後端**：
- Python 3.9-3.11
- FastAPI (Web 框架)
- Faster-Whisper (語音轉錄)
- Pyannote (語者分離)
- FFmpeg (音頻處理)
- OpenCC (繁體中文轉換)

**前端**：
- React 18
- TypeScript
- Vite
- Axios
- EventSource (SSE)

**安全**：
- cryptography (加密庫)
- PBKDF2-SHA256 (密碼雜湊)
- Fernet (資料加密)
- TLS 1.2+ (傳輸加密)

---

## 2. 需求分析

### 2.1 功能需求

#### 2.1.1 郵件驗證功能

**FR-001**：系統應提供郵件驗證機制
- 發送 6 位數驗證碼到使用者郵箱
- 驗證碼有效期 5 分鐘
- 驗證成功後有效期延長至 24 小時
- 支援重新發送驗證碼

**FR-002**：系統應限制驗證碼發送頻率
- 每小時最多發送 5 次驗證碼（每個郵箱）
- 5 次失敗驗證後禁用郵箱 30 分鐘
- IP 地址 10 分鐘內失敗 10 次後禁用 10 分鐘

#### 2.1.2 語音轉錄功能

**FR-003**：系統應支援多種音頻格式
- 支援格式：MP3, WAV, M4A, FLAC
- 檔案大小限制：500MB
- 自動格式轉換（使用 FFmpeg）

**FR-004**：系統應提供多種轉錄選項
- 多種 Whisper 模型選擇
- 語言選擇（自動檢測或指定）
- 任務類型（轉錄 / 翻譯）
- 計算類型（float32, int8, float16）
- 時間範圍選擇（轉錄音頻片段）

**FR-005**：系統應支援語者分離
- 可選啟用 Pyannote 語者分離
- 可配置最小/最大說話者數量
- VAD 敏感度調整

**FR-006**：系統應提供信心分數分析
- 可選啟用詞級信心分數
- 生成互動式 HTML 視覺化報告
- 顏色標示低信心詞彙

#### 2.1.3 任務管理功能

**FR-007**：系統應提供任務狀態查詢
- 查詢單一任務狀態
- 批次查詢多個任務狀態
- 查詢使用者所有任務（依郵箱）

**FR-008**：系統應提供即時進度追蹤
- 使用 SSE 串流即時進度
- 顯示當前處理階段
- 顯示部分轉錄結果（進行中）

**FR-009**：系統應支援任務取消
- 取消排隊中的任務
- 取消處理中的任務（軟取消）
- 永久刪除任務資料

#### 2.1.4 結果傳送功能

**FR-010**：系統應透過郵件傳送結果
- 任務完成後自動發送郵件
- 附件包含轉錄文字檔（transcript.txt）
- 可選附件：信心分數報告（confidence_report.html）
- 郵件正文包含前 500 字預覽

### 2.2 非功能需求

#### 2.2.1 安全需求（SSDLC）

**NFR-001**：身份驗證
- 郵件 + 驗證碼雙因子驗證
- 24 小時 session 有效期
- 管理員 API 需 Token 驗證

**NFR-002**：存取控制
- 使用者只能查詢自己的任務
- 管理員可查詢所有任務（郵箱遮罩）
- 速率限制防止暴力攻擊

**NFR-003**：資料保護
- 暫存檔案自動刪除
- 郵箱雜湊儲存（SHA-256 + salt）
- 敏感資料加密（Fernet）
- 安全檔案刪除（3 次覆寫）

**NFR-004**：日誌稽核
- 5 種日誌類型（auth, operation, security, error, audit）
- 日誌保留期限：6 個月 - 5 年
- 記錄所有安全事件和個資存取

**NFR-005**：輸入驗證
- 郵箱格式驗證（RFC 5322）
- 檔案類型驗證（magic number）
- 路徑遍歷防護
- SQL/XSS 注入防護

**NFR-006**：傳輸安全
- 生產環境強制 HTTPS
- TLS 1.2 或更高版本
- 安全 HTTP 標頭（HSTS, CSP, X-Frame-Options 等）

#### 2.2.2 效能需求

**NFR-007**：回應時間
- API 健康檢查：< 100ms
- 郵件驗證：< 2s
- 任務提交：< 3s
- 任務狀態查詢：< 500ms

**NFR-008**：處理能力
- 單一任務處理時間：依音頻長度（約 1:1 比例）
- 支援任務排隊（記憶體限制）
- 自動記憶體管理（CUDA cache 清理）

**NFR-009**：可用性
- 系統正常運行時間：> 99%
- 優雅降級（GPU → CPU）
- 錯誤處理和恢復機制

#### 2.2.3 可維護性需求

**NFR-010**：日誌和監控
- 完整的日誌記錄
- 日誌輪替和備份
- 系統健康檢查端點

**NFR-011**：文件化
- API 文件（Swagger/ReDoc）
- 部署指南
- 安全配置指南
- 維運手冊

**NFR-012**：可擴展性
- 模組化設計
- 配置驅動（環境變數）
- 支援 Docker 部署

---

## 3. 系統架構設計

### 3.1 整體架構

```
┌─────────────────────────────────────────────────────────────┐
│                        使用者層                              │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  Web 前端     │              │  API 客戶端   │            │
│  │  (React)     │              │  (curl/其他)  │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼──────────────────┘
          │                              │
          │         HTTPS/TLS 1.2+       │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Uvicorn + FastAPI 應用層（直接 HTTPS）          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  TLS 終止（Uvicorn 內建）                            │    │
│  │  - TLS 1.2 / TLS 1.3 支援                           │    │
│  │  - SSL 憑證驗證                                      │    │
│  │  - 可選 Port 轉發（443 → 8100）                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  安全中介軟體                                        │    │
│  │  ├─ 安全標頭（HSTS, CSP, X-Frame-Options）          │    │
│  │  ├─ CORS 白名單                                     │    │
│  │  ├─ Trusted Host 保護                               │    │
│  │  └─ 速率限制                                        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  API 端點                                            │    │
│  │  ├─ /api/email/* (郵件驗證)                         │    │
│  │  ├─ /api/tasks/* (任務管理)                         │    │
│  │  ├─ /api/my-tasks (使用者任務)                      │    │
│  │  ├─ /api/admin/* (管理員 API)                       │    │
│  │  ├─ /api/stats (統計資訊)                           │    │
│  │  └─ /health (健康檢查)                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心服務層                              │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  EmailService  │  │  TaskProcessor │  │MemoryStorage │ │
│  │  (郵件服務)     │  │  (任務處理器)   │  │(記憶體儲存)   │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │SecurityLogger  │  │InputValidator  │  │ RateLimiter  │ │
│  │  (安全日誌)     │  │  (輸入驗證)     │  │  (速率限制)   │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│                                                              │
│  ┌────────────────┐                                         │
│  │  CryptoUtils   │                                         │
│  │  (加密工具)     │                                         │
│  └────────────────┘                                         │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       AI 模型層                              │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │  Faster-Whisper      │    │  Pyannote            │      │
│  │  (語音轉錄)           │    │  (語者分離)           │      │
│  │  - 多模型支援          │    │  - 自動說話者檢測     │      │
│  │  - GPU/CPU 自適應     │    │  - VAD 配置          │      │
│  └──────────────────────┘    └──────────────────────┘      │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      儲存層                                  │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  記憶體儲存     │  │  暫存檔案系統   │  │  日誌檔案     │ │
│  │  (OrderedDict) │  │  uploads/      │  │  logs/       │ │
│  │  - 任務元資料   │  │  result/       │  │  - 5 種日誌   │ │
│  │  - 驗證狀態     │  │  (自動清理)     │  │  - 輪替備份   │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 架構特點

#### 3.2.1 分層架構（Layered Architecture）

- **展示層**：Web 前端（React）+ API 客戶端
- **應用層**：Uvicorn + FastAPI（TLS 終止、API 端點、中介軟體）
- **服務層**：核心業務邏輯（任務處理、郵件服務等）
- **AI 模型層**：Whisper、Pyannote 模型
- **儲存層**：記憶體儲存 + 暫存檔案系統 + 日誌

**架構優勢**：
- **簡化部署**：無需 Nginx，單一應用處理所有請求
- **降低延遲**：少一層代理，請求直達應用（延遲降低 ~20%）
- **統一配置**：TLS 和應用配置在同一處管理
- **適合 API**：本專案以 API 為主，無需複雜的靜態檔案服務

#### 3.2.2 記憶體優先設計（Memory-First Design）

- **無資料庫**：所有任務資料存於記憶體（OrderedDict）
- **優勢**：快速存取、簡化部署、無 SQL 注入風險
- **權衡**：重啟清空資料（設計如此，符合暫存性質）

#### 3.2.3 非同步處理（Asynchronous Processing）

- **任務佇列**：asyncio.Queue 管理待處理任務
- **背景處理**：獨立執行緒處理轉錄任務
- **即時反饋**：SSE 串流進度更新

#### 3.2.4 安全優先設計（Security-First Design）

- **縱深防禦**：多層安全機制（網路、應用、資料）
- **最小權限**：使用者只能存取自己的資料
- **稽核追蹤**：完整日誌記錄所有操作

---

## 4. 安全設計

### 4.1 安全架構

```
┌─────────────────────────────────────────────────────────────┐
│                   安全層級架構                               │
└─────────────────────────────────────────────────────────────┘

第 1 層：網路安全與傳輸加密
  ├─ TLS 1.2+ 加密傳輸（Uvicorn 內建）
  ├─ 防火牆（僅開放 443 或 8100）
  ├─ 可選 Port 轉發（443 → 8100，避免 root 權限）
  └─ DDoS 保護（速率限制 + Cloudflare 可選）

第 2 層：應用安全（FastAPI 中介軟體）
  ├─ 安全 HTTP 標頭（在應用層注入）
  │   ├─ Strict-Transport-Security: max-age=31536000
  │   ├─ X-Frame-Options: DENY
  │   ├─ X-Content-Type-Options: nosniff
  │   ├─ Content-Security-Policy: default-src 'self'
  │   ├─ X-XSS-Protection: 1; mode=block
  │   └─ Referrer-Policy: strict-origin-when-cross-origin
  ├─ CORS 白名單
  ├─ Trusted Host 保護
  └─ 速率限制（IP + 端點）

第 3 層：身份驗證與授權
  ├─ 郵件驗證（2FA）
  │   ├─ 6 位數驗證碼
  │   ├─ 5 分鐘有效期
  │   └─ 24 小時 session
  ├─ 管理員 Token 驗證
  └─ 基於郵箱的存取控制

第 4 層：輸入驗證
  ├─ 郵箱格式驗證（RFC 5322）
  ├─ 檔案驗證
  │   ├─ 大小限制（500MB）
  │   ├─ 類型白名單
  │   └─ Magic number 檢查
  ├─ 路徑遍歷防護
  └─ 參數範圍驗證

第 5 層：資料保護
  ├─ 傳輸加密（TLS）
  ├─ 靜態資料加密（Fernet）
  ├─ 密碼雜湊（PBKDF2-SHA256）
  ├─ 郵箱雜湊（SHA-256 + salt）
  ├─ 安全刪除（3 次覆寫）
  └─ 資料遮罩（日誌、管理介面）

第 6 層：日誌與監控
  ├─ 5 種日誌類型
  ├─ 保留期限（180 天 - 5 年）
  ├─ 安全事件記錄
  └─ 個資存取稽核
```

### 4.2 安全模組設計

#### 4.2.1 SecurityLogger（安全日誌模組）

**檔案**：[remote_server/security_logger.py](remote_server/security_logger.py)

**設計模式**：單例模式（Singleton）

**功能**：
- 管理 5 種日誌類型
- 自動日誌輪替
- 差異化保留期限
- 結構化日誌格式

**日誌類型**：

| 日誌類型 | 保留期限 | 用途 |
|---------|---------|------|
| auth.log | 180 天 | 驗證嘗試、session 管理 |
| operation.log | 180 天 | 任務創建、檔案上傳、任務完成 |
| security.log | 365 天 | 安全事件、速率限制、未授權存取 |
| error.log | 180 天 | 系統錯誤、處理失敗 |
| audit.log | 1825 天 (5年) | 個資存取、資料刪除（GDPR） |

**日誌格式**：
```
2025-01-10 12:34:56 | INFO | event_type=LOGIN | ip=192.168.1.1 | user_id=hash@example.com | action=verify_code | result=success | details={"attempts": 1}
```

**類別圖**：
```
┌──────────────────────────────┐
│     SecurityLogger           │
├──────────────────────────────┤
│ - _instance: SecurityLogger  │
│ - auth_logger: Logger        │
│ - operation_logger: Logger   │
│ - security_logger: Logger    │
│ - error_logger: Logger       │
│ - audit_logger: Logger       │
├──────────────────────────────┤
│ + log_auth()                 │
│ + log_operation()            │
│ + log_security()             │
│ + log_error()                │
│ + log_audit()                │
│ + _create_logger()           │
│ + _format_log_message()      │
└──────────────────────────────┘
```

#### 4.2.2 InputValidator（輸入驗證模組）

**檔案**：[remote_server/input_validator.py](remote_server/input_validator.py)

**設計模式**：靜態方法類（Static Methods Class）

**功能**：
- 郵箱驗證（RFC 5322）
- 檔案驗證（大小、類型、magic number）
- 檔名安全檢查（路徑遍歷、null byte）
- 參數範圍驗證
- 危險字元檢測

**驗證流程**：
```
輸入 → 格式驗證 → 範圍檢查 → 危險字元檢測 → 白名單驗證 → 通過/拒絕
```

**類別圖**：
```
┌──────────────────────────────────────┐
│      InputValidator                  │
├──────────────────────────────────────┤
│ + EMAIL_PATTERN: Pattern             │
│ + ALLOWED_FILE_TYPES: Set            │
│ + DANGEROUS_CHARS: List              │
│ + MAX_FILE_SIZE: int                 │
├──────────────────────────────────────┤
│ + validate_email()                   │
│ + validate_file()                    │
│ + validate_filename()                │
│ + validate_time_range()              │
│ + validate_language()                │
│ + validate_model_name()              │
│ + validate_vad_parameters()          │
│ + validate_task_id()                 │
│ + _check_file_magic_number()         │
└──────────────────────────────────────┘
```

#### 4.2.3 RateLimiter（速率限制模組）

**檔案**：[remote_server/rate_limiter.py](remote_server/rate_limiter.py)

**設計模式**：單例模式（Singleton）

**功能**：
- IP 速率限制
- 郵箱速率限制
- 端點級別限制
- 自動黑名單管理
- 暴力攻擊防護

**速率限制策略**：

| 端點類型 | 限制 | 時間窗口 | 超限後果 |
|---------|------|---------|---------|
| 一般 API | 100 請求 | 1 分鐘 | 429 錯誤 |
| 郵件驗證 | 5 次 | 1 小時 | 郵箱封禁 30 分鐘 |
| 任務創建 | 10 次 | 1 小時 | 429 錯誤 |
| 驗證失敗 | 5 次 | - | 郵箱封禁 30 分鐘 |
| IP 失敗 | 10 次 | 10 分鐘 | IP 封禁 10 分鐘 |

**類別圖**：
```
┌──────────────────────────────────────┐
│        RateLimiter                   │
├──────────────────────────────────────┤
│ - _instance: RateLimiter             │
│ - ip_requests: Dict                  │
│ - email_requests: Dict               │
│ - failed_attempts: Dict              │
│ - ip_blacklist: Dict                 │
│ - email_blacklist: Dict              │
│ - lock: RLock                        │
├──────────────────────────────────────┤
│ + check_ip_rate_limit()              │
│ + check_email_rate_limit()           │
│ + record_failed_attempt()            │
│ + is_ip_blacklisted()                │
│ + is_email_blacklisted()             │
│ + add_to_blacklist()                 │
│ + cleanup_expired()                  │
└──────────────────────────────────────┘
```

#### 4.2.4 CryptoUtils（加密工具模組）

**檔案**：[remote_server/crypto_utils.py](remote_server/crypto_utils.py)

**設計模式**：靜態方法類（Static Methods Class）

**功能**：
- 密碼雜湊（PBKDF2-SHA256）
- 資料加密/解密（Fernet）
- 郵箱雜湊（SHA-256 + salt）
- 安全檔案刪除（3 次覆寫）
- 資料遮罩（郵箱、IP）
- 常數時間比較（防時序攻擊）

**加密演算法**：

| 用途 | 演算法 | 參數 |
|------|--------|------|
| 密碼雜湊 | PBKDF2-SHA256 | 100,000 迭代 |
| 資料加密 | Fernet (AES-128-CBC + HMAC-SHA256) | 自動生成金鑰 |
| 郵箱雜湊 | SHA-256 | 環境變數 salt |
| 檔案刪除 | 多次覆寫 | 3 次（隨機、零、隨機） |

**類別圖**：
```
┌──────────────────────────────────────┐
│       CryptoUtils                    │
├──────────────────────────────────────┤
│ + hash_password()                    │
│ + verify_password()                  │
│ + encrypt_data()                     │
│ + decrypt_data()                     │
│ + hash_email()                       │
│ + secure_delete_file()               │
│ + mask_email()                       │
│ + mask_ip()                          │
│ + constant_time_compare()            │
│ + generate_encryption_key()          │
└──────────────────────────────────────┘
```

### 4.3 安全設計模式

#### 4.3.1 縱深防禦（Defense in Depth）

多層安全機制，即使一層被突破，其他層仍能保護：

```
攻擊者
  ↓
[第 1 層] 網路防火牆 → 阻擋
  ↓ (突破)
[第 2 層] TLS 加密（Uvicorn） → 阻擋 MITM
  ↓ (突破)
[第 3 層] 速率限制（應用層） → 阻擋 DDoS
  ↓ (突破)
[第 4 層] CORS/Trusted Host → 阻擋
  ↓ (突破)
[第 5 層] 身份驗證 → 阻擋
  ↓ (突破)
[第 6 層] 輸入驗證 → 阻擋注入攻擊
  ↓ (突破)
[第 7 層] 授權檢查 → 阻擋越權存取
  ↓ (突破)
[第 8 層] 日誌記錄 → 偵測與告警
```

#### 4.3.2 最小權限原則（Principle of Least Privilege）

- 使用者只能查詢自己提交的任務
- 管理員只能查看遮罩後的郵箱
- 任務檔案存於獨立目錄（task_id）
- 處理完成後自動刪除檔案

#### 4.3.3 預設安全（Secure by Default）

- API 文件預設關閉（生產環境）
- CORS 預設拒絕（需白名單）
- 所有端點預設需驗證
- 敏感資料預設加密

#### 4.3.4 失敗安全（Fail Securely）

- 錯誤訊息不洩露內部資訊
- 驗證失敗不區分「郵箱不存在」vs「密碼錯誤」
- 異常自動記錄但不中斷服務
- 資源不足時拒絕新任務（不會崩潰）

---

## 5. 資料流程設計

### 5.1 郵件驗證流程

```
┌──────┐                                              ┌──────┐
│ 使用者│                                              │ 系統  │
└───┬──┘                                              └───┬──┘
    │                                                     │
    │ 1. POST /api/email/send-verification               │
    │    email=user@example.com                          │
    ├────────────────────────────────────────────────────>│
    │                                                     │
    │                                    2. 驗證郵箱格式  │
    │                                       (InputValidator)
    │                                                     │
    │                                    3. 檢查速率限制  │
    │                                       (RateLimiter)
    │                                       - 5次/小時    │
    │                                                     │
    │                                    4. 生成驗證碼    │
    │                                       - 6位數字     │
    │                                       - 5分鐘有效   │
    │                                       (EmailService)
    │                                                     │
    │                                    5. 發送郵件      │
    │                                       (SMTP)        │
    │                                                     │
    │                                    6. 記錄日誌      │
    │                                       (auth.log)    │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 7. {"message": "驗證碼已發送"}                      │
    │                                                     │
    │ 8. 收到郵件（驗證碼：123456）                       │
    │                                                     │
    │ 9. POST /api/email/verify-code                     │
    │    email=user@example.com&code=123456              │
    ├────────────────────────────────────────────────────>│
    │                                                     │
    │                                   10. 驗證郵箱格式  │
    │                                       (InputValidator)
    │                                                     │
    │                                   11. 檢查黑名單    │
    │                                       (RateLimiter)
    │                                                     │
    │                                   12. 驗證驗證碼    │
    │                                       - 檢查有效期  │
    │                                       - 常數時間比較│
    │                                       (EmailService)
    │                                                     │
    │                                   13. 更新狀態      │
    │                                       - 24小時有效  │
    │                                       - 清除驗證碼  │
    │                                                     │
    │                                   14. 記錄日誌      │
    │                                       (auth.log)    │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 15. {"verified": true, "valid_until": "..."}       │
    │                                                     │
```

### 5.2 任務提交與處理流程

```
┌──────┐                                              ┌──────┐
│ 使用者│                                              │ 系統  │
└───┬──┘                                              └───┬──┘
    │                                                     │
    │ 1. POST /api/tasks                                 │
    │    - email=user@example.com                        │
    │    - file=audio.mp3                                │
    │    - enable_diarization=true                       │
    │    - language=zh                                   │
    ├────────────────────────────────────────────────────>│
    │                                                     │
    │                                    2. 驗證郵箱狀態  │
    │                                       - 檢查24小時有效
    │                                       (EmailService)
    │                                                     │
    │                                    3. 驗證輸入      │
    │                                       - 檔案大小    │
    │                                       - 檔案類型    │
    │                                       - 參數範圍    │
    │                                       (InputValidator)
    │                                                     │
    │                                    4. 檢查速率限制  │
    │                                       - 10次/小時   │
    │                                       (RateLimiter)
    │                                                     │
    │                                    5. 創建任務      │
    │                                       - 生成task_id │
    │                                       - 儲存元資料  │
    │                                       (MemoryStorage)
    │                                                     │
    │                                    6. 儲存檔案      │
    │                                       uploads/{task_id}/
    │                                                     │
    │                                    7. 加入佇列      │
    │                                       (asyncio.Queue)
    │                                                     │
    │                                    8. 記錄日誌      │
    │                                       (operation.log)
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 9. {"task_id": "xxx", "queue_position": 2}         │
    │                                                     │
    │ 10. GET /api/tasks/{task_id}/stream                │
    │     (建立 SSE 連線)                                 │
    ├────────────────────────────────────────────────────>│
    │                                                     │
    │                                   11. 背景處理開始  │
    │                                       (TaskProcessor)
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 12. data: {"progress": 5, "stage": "載入模型"}     │
    │                                                     │
    │                                   13. 載入 Whisper  │
    │                                       - 檢查CUDA    │
    │                                       - 載入模型    │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 14. data: {"progress": 25, "stage": "轉換音頻"}    │
    │                                                     │
    │                                   15. 音頻轉換      │
    │                                       - FFmpeg處理  │
    │                                       - 時間範圍裁切│
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 16. data: {"progress": 40, "stage": "轉錄中"}      │
    │                                                     │
    │                                   17. Whisper轉錄   │
    │                                       - 逐段處理    │
    │                                       - 信心分數    │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 18. data: {"progress": 50, "partial": "已轉錄文字"}│
    │                                                     │
    │                                   19. 語者分離      │
    │                                       - 載入Pyannote│
    │                                       - 檢測說話者  │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 20. data: {"progress": 80, "stage": "語者分離"}    │
    │                                                     │
    │                                   21. 整合結果      │
    │                                       - 合併轉錄    │
    │                                       - 信心分數HTML│
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 22. data: {"progress": 95, "stage": "發送郵件"}    │
    │                                                     │
    │                                   23. 發送結果郵件  │
    │                                       - transcript.txt
    │                                       - confidence.html
    │                                       (EmailService)
    │                                                     │
    │ 24. 收到結果郵件                                    │
    │                                                     │
    │                                   25. 清理檔案      │
    │                                       - 刪除上傳檔案│
    │                                       - 刪除結果檔案│
    │                                       - 安全刪除    │
    │                                                     │
    │                                   26. 記錄日誌      │
    │                                       (operation.log)
    │                                       (audit.log)   │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 27. data: {"progress": 100, "status": "completed"} │
    │                                                     │
```

### 5.3 管理員查詢流程

```
┌──────┐                                              ┌──────┐
│ 管理員│                                              │ 系統  │
└───┬──┘                                              └───┬──┘
    │                                                     │
    │ 1. GET /api/admin/tasks                            │
    │    Authorization: Bearer <ADMIN_TOKEN>             │
    ├────────────────────────────────────────────────────>│
    │                                                     │
    │                                    2. 驗證Token     │
    │                                       - 長度檢查    │
    │                                       - 環境變數比對│
    │                                       - 常數時間比較│
    │                                       (CryptoUtils) │
    │                                                     │
    │                                    3. 檢查速率限制  │
    │                                       - 20次/分鐘   │
    │                                       (RateLimiter)
    │                                                     │
    │                                    4. 查詢所有任務  │
    │                                       (MemoryStorage)
    │                                                     │
    │                                    5. 遮罩郵箱      │
    │                                       - ab***@domain
    │                                       (CryptoUtils) │
    │                                                     │
    │                                    6. 記錄稽核日誌  │
    │                                       - 管理員操作  │
    │                                       (audit.log)   │
    │                                                     │
    │ <────────────────────────────────────────────────── │
    │ 7. {"tasks": [{"email": "ab***@domain.com", ...}]} │
    │                                                     │
```

---

## 6. 組件設計

### 6.1 核心組件

#### 6.1.1 EmailService（郵件服務）

**檔案**：[remote_server/email_service.py](remote_server/email_service.py)

**職責**：
- 管理郵件驗證流程
- 發送驗證碼郵件
- 驗證驗證碼
- 發送結果郵件

**主要方法**：

```python
class EmailService:
    def __init__(self):
        """初始化 SMTP 配置"""

    def send_verification_code(self, email: str) -> bool:
        """發送驗證碼到指定郵箱"""
        # 1. 生成6位數驗證碼
        # 2. 儲存到記憶體（5分鐘有效）
        # 3. 透過SMTP發送郵件
        # 4. 記錄日誌

    def verify_code(self, email: str, code: str) -> bool:
        """驗證驗證碼"""
        # 1. 檢查驗證碼是否存在
        # 2. 檢查是否過期
        # 3. 常數時間比較
        # 4. 更新驗證狀態（24小時）
        # 5. 記錄日誌

    def is_email_verified(self, email: str) -> bool:
        """檢查郵箱是否已驗證且未過期"""

    def send_result_email(self, email: str, task_id: str,
                         transcript_path: str,
                         confidence_path: str = None):
        """發送轉錄結果到郵箱"""
        # 1. 讀取轉錄文字
        # 2. 準備郵件內容（含預覽）
        # 3. 附加檔案
        # 4. 發送郵件
        # 5. 記錄日誌
```

**狀態管理**：
```python
verification_codes = {
    "user@example.com": {
        "code": "123456",
        "expires_at": datetime(2025, 1, 10, 12, 40, 0),
        "attempts": 1
    }
}

verified_emails = {
    "user@example.com": {
        "verified_at": datetime(2025, 1, 10, 12, 35, 0),
        "valid_until": datetime(2025, 1, 11, 12, 35, 0)
    }
}
```

#### 6.1.2 MemoryStorage（記憶體儲存）

**檔案**：[remote_server/memory_storage.py](remote_server/memory_storage.py)

**職責**：
- 管理任務元資料
- 提供執行緒安全的存取
- 支援依郵箱查詢
- 支援批次操作

**資料結構**：

```python
tasks = OrderedDict({
    "task_id_123": {
        # 基本資訊
        "task_id": "task_id_123",
        "email": "hashed_email",
        "filename": "audio.mp3",
        "status": "processing",  # queued/processing/completed/failed/cancelled
        "progress": 45,
        "current_stage": "轉錄中",

        # 配置
        "enable_diarization": True,
        "start_time": 0,
        "end_time": None,
        "language": "zh",
        "task": "transcribe",
        "model": "CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32",
        "compute_type": "float32",
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "min_speakers": None,
        "max_speakers": None,
        "enable_confidence_score": True,

        # 時間戳記
        "created_at": "2025-01-10T12:00:00",
        "started_at": "2025-01-10T12:05:00",
        "completed_at": None,

        # 結果
        "transcript": [...],  # 部分結果
        "error": None,

        # 檔案路徑
        "upload_path": "uploads/task_id_123/audio.mp3",
        "result_path": "result/task_id_123/transcript.txt"
    }
})
```

**主要方法**：

```python
class MemoryStorage:
    def __init__(self):
        """初始化記憶體儲存"""
        self.tasks = OrderedDict()
        self.lock = RLock()

    def create_task(self, task_data: dict) -> str:
        """創建新任務，返回task_id"""

    def get_task(self, task_id: str) -> Optional[dict]:
        """查詢單一任務"""

    def update_task(self, task_id: str, updates: dict):
        """更新任務資料"""

    def delete_task(self, task_id: str):
        """刪除任務"""

    def get_tasks_by_email(self, email: str) -> List[dict]:
        """查詢使用者所有任務"""

    def get_all_tasks(self, mask_email: bool = False) -> List[dict]:
        """查詢所有任務（管理員）"""

    def get_queue_size(self) -> int:
        """取得排隊任務數量"""

    def cleanup_temp_files(self, task_id: str):
        """清理任務暫存檔案"""
```

#### 6.1.3 TaskProcessor（任務處理器）

**檔案**：[remote_server/task_processor.py](remote_server/task_processor.py)

**職責**：
- 管理 AI 模型（Whisper, Pyannote）
- 處理音頻轉換
- 執行轉錄和語者分離
- 生成信心分數報告
- 發送結果郵件

**處理階段**：

| 階段 | 進度 | 描述 |
|------|------|------|
| 初始化 | 0% | 任務開始 |
| 載入模型 | 0-5% | 載入 Whisper 模型 |
| 音頻轉換 | 20-25% | FFmpeg 格式轉換 |
| 語音辨識 | 30-60% | Whisper 轉錄 |
| 語者分離 | 70-85% | Pyannote 處理（可選） |
| 結果整合 | 85-95% | 合併結果、生成報告 |
| 郵件發送 | 95-100% | 發送結果郵件 |
| 清理 | 100% | 刪除暫存檔案 |

**主要方法**：

```python
class TaskProcessor:
    def __init__(self):
        """初始化處理器"""
        self.whisper_model = None
        self.diarization_pipeline = None
        self.current_model_name = None
        self.processing = False

    def process_task(self, task_id: str):
        """處理單一任務"""
        # 1. 載入模型
        # 2. 轉換音頻
        # 3. 執行轉錄
        # 4. 語者分離（可選）
        # 5. 生成報告
        # 6. 發送郵件
        # 7. 清理檔案

    def load_whisper_model(self, model_name: str, compute_type: str):
        """載入或切換 Whisper 模型"""

    def load_diarization_pipeline(self):
        """載入 Pyannote 語者分離模型"""

    def convert_audio(self, input_path: str, output_path: str,
                     start_time: float = None, end_time: float = None):
        """使用 FFmpeg 轉換音頻"""

    def transcribe_audio(self, audio_path: str, **kwargs) -> List[dict]:
        """使用 Whisper 轉錄音頻"""

    def diarize_audio(self, audio_path: str, **kwargs) -> List[dict]:
        """使用 Pyannote 進行語者分離"""

    def merge_transcription_diarization(self,
                                       transcription: List[dict],
                                       diarization: List[dict]) -> str:
        """合併轉錄和語者資訊"""

    def generate_confidence_report(self,
                                   transcription: List[dict]) -> str:
        """生成信心分數 HTML 報告"""

    def unload_models(self):
        """卸載模型釋放記憶體"""
```

**模型管理策略**：

```python
# 單例模式：確保只有一個模型實例
# 動態載入：根據任務需求載入不同模型
# 記憶體管理：處理完成後可選擇卸載模型

if self.current_model_name != model_name:
    # 卸載舊模型
    self.unload_models()
    # 載入新模型
    self.whisper_model = WhisperModel(model_name,
                                     device="cuda" if cuda.is_available() else "cpu",
                                     compute_type=compute_type)
    self.current_model_name = model_name
```

#### 6.1.4 API 端點（FastAPI）

**檔案**：[remote_server/api.py](remote_server/api.py)

**端點分類**：

**郵件驗證端點**：
- `POST /api/email/send-verification` - 發送驗證碼
- `POST /api/email/verify-code` - 驗證驗證碼

**任務管理端點**：
- `POST /api/tasks` - 創建任務
- `GET /api/tasks/{task_id}` - 查詢任務狀態
- `GET /api/tasks/{task_id}/stream` - SSE 進度串流
- `DELETE /api/tasks/{task_id}` - 取消/刪除任務
- `POST /api/tasks/batch` - 批次查詢任務
- `GET /api/my-tasks` - 查詢使用者任務

**統計端點**：
- `GET /api/stats` - 服務統計
- `GET /health` - 健康檢查

**管理員端點**：
- `GET /api/admin/tasks` - 查詢所有任務
- `GET /api/admin/stats` - 管理員統計

**中介軟體**：

```python
# 安全標頭中介軟體
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# CORS 中介軟體（白名單模式）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # 環境變數配置
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"]
)

# Trusted Host 中介軟體
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=TRUSTED_HOSTS  # 環境變數配置
)
```

### 6.2 前端組件

#### 6.2.1 EmailVerification（郵件驗證組件）

**檔案**：[frontend/src/components/EmailVerification.tsx](frontend/src/components/EmailVerification.tsx)

**功能**：
- 郵箱輸入與驗證
- 發送驗證碼
- 驗證碼輸入（6 位數）
- 倒數計時器
- 驗證狀態持久化（localStorage）

**狀態管理**：
```typescript
interface EmailVerificationProps {
  onVerified: (email: string) => void;
}

interface State {
  email: string;
  code: string;
  isCodeSent: boolean;
  isVerified: boolean;
  countdown: number;
  error: string | null;
  remainingAttempts: number;
}
```

**關鍵功能**：
```typescript
// 發送驗證碼
const handleSendCode = async () => {
  const response = await api.sendVerificationCode(email);
  // 啟動60秒倒數計時
  startCountdown(60);
};

// 驗證驗證碼
const handleVerifyCode = async () => {
  const response = await api.verifyCode(email, code);
  if (response.verified) {
    // 儲存到 localStorage（24小時）
    emailStorage.setVerifiedEmail(email, response.valid_until);
    onVerified(email);
  }
};

// 頁面載入時檢查 localStorage
useEffect(() => {
  const stored = emailStorage.getVerifiedEmail();
  if (stored && new Date(stored.valid_until) > new Date()) {
    setIsVerified(true);
    onVerified(stored.email);
  }
}, []);
```

#### 6.2.2 UploadSection（上傳區組件）

**檔案**：[frontend/src/components/UploadSection.tsx](frontend/src/components/UploadSection.tsx)

**功能**：
- 拖放上傳
- 檔案驗證（大小、類型）
- 模型選擇
- 進階參數配置
- 任務提交

**配置選項**：
```typescript
interface UploadConfig {
  // 基本選項
  enableDiarization: boolean;
  language: string;
  task: 'transcribe' | 'translate';
  model: string;
  computeType: 'float32' | 'int8' | 'float16';

  // 進階選項
  startTime?: number;
  endTime?: number;
  vadOnset: number;
  vadOffset: number;
  minSpeakers?: number;
  maxSpeakers?: number;
  enableConfidenceScore: boolean;
}
```

**檔案上傳**：
```typescript
const handleFileUpload = (file: File) => {
  // 1. 驗證檔案大小（500MB）
  if (file.size > 500 * 1024 * 1024) {
    setError('檔案太大，最大 500MB');
    return;
  }

  // 2. 驗證檔案類型
  const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/flac'];
  if (!allowedTypes.includes(file.type)) {
    setError('不支援的檔案格式');
    return;
  }

  setSelectedFile(file);
};

const handleSubmit = async () => {
  const response = await api.createTask(email, selectedFile, config);
  // 儲存 task_id 到 localStorage
  taskStorage.addTask(response.task_id);
  // 導航到進度頁面
  navigate(`/progress/${response.task_id}`);
};
```

#### 6.2.3 TaskProgress（進度顯示組件）

**檔案**：[frontend/src/components/TaskProgress.tsx](frontend/src/components/TaskProgress.tsx)

**功能**：
- 即時進度追蹤（SSE）
- 進度條顯示
- 階段資訊
- 部分結果預覽
- 任務取消

**SSE 連線**：
```typescript
useEffect(() => {
  const eventSource = new EventSource(
    `${API_BASE_URL}/api/tasks/${taskId}/stream`
  );

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    setProgress(data.progress);
    setStage(data.stage);
    if (data.partial) {
      setPartialResult(data.partial);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    // 自動重連（最多3次）
    if (reconnectAttempts < 3) {
      setTimeout(() => {
        setReconnectAttempts(prev => prev + 1);
      }, 2000);
    }
  };

  return () => eventSource.close();
}, [taskId, reconnectAttempts]);
```

**任務取消**：
```typescript
const handleCancel = async () => {
  await api.cancelTask(taskId, permanent);
  navigate('/');
};
```

#### 6.2.4 TaskHistory（任務歷史組件）

**檔案**：[frontend/src/components/TaskHistory.tsx](frontend/src/components/TaskHistory.tsx)

**功能**：
- 顯示使用者所有任務
- 批次狀態查詢
- 任務詳情查看
- 任務刪除

**資料載入**：
```typescript
useEffect(() => {
  const loadTasks = async () => {
    // 從 localStorage 載入 task_ids
    const taskIds = taskStorage.getTasks();

    // 批次查詢狀態
    const response = await api.batchQueryTasks(taskIds);
    setTasks(response.tasks);

    // 或使用郵箱查詢（如果已驗證）
    if (verifiedEmail) {
      const response = await api.getMyTasks(verifiedEmail);
      setTasks(response.tasks);
    }
  };

  loadTasks();

  // 每 10 秒自動重新整理
  const interval = setInterval(loadTasks, 10000);
  return () => clearInterval(interval);
}, [verifiedEmail]);
```

---

## 7. 介面設計

### 7.1 API 介面規範

#### 7.1.1 RESTful 設計原則

- **資源導向**：端點代表資源（tasks, email）
- **HTTP 方法語意**：GET（查詢）、POST（創建）、DELETE（刪除）
- **狀態碼標準**：200（成功）、400（錯誤請求）、401（未授權）、429（速率限制）、500（伺服器錯誤）
- **JSON 格式**：所有請求和回應使用 JSON

#### 7.1.2 郵件驗證 API

**發送驗證碼**：

```
POST /api/email/send-verification?email={email}

Response 200:
{
  "message": "驗證碼已發送到您的郵箱"
}

Response 429:
{
  "detail": "請求過於頻繁，請稍後再試",
  "remaining_time": 300
}
```

**驗證驗證碼**：

```
POST /api/email/verify-code?email={email}&code={code}

Response 200:
{
  "verified": true,
  "valid_until": "2025-01-11T12:35:00",
  "message": "郵箱驗證成功"
}

Response 400:
{
  "detail": "驗證碼錯誤或已過期",
  "remaining_attempts": 3
}
```

#### 7.1.3 任務管理 API

**創建任務**：

```
POST /api/tasks
Content-Type: multipart/form-data

Parameters:
- email: string (required)
- file: binary (required, max 500MB)
- enable_diarization: boolean (optional, default false)
- language: string (optional, default "zh")
- task: string (optional, "transcribe" or "translate")
- model: string (optional)
- compute_type: string (optional, "float32", "int8", "float16")
- start_time: float (optional)
- end_time: float (optional)
- vad_onset: float (optional)
- vad_offset: float (optional)
- min_speakers: int (optional)
- max_speakers: int (optional)
- enable_confidence_score: boolean (optional)

Response 200:
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "queue_position": 2,
  "message": "任務已加入佇列"
}

Response 400:
{
  "detail": "檔案大小超過限制"
}

Response 401:
{
  "detail": "郵箱未驗證或已過期"
}
```

**查詢任務狀態**：

```
GET /api/tasks/{task_id}

Response 200:
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "current_stage": "語音辨識中",
  "created_at": "2025-01-10T12:00:00",
  "started_at": "2025-01-10T12:05:00",
  "transcript": ["部分轉錄結果..."]
}
```

**SSE 進度串流**：

```
GET /api/tasks/{task_id}/stream

Response: text/event-stream

data: {"progress": 5, "stage": "載入模型"}

data: {"progress": 25, "stage": "音頻轉換"}

data: {"progress": 45, "stage": "語音辨識", "partial": "已轉錄的文字..."}

data: {"progress": 100, "status": "completed"}
```

**取消任務**：

```
DELETE /api/tasks/{task_id}?permanent=false

Response 200:
{
  "message": "任務已取消"
}
```

**批次查詢**：

```
POST /api/tasks/batch
Content-Type: application/json

{
  "task_ids": ["task_id_1", "task_id_2", "task_id_3"]
}

Response 200:
{
  "tasks": [
    {"task_id": "task_id_1", "status": "completed", ...},
    {"task_id": "task_id_2", "status": "processing", ...},
    {"task_id": "task_id_3", "status": "failed", ...}
  ]
}
```

**查詢使用者任務**：

```
GET /api/my-tasks?email={email}

Response 200:
{
  "tasks": [
    {"task_id": "...", "status": "completed", ...},
    {"task_id": "...", "status": "processing", ...}
  ]
}
```

#### 7.1.4 管理員 API

**查詢所有任務**：

```
GET /api/admin/tasks
Authorization: Bearer {ADMIN_TOKEN}

Response 200:
{
  "tasks": [
    {
      "task_id": "...",
      "email": "ab***@example.com",  // 遮罩處理
      "status": "completed",
      ...
    }
  ]
}

Response 401:
{
  "detail": "無效的管理員 Token"
}
```

**管理員統計**：

```
GET /api/admin/stats
Authorization: Bearer {ADMIN_TOKEN}

Response 200:
{
  "total_tasks": 150,
  "status_counts": {
    "queued": 2,
    "processing": 1,
    "completed": 120,
    "failed": 20,
    "cancelled": 7
  },
  "average_processing_time": 180.5
}
```

### 7.2 資料庫介面（記憶體儲存）

雖然不使用傳統資料庫，但 MemoryStorage 提供類似介面：

```python
# CRUD 操作
task_id = memory_storage.create_task(task_data)
task = memory_storage.get_task(task_id)
memory_storage.update_task(task_id, {"progress": 50})
memory_storage.delete_task(task_id)

# 查詢操作
tasks = memory_storage.get_tasks_by_email(email)
all_tasks = memory_storage.get_all_tasks(mask_email=True)
queue_size = memory_storage.get_queue_size()

# 清理操作
memory_storage.cleanup_temp_files(task_id)
```

### 7.3 模型介面

**Whisper 模型介面**：

```python
# 載入模型
model = WhisperModel(
    model_name="CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32",
    device="cuda",
    compute_type="float32"
)

# 轉錄音頻
segments, info = model.transcribe(
    audio_path,
    language="zh",
    task="transcribe",
    vad_filter=True,
    vad_parameters={"onset": 0.5, "offset": 0.363},
    word_timestamps=True  # 啟用詞級時間戳記（用於信心分數）
)

# 結果格式
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
    if segment.words:
        for word in segment.words:
            print(f"  {word.word} (confidence: {word.probability:.2f})")
```

**Pyannote 模型介面**：

```python
# 載入語者分離 pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# 執行語者分離
diarization = pipeline(
    audio_path,
    min_speakers=2,
    max_speakers=5
)

# 結果格式
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
```

---

## 8. 部署架構

### 8.1 開發環境架構

```
┌─────────────────────────────────────────┐
│         開發機器（Windows/Linux/Mac）     │
│                                         │
│  ┌────────────────────────────────┐    │
│  │  前端開發伺服器                  │    │
│  │  localhost:5173                │    │
│  │  (Vite)                        │    │
│  └───────────┬────────────────────┘    │
│              │ Proxy                   │
│              ▼                          │
│  ┌────────────────────────────────┐    │
│  │  後端 API 伺服器                │    │
│  │  localhost:8100                │    │
│  │  (Uvicorn + FastAPI)           │    │
│  │  - API 文件開啟                 │    │
│  │  - CORS 寬鬆設定                │    │
│  │  - 本地 SMTP 測試               │    │
│  └────────────────────────────────┘    │
│                                         │
│  ┌────────────────────────────────┐    │
│  │  Python 虛擬環境                │    │
│  │  .venv/                        │    │
│  │  - Faster-Whisper              │    │
│  │  - Pyannote                    │    │
│  │  - FastAPI                     │    │
│  └────────────────────────────────┘    │
│                                         │
│  ┌────────────────────────────────┐    │
│  │  記憶體儲存 + 暫存檔案           │    │
│  │  - uploads/                    │    │
│  │  - result/                     │    │
│  │  - logs/                       │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 8.2 生產環境架構（Uvicorn 直接 HTTPS）

```
                         網際網路
                            │
                            ▼
        ┌──────────────────────────────────┐
        │     Cloudflare CDN (可選)        │
        │     - DDoS 保護                   │
        │     - 免費 SSL 憑證                │
        │     - WAF 防護                    │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │     防火牆 + Port 轉發（可選）     │
        │     - 443 → 8100 (iptables)      │
        │     - 避免需要 root 權限          │
        └──────────┬───────────────────────┘
                   │
     ┌─────────────┴──────────────┐
     │                            │
     ▼                            ▼
┌─────────────┐            ┌─────────────┐
│  伺服器 1    │            │  伺服器 2    │
│             │            │  (水平擴展)  │
└─────────────┘            └─────────────┘
     │
     │
     ▼
┌───────────────────────────────────────────────┐
│              單一伺服器架構                     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │   Uvicorn + FastAPI 應用             │     │
│  │   (Systemd Service)                 │     │
│  │                                      │     │
│  │   - 直接處理 HTTPS (Port 8100)       │     │
│  │   - TLS 1.2+ 內建支援                │     │
│  │   - 單 Worker（記憶體儲存）           │     │
│  │   - 環境變數配置 (.env)              │     │
│  │   - 安全標頭中介軟體                  │     │
│  └──────────┬──────────────────────────┘     │
│             │                                 │
│             ▼                                 │
│  ┌─────────────────────────────────────┐     │
│  │      AI 模型層                        │     │
│  │      - Whisper (GPU/CPU)             │     │
│  │      - Pyannote                      │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      儲存層                           │     │
│  │      - 記憶體儲存（RAM）              │     │
│  │      - /opt/app/uploads (暫存)       │     │
│  │      - /opt/app/result (暫存)        │     │
│  │      - /var/log/app/ (日誌)          │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      監控與日誌                       │     │
│  │      - Logrotate (日誌輪替)          │     │
│  │      - Prometheus (可選)             │     │
│  │      - Grafana (可選)                │     │
│  └─────────────────────────────────────┘     │
└───────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│      外部服務                                  │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │   SMTP 郵件伺服器                    │     │
│  │   (Gmail / Outlook / SendGrid)      │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │   備份服務（可選）                    │     │
│  │   (AWS S3 / Azure Blob)             │     │
│  └─────────────────────────────────────┘     │
└───────────────────────────────────────────────┘
```

### 8.3 Docker 容器架構（簡化版 - 無 Nginx）

```
┌──────────────────────────────────────────────┐
│           Docker Compose 環境                 │
│                                              │
│  ┌────────────────────────────────────┐     │
│  │   api 容器（直接 HTTPS）            │     │
│  │   - Image: python:3.11-slim        │     │
│  │   - Port: 8100:8100 (HTTPS)        │     │
│  │   - Volumes:                       │     │
│  │     - ./remote_server:/app         │     │
│  │     - uploads:/app/uploads         │     │
│  │     - result:/app/result           │     │
│  │     - logs:/app/logs               │     │
│  │     - SSL 憑證掛載                  │     │
│  │   - Env: .env                      │     │
│  │     - USE_HTTPS=true               │     │
│  │     - SSL_KEYFILE=/certs/key.pem   │     │
│  │     - SSL_CERTFILE=/certs/cert.pem │     │
│  │   - Resources:                     │     │
│  │     - CPU: 4 cores                 │     │
│  │     - Memory: 8GB                  │     │
│  │     - GPU: optional (CUDA)         │     │
│  └────────────────────────────────────┘     │
│                                              │
│  ┌────────────────────────────────────┐     │
│  │   frontend 容器（可選）              │     │
│  │   - Image: nginx:alpine            │     │
│  │   - Port: 5173                     │     │
│  │   - Volumes: ./frontend/dist       │     │
│  │   - 或直接使用前端開發伺服器          │     │
│  └────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

**docker-compose.yml 範例（Uvicorn 直接 HTTPS）**：

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8100:8100"  # HTTPS 端口
    env_file:
      - remote_server/.env
    environment:
      - USE_HTTPS=true
      - SSL_KEYFILE=/certs/privkey.pem
      - SSL_CERTFILE=/certs/fullchain.pem
    volumes:
      - ./remote_server/logs:/app/logs
      - ./remote_server/uploads:/app/uploads
      - ./remote_server/result:/app/result
      # 掛載 SSL 憑證（Let's Encrypt 範例）
      - /etc/letsencrypt/live/yourdomain.com:/certs:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]  # GPU 支援（可選）
```

**使用自簽憑證的替代方案**：

```yaml
volumes:
  # Windows 範例
  - C:/nginx/ssl:/certs:ro

  # Linux 範例
  - /path/to/ssl:/certs:ro
```

### 8.4 Kubernetes 架構（大規模部署）

```
┌───────────────────────────────────────────────┐
│         Kubernetes Cluster                    │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      Ingress Controller              │     │
│  │      - cert-manager (自動SSL)        │     │
│  │      - nginx-ingress                 │     │
│  └──────────┬──────────────────────────┘     │
│             │                                 │
│             ▼                                 │
│  ┌─────────────────────────────────────┐     │
│  │      Service (LoadBalancer)          │     │
│  └──────────┬──────────────────────────┘     │
│             │                                 │
│             ▼                                 │
│  ┌─────────────────────────────────────┐     │
│  │      Deployment (API Pods)           │     │
│  │      - Replicas: 3                   │     │
│  │      - Resources:                    │     │
│  │        CPU: 2 cores / pod            │     │
│  │        Memory: 4GB / pod             │     │
│  │      - Health checks                 │     │
│  │      - Rolling updates               │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      ConfigMap                       │     │
│  │      - 環境變數配置                   │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      Secret                          │     │
│  │      - ADMIN_TOKEN                   │     │
│  │      - EMAIL_HASH_SALT               │     │
│  │      - SMTP credentials              │     │
│  └─────────────────────────────────────┘     │
│                                               │
│  ┌─────────────────────────────────────┐     │
│  │      PersistentVolume                │     │
│  │      - logs/                         │     │
│  └─────────────────────────────────────┘     │
└───────────────────────────────────────────────┘
```

### 8.5 網路架構與安全

**防火牆規則（Uvicorn 直接 HTTPS）**：

```
# 入站規則（Inbound）
允許 TCP 443 (HTTPS) - 從任何來源 → Uvicorn (Port 8100，透過 Port 轉發)
允許 TCP 8100 (HTTPS) - 從任何來源 → Uvicorn（或使用 Port 轉發）
拒絕所有其他端口

# 出站規則（Outbound）
允許 TCP 587 (SMTP TLS) - 到郵件伺服器
允許 TCP 443 (HTTPS) - 到 Hugging Face（下載模型）
允許 TCP 80 (HTTP) - 到套件源

# Port 轉發設置（Linux iptables）
sudo iptables -t nat -A PREROUTING -p tcp --dport 443 -j REDIRECT --to-port 8100

# Port 轉發設置（Windows netsh）
netsh interface portproxy add v4tov4 listenport=443 connectport=8100 connectaddress=127.0.0.1
```

**網路架構（簡化版 - 無 DMZ）**：

```
┌────────────────────────────────────┐
│     公開網路層                      │
│     - Uvicorn + FastAPI            │
│     - 直接處理 HTTPS（Port 8100）   │
│     - 防火牆保護                    │
│     - 速率限制（應用層）             │
└────────────────────────────────────┘

優勢：
✅ 簡化架構，減少攻擊面
✅ 降低延遲（無代理層）
✅ 統一配置管理
✅ 適合 API 為主的服務
```

**與 Cloudflare 整合（推薦）**：

```
網際網路
    ↓
Cloudflare（免費 SSL、DDoS 保護、WAF）
    ↓
您的伺服器（Uvicorn Port 8100）

配置：
1. DNS 設置 A 記錄指向伺服器 IP
2. 啟用 Cloudflare Proxy（橙色雲朵）
3. SSL/TLS 模式：Full (Strict)
4. Uvicorn 使用 Let's Encrypt 或自簽憑證
```

---

## 9. 風險分析

### 9.1 安全風險

| 風險 ID | 風險描述 | 影響 | 機率 | 風險等級 | 緩解措施 | 狀態 |
|--------|---------|------|------|---------|---------|------|
| SEC-001 | 暴力破解郵箱驗證 | 高 | 中 | 高 | 速率限制（5次/小時）、自動封禁、驗證碼過期（5分鐘） | 已緩解 |
| SEC-002 | DDoS 攻擊 | 高 | 高 | 高 | 應用層速率限制（RateLimiter）、Cloudflare（推薦）、IP 黑名單 | 已緩解 |
| SEC-003 | 檔案上傳漏洞 | 高 | 中 | 高 | 檔案類型白名單、magic number 檢查、大小限制（500MB） | 已緩解 |
| SEC-004 | 路徑遍歷攻擊 | 高 | 低 | 中 | 檔名驗證、路徑規範化、危險字元檢測 | 已緩解 |
| SEC-005 | MITM（中間人攻擊） | 高 | 中 | 高 | 強制 HTTPS/TLS 1.2+、HSTS 標頭 | 已緩解 |
| SEC-006 | XSS 攻擊 | 中 | 低 | 中 | CSP 標頭、輸入驗證、React 自動轉義 | 已緩解 |
| SEC-007 | CSRF 攻擊 | 中 | 低 | 中 | SameSite cookies、CORS 白名單 | 已緩解 |
| SEC-008 | 管理員 Token 洩露 | 高 | 低 | 中 | 環境變數儲存、最小權限、稽核日誌 | 已緩解 |
| SEC-009 | 敏感資料洩露（日誌） | 中 | 中 | 中 | 郵箱遮罩、日誌存取控制、加密儲存 | 已緩解 |
| SEC-010 | 依賴套件漏洞 | 中 | 中 | 中 | 定期 pip-audit、安全更新、版本鎖定 | 持續監控 |

### 9.2 可用性風險

| 風險 ID | 風險描述 | 影響 | 機率 | 風險等級 | 緩解措施 | 狀態 |
|--------|---------|------|------|---------|---------|------|
| AVL-001 | 模型下載失敗 | 高 | 中 | 高 | 錯誤重試、鏡像站、離線模型 | 部分緩解 |
| AVL-002 | CUDA 記憶體不足 | 中 | 中 | 中 | 自動降級 CPU、模型卸載、記憶體監控 | 已緩解 |
| AVL-003 | 任務佇列阻塞 | 中 | 中 | 中 | 超時機制、任務取消、佇列大小限制 | 已緩解 |
| AVL-004 | SMTP 服務中斷 | 高 | 低 | 中 | 重試機制、錯誤日誌、備用 SMTP | 部分緩解 |
| AVL-005 | 磁碟空間不足 | 高 | 中 | 高 | 自動清理、定期監控、告警機制 | 已緩解 |
| AVL-006 | 伺服器重啟資料丟失 | 中 | 低 | 低 | 設計如此（記憶體儲存）、使用者郵件通知 | 接受風險 |

### 9.3 效能風險

| 風險 ID | 風險描述 | 影響 | 機率 | 風險等級 | 緩解措施 | 狀態 |
|--------|---------|------|------|---------|---------|------|
| PERF-001 | 長音頻處理時間過長 | 中 | 高 | 中 | 非同步處理、進度反饋、時間範圍選擇 | 已緩解 |
| PERF-002 | 並發請求過多 | 中 | 中 | 中 | 速率限制、佇列機制、負載均衡（可擴展） | 已緩解 |
| PERF-003 | 記憶體洩漏 | 高 | 低 | 中 | 定期模型卸載、CUDA cache 清理、監控 | 持續監控 |
| PERF-004 | SSE 連線過多 | 中 | 中 | 中 | 連線超時、自動重連限制、連線池管理 | 部分緩解 |

### 9.4 合規風險

| 風險 ID | 風險描述 | 影響 | 機率 | 風險等級 | 緩解措施 | 狀態 |
|--------|---------|------|------|---------|---------|------|
| COMP-001 | GDPR 個資保護不足 | 高 | 低 | 中 | 5年稽核日誌、資料刪除、最小化收集 | 已緩解 |
| COMP-002 | SSDLC 稽核失敗 | 中 | 低 | 低 | 91.1% 合規率、完整文件、持續改進 | 已緩解 |
| COMP-003 | 日誌保留期限不符 | 中 | 低 | 低 | 自動輪替、差異化保留、備份機制 | 已緩解 |

### 9.5 風險矩陣

```
影響 ▲
高 │ SEC-001  AVL-001               SEC-002
    │ SEC-003  AVL-005               SEC-005
    │ SEC-008                        COMP-001
    │
中 │ SEC-004  AVL-002  PERF-001     SEC-006
    │ SEC-006  AVL-003  PERF-002     SEC-007
    │ SEC-009  AVL-004               SEC-010
    │ SEC-010  AVL-006
    │ PERF-003
    │
低 │          COMP-002              COMP-003
    │          COMP-003
    │
    └─────────────────────────────────────────> 機率
              低         中         高
```

**風險等級定義**：
- **高**（紅色）：需立即處理
- **中**（黃色）：需密切關注
- **低**（綠色）：可接受風險

---

## 10. 設計追溯性

### 10.1 SSDLC 需求追溯

本節說明系統設計如何滿足 45 項 SSDLC 檢查清單需求：

#### 需求分析及規劃階段（6 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 1.1 | 系統防護基準評估 | 威脅模型分析、風險評估 | 本文件第 9 節 |
| 1.2 | 存取控制 | 郵箱驗證、Token 驗證、速率限制 | security_logger.py, rate_limiter.py |
| 1.3 | 帳號控管 | 郵箱唯一性、24小時 session | email_service.py |
| 1.4 | 可歸責性 | 完整稽核日誌（audit.log） | security_logger.py |
| 1.5 | 多因子認證 | 郵箱 + 驗證碼 2FA | email_service.py |
| 1.6 | 機敏資料保護 | 加密、遮罩、安全刪除 | crypto_utils.py |

#### 架構設計階段（12 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 2.1 | 資源充足性 | 記憶體管理、CUDA cache 清理 | task_processor.py |
| 2.2 | 帳號唯一性 | 郵箱作為唯一識別 | email_service.py |
| 2.3 | 密碼強度 | 驗證碼強度（6位數、5分鐘過期） | email_service.py |
| 2.4 | 連線時間控制 | Session 24小時過期 | email_service.py |
| 2.5 | 操作日誌 | 5 種日誌類型 | security_logger.py |
| 2.6 | 日誌保存期限 | 180天-5年差異化保留 | security_logger.py |
| 2.7 | 日誌內容完整性 | 事件、時間、IP、使用者、結果 | security_logger.py |
| 2.8 | 日誌資源充足 | 自動輪替、壓縮 | security_logger.py |
| 2.9 | 日誌監控 | 結構化日誌、可整合監控系統 | security_logger.py |
| 2.10 | 資料刪除機制 | 任務完成自動刪除、安全刪除（3次覆寫） | crypto_utils.py, task_processor.py |
| 2.11 | 加密機制 | PBKDF2-SHA256, Fernet, TLS 1.2+ | crypto_utils.py, api.py |
| 2.12 | 單一存取路徑 | Nginx 反向代理 | INSTALL.md |

#### 開發與測試階段（9 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 3.1 | 安全需求文件化 | SECURITY.md, SSDLC-COMPLIANCE.md | 專案根目錄 |
| 3.2 | 環境區隔 | .env 配置、ENABLE_DOCS 開關 | .env.example |
| 3.3 | 源碼檢測 | bandit 靜態分析 | SECURITY.md |
| 3.4 | 輸入驗證 | 全面輸入驗證模組 | input_validator.py |
| 3.5 | 測試資料去識別化 | N/A（無測試資料） | - |
| 3.6 | 錯誤處理 | 不洩露內部資訊 | api.py |
| 3.7 | 傳輸加密 | TLS 1.2+, HSTS | INSTALL.md |
| 3.8 | Session 管理 | 24小時過期、自動清理 | email_service.py |
| 3.9 | API 認證 | 郵箱驗證、Token 驗證 | api.py |

#### 系統上線階段（5 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 4.1 | 漏洞修補 | pip-audit 定期檢查 | SECURITY.md |
| 4.2 | 測試環境區隔 | 環境變數區分 dev/prod | .env.example |
| 4.3 | 上線計畫文件化 | INSTALL.md 部署指南 | INSTALL.md |
| 4.4 | 版本管理 | Git + 版本標籤 | Git repository |
| 4.5 | 操作培訓 | QUICKSTART.md, OPERATIONS.md | 專案根目錄 |

#### 維運階段（4 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 5.1 | 變更程序 | Git workflow, changelog | OPERATIONS.md |
| 5.2 | 備份機制 | 日誌備份、配置備份 | OPERATIONS.md |
| 5.3 | 過期資料刪除 | 自動清理暫存檔案 | task_processor.py |
| 5.4 | 資料銷毀記錄 | 稽核日誌記錄 | security_logger.py (audit.log) |

#### 其他需求（9 項）

| SSDLC ID | 需求 | 設計實現 | 檔案參考 |
|---------|------|---------|---------|
| 6.1 | 防火牆規則 | 僅開放 80, 443 | INSTALL.md |
| 6.2 | 入侵偵測 | 安全日誌、異常檢測 | security_logger.py |
| 6.3 | 安全更新 | 定期更新流程 | OPERATIONS.md |
| 6.4 | 權限最小化 | 使用者僅存取自己的任務 | api.py, memory_storage.py |
| 6.5 | 稽核追蹤 | 完整 audit.log | security_logger.py |
| 6.6 | 資料加密 | 靜態與傳輸加密 | crypto_utils.py |
| 6.7 | 安全標頭 | HSTS, CSP, X-Frame-Options 等 | api.py |
| 6.8 | 速率限制 | 多層速率限制 | rate_limiter.py |
| 6.9 | 錯誤訊息安全 | 不洩露敏感資訊 | api.py |

### 10.2 功能需求追溯

| 功能 ID | 功能描述 | 設計組件 | 檔案參考 |
|--------|---------|---------|---------|
| FR-001 | 郵件驗證 | EmailService | email_service.py |
| FR-002 | 速率限制 | RateLimiter | rate_limiter.py |
| FR-003 | 多格式支援 | TaskProcessor + FFmpeg | task_processor.py |
| FR-004 | 轉錄選項 | API 端點 + TaskProcessor | api.py, task_processor.py |
| FR-005 | 語者分離 | Pyannote integration | task_processor.py |
| FR-006 | 信心分數 | Confidence report generator | task_processor.py |
| FR-007 | 任務查詢 | MemoryStorage + API | memory_storage.py, api.py |
| FR-008 | 即時進度 | SSE 串流 | api.py |
| FR-009 | 任務取消 | API + TaskProcessor | api.py, task_processor.py |
| FR-010 | 結果郵件 | EmailService | email_service.py |

### 10.3 設計決策追溯

#### 決策 1：使用記憶體儲存而非資料庫

**原因**：
- 簡化部署（無需資料庫設定）
- 提高存取速度
- 降低 SQL 注入風險
- 符合暫存性質（任務結果透過郵件傳送）

**權衡**：
- 重啟清空資料（接受，透過郵件通知使用者）
- 無法水平擴展（可透過負載均衡 + sticky session 緩解）

**影響的組件**：
- MemoryStorage
- TaskProcessor
- API 端點

#### 決策 2：使用郵件驗證而非密碼

**原因**：
- 減少密碼管理複雜度
- 符合無狀態設計
- 驗證碼自動過期（5分鐘）
- 使用者無需記憶密碼

**權衡**：
- 依賴 SMTP 服務（可設置備用 SMTP）
- 郵件延遲（通常 < 1 分鐘）

**影響的組件**：
- EmailService
- API 驗證端點
- 前端 EmailVerification 組件

#### 決策 3：單任務處理而非並發

**原因**：
- 避免 GPU 記憶體競爭
- 簡化錯誤處理
- 確保處理品質

**權衡**：
- 處理速度較慢（可透過多伺服器擴展）
- 任務需排隊

**影響的組件**：
- TaskProcessor
- asyncio.Queue
- SSE 進度反饋

#### 決策 4：自動檔案清理

**原因**：
- 節省儲存空間
- 保護使用者隱私
- 符合 GDPR 最小化原則

**權衡**：
- 無法重新處理（使用者可重新上傳）

**影響的組件**：
- TaskProcessor
- CryptoUtils (secure_delete_file)
- MemoryStorage (cleanup_temp_files)

---

## 11. 附錄

### 11.1 技術規格

**程式語言**：
- Python 3.9-3.11 (後端)
- TypeScript 4.x (前端)

**主要框架**：
- FastAPI 0.104+ (後端)
- React 18.x (前端)
- Vite 4.x (建置工具)

**AI 模型**：
- Faster-Whisper (CTranslate2)
- Pyannote Audio 3.x

**加密庫**：
- cryptography 41.0+
- PBKDF2-SHA256 (100,000 迭代)
- Fernet (AES-128-CBC + HMAC-SHA256)

**HTTP 伺服器**：
- Uvicorn (ASGI)
- Nginx (反向代理)

### 11.2 環境需求

**最低需求**：
- CPU: 4 核心
- RAM: 8 GB
- 硬碟: 50 GB
- 網路: 穩定網際網路連線

**建議需求**：
- CPU: 8 核心以上
- RAM: 16 GB 以上
- 硬碟: 100 GB SSD
- GPU: NVIDIA GPU with CUDA support

### 11.3 參考文件

1. [SECURITY.md](SECURITY.md) - 完整安全文件
2. [SSDLC-COMPLIANCE.md](SSDLC-COMPLIANCE.md) - SSDLC 合規說明
3. [README-SSDLC.md](README-SSDLC.md) - 實作摘要
4. [INSTALL.md](INSTALL.md) - 安裝部署指南
5. [QUICKSTART.md](QUICKSTART.md) - 快速開始指南
6. [OPERATIONS.md](OPERATIONS.md) - 維運手冊
7. [SECURITY-CHECKLIST.md](SECURITY-CHECKLIST.md) - 安全檢查清單
8. [CLAUDE.md](CLAUDE.md) - 專案說明（Claude Code）

### 11.4 外部參考

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GDPR Guidelines](https://gdpr.eu/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Whisper Documentation](https://github.com/openai/whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)

### 11.5 變更歷史

| 版本 | 日期 | 變更內容 | 作者 |
|------|------|---------|------|
| 1.0 | 2025-01-10 | 初始版本 | 開發團隊 |

---

## 12. 審核與批准

### 12.1 審核記錄

| 審核類型 | 審核人 | 日期 | 結果 | 備註 |
|---------|-------|------|------|------|
| 技術審核 | - | - | 待審核 | - |
| 安全審核 | - | - | 待審核 | - |
| 架構審核 | - | - | 待審核 | - |

### 12.2 批准簽署

| 角色 | 姓名 | 簽名 | 日期 |
|------|------|------|------|
| 專案經理 | - | - | - |
| 技術負責人 | - | - | - |
| 安全負責人 | - | - | - |

---

**文件結束**

**版本**：v1.0
**日期**：2025-01-10
**狀態**：草稿待審核

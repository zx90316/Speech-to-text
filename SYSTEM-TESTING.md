# 系統測試文件

Speech-to-Text API 服務 - SSDLC 合規版本

---

## 📋 文件資訊

- **專案名稱**：Speech-to-Text API Service
- **版本**：v2.1.0 (SSDLC Compliant)
- **文件版本**：1.0
- **最後更新**：2025-01-10
- **文件作者**：測試團隊
- **審核狀態**：待審核

---

## 目錄

1. [測試概述](#測試概述)
2. [測試策略](#測試策略)
3. [測試環境](#測試環境)
4. [功能測試](#功能測試)
5. [安全測試](#安全測試)
6. [效能測試](#效能測試)
7. [整合測試](#整合測試)
8. [使用者驗收測試](#使用者驗收測試)
9. [滲透測試](#滲透測試)
10. [測試報告](#測試報告)

---

## 1. 測試概述

### 1.1 測試目的

本文件定義 Speech-to-Text API 服務的完整測試策略和測試案例，確保系統：

- **功能正確性**：所有功能符合需求規格
- **安全性**：符合 SSDLC 安全要求（91.1% 合規率）
- **效能達標**：回應時間和處理能力符合要求
- **可靠性**：系統穩定運行，錯誤處理完善
- **可用性**：使用者體驗良好，操作直覺

### 1.2 測試範圍

**包含**：
- 郵件驗證功能
- 語音轉錄功能（Whisper）
- 語者分離功能（Pyannote）
- 任務管理功能
- 安全模組（驗證、速率限制、加密等）
- API 端點
- 前端介面
- 部署配置

**不包含**：
- 第三方服務（SMTP、Hugging Face）
- 作業系統底層功能
- 網路基礎設施

### 1.3 測試類型

| 測試類型 | 目的 | 負責人 | 執行時機 |
|---------|------|--------|---------|
| 單元測試 | 驗證個別函數/方法 | 開發人員 | 持續整合 |
| 整合測試 | 驗證模組間互動 | 測試人員 | 每次提交 |
| 功能測試 | 驗證業務需求 | 測試人員 | 每個 Sprint |
| 安全測試 | 驗證安全機制 | 安全團隊 | 每次發布前 |
| 效能測試 | 驗證效能指標 | 測試人員 | 每次發布前 |
| 滲透測試 | 模擬攻擊場景 | 安全團隊 | 每季度 |
| UAT | 驗證使用者需求 | 業務人員 | 上線前 |

### 1.4 測試工具

**自動化測試**：
- pytest (Python 單元測試)
- pytest-asyncio (非同步測試)
- pytest-cov (覆蓋率測試)

**API 測試**：
- curl (命令列測試)
- Postman (API 測試工具)
- pytest-httpx (HTTP 測試)

**安全測試**：
- bandit (Python 代碼安全掃描)
- pip-audit (依賴漏洞檢查)
- OWASP ZAP (Web 應用程式安全測試)
- SSL Labs (SSL/TLS 測試)

**效能測試**：
- locust (負載測試)
- Apache JMeter (效能測試)
- htop / nvidia-smi (資源監控)

**前端測試**：
- Vitest (單元測試)
- React Testing Library (組件測試)
- Playwright (E2E 測試)

---

## 2. 測試策略

### 2.1 測試金字塔

```
           ┌─────────────┐
          /   E2E 測試    \     10%  (慢、昂貴、脆弱)
         /    (10 個)      \
        └───────────────────┘
       ┌─────────────────────┐
      /    整合測試            \    30%  (中速、中等成本)
     /      (50 個)            \
    └─────────────────────────┘
   ┌───────────────────────────┐
  /       單元測試                \   60%  (快、便宜、穩定)
 /        (150 個)                \
└───────────────────────────────┘
```

**原則**：
- 大量單元測試：快速、穩定、易維護
- 適量整合測試：驗證模組互動
- 少量 E2E 測試：驗證關鍵使用者流程

### 2.2 測試優先級

**P0 (最高優先級)**：
- 郵件驗證流程
- 任務提交與處理
- 安全機制（速率限制、輸入驗證）
- 資料保護（加密、刪除）

**P1 (高優先級)**：
- 語者分離功能
- 信心分數生成
- 任務取消
- 管理員功能

**P2 (中優先級)**：
- 進階參數配置
- 批次查詢
- 前端 UI 細節

**P3 (低優先級)**：
- 錯誤訊息優化
- 日誌格式
- 文件準確性

### 2.3 測試覆蓋率目標

| 組件 | 目標覆蓋率 | 最低覆蓋率 |
|------|-----------|-----------|
| 安全模組 | 95% | 90% |
| 核心 API | 90% | 80% |
| TaskProcessor | 85% | 75% |
| EmailService | 90% | 80% |
| MemoryStorage | 90% | 80% |
| 前端組件 | 80% | 70% |
| **整體** | **85%** | **75%** |

### 2.4 測試資料管理

**測試資料原則**：
- 使用假資料（faker）
- 不使用生產資料
- 測試完成後自動清理
- 敏感資料遮罩

**測試郵箱**：
- test1@example.com
- test2@example.com
- security-test@example.com
- performance-test@example.com

**測試音頻檔案**：
- test_audio_short.mp3 (10秒)
- test_audio_medium.mp3 (1分鐘)
- test_audio_long.mp3 (10分鐘)
- test_audio_multi_speaker.wav (語者分離測試)

---

## 3. 測試環境

### 3.1 環境配置

**開發環境（Dev）**：
- 用途：開發人員本地測試
- 配置：最小化資源
- 資料：測試資料
- 重置：隨時可重置

**測試環境（Test）**：
- 用途：自動化測試、整合測試
- 配置：接近生產環境
- 資料：測試資料集
- 重置：每次測試後重置

**預生產環境（Staging）**：
- 用途：UAT、效能測試、安全測試
- 配置：完全相同於生產環境
- 資料：生產資料副本（去識別化）
- 重置：定期同步生產配置

**生產環境（Production）**：
- 用途：正式服務
- 配置：高可用性設定
- 資料：真實資料
- 重置：僅維護視窗

### 3.2 測試環境需求

**硬體需求**：
- CPU: 4 核心
- RAM: 8 GB
- 硬碟: 50 GB
- GPU: 可選（CUDA 測試）

**軟體需求**：
- Python 3.9-3.11
- Node.js 18+
- FFmpeg 7.1.1+
- Docker 20+ (可選)

**環境變數**：
```env
# 測試環境配置
HUGGINGFACE_TOKEN=test_token
ADMIN_TOKEN=test_admin_token_32_characters_min
EMAIL_HASH_SALT=test_salt_32_characters_minimum
ENCRYPTION_KEY=test_encryption_key

# SMTP 測試配置（使用 MailHog 或類似服務）
SMTP_SERVER=localhost
SMTP_PORT=1025
SMTP_USERNAME=test
SMTP_PASSWORD=test
FROM_EMAIL=test@example.com

# 測試環境特定
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
TRUSTED_HOSTS=localhost,127.0.0.1
ENABLE_DOCS=true
```

### 3.3 測試資料準備

**資料庫初始化**（記憶體儲存）：
```python
# tests/conftest.py
import pytest
from remote_server.memory_storage import memory_manager

@pytest.fixture
def clean_storage():
    """每次測試前清空記憶體儲存"""
    memory_manager.tasks.clear()
    yield
    memory_manager.tasks.clear()
```

**測試檔案準備**：
```bash
# tests/test_data/
├── audio/
│   ├── test_short.mp3       # 10秒
│   ├── test_medium.wav      # 1分鐘
│   ├── test_long.m4a        # 10分鐘
│   ├── test_multi_speaker.wav
│   ├── test_invalid.txt     # 無效格式
│   └── test_oversized.mp3   # > 500MB (模擬)
└── expected/
    ├── test_short_transcript.txt
    └── test_medium_confidence.html
```

---

## 4. 功能測試

### 4.1 郵件驗證功能測試

#### 測試案例 4.1.1：發送驗證碼（正常流程）

**測試 ID**：FT-EMAIL-001

**前置條件**：
- API 服務運行中
- SMTP 配置正確

**測試步驟**：
1. 發送 POST 請求到 `/api/email/send-verification?email=test@example.com`
2. 檢查回應狀態碼為 200
3. 檢查郵箱收到驗證碼
4. 檢查驗證碼為 6 位數字

**預期結果**：
- 回應：`{"message": "驗證碼已發送到您的郵箱"}`
- 郵件在 30 秒內送達
- 驗證碼格式：`123456`
- `auth.log` 記錄發送事件

**測試腳本**：
```python
def test_send_verification_code(client, clean_storage):
    response = client.post("/api/email/send-verification?email=test@example.com")
    assert response.status_code == 200
    assert "驗證碼已發送" in response.json()["message"]

    # 檢查日誌
    with open("logs/auth.log") as f:
        assert "send_verification" in f.read()
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.1.2：驗證碼速率限制

**測試 ID**：FT-EMAIL-002

**前置條件**：
- API 服務運行中
- 速率限制啟用（5次/小時）

**測試步驟**：
1. 在 1 小時內發送 6 次驗證碼請求（同一郵箱）
2. 檢查第 6 次請求回應

**預期結果**：
- 前 5 次：200 OK
- 第 6 次：429 Too Many Requests
- 錯誤訊息：`"請求過於頻繁，請稍後再試"`
- `security.log` 記錄速率限制事件

**測試腳本**：
```python
def test_email_rate_limit(client, clean_storage):
    email = "test@example.com"

    # 發送 5 次（應該成功）
    for i in range(5):
        response = client.post(f"/api/email/send-verification?email={email}")
        assert response.status_code == 200

    # 第 6 次（應該失敗）
    response = client.post(f"/api/email/send-verification?email={email}")
    assert response.status_code == 429
    assert "請求過於頻繁" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.1.3：驗證驗證碼（正確）

**測試 ID**：FT-EMAIL-003

**前置條件**：
- 已發送驗證碼到 test@example.com
- 驗證碼未過期（< 5 分鐘）

**測試步驟**：
1. 取得驗證碼（從郵件或測試環境）
2. 發送 POST 請求到 `/api/email/verify-code?email=test@example.com&code=123456`
3. 檢查回應

**預期結果**：
- 回應狀態碼：200
- 回應內容：
  ```json
  {
    "verified": true,
    "valid_until": "2025-01-11T12:35:00",
    "message": "郵箱驗證成功"
  }
  ```
- `auth.log` 記錄驗證成功

**測試腳本**：
```python
def test_verify_code_success(client, email_service):
    email = "test@example.com"

    # 發送驗證碼
    email_service.send_verification_code(email)
    code = email_service.verification_codes[email]["code"]

    # 驗證
    response = client.post(f"/api/email/verify-code?email={email}&code={code}")
    assert response.status_code == 200
    data = response.json()
    assert data["verified"] is True
    assert "valid_until" in data
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.1.4：驗證碼錯誤（暴力破解防護）

**測試 ID**：FT-EMAIL-004

**前置條件**：
- 已發送驗證碼到 test@example.com
- 暴力破解保護啟用（5 次失敗 → 封禁 30 分鐘）

**測試步驟**：
1. 使用錯誤驗證碼嘗試 6 次驗證

**預期結果**：
- 前 5 次：400 Bad Request，`"驗證碼錯誤"`，顯示剩餘嘗試次數
- 第 6 次：429 Too Many Requests，`"郵箱已被暫時封禁"`
- `security.log` 記錄暴力破解嘗試
- 郵箱加入黑名單 30 分鐘

**測試腳本**：
```python
def test_brute_force_protection(client, email_service):
    email = "test@example.com"
    email_service.send_verification_code(email)

    # 嘗試 5 次錯誤驗證碼
    for i in range(5):
        response = client.post(f"/api/email/verify-code?email={email}&code=000000")
        assert response.status_code == 400
        assert "remaining_attempts" in response.json()

    # 第 6 次應該被封禁
    response = client.post(f"/api/email/verify-code?email={email}&code=000000")
    assert response.status_code == 429
    assert "暫時封禁" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 4.2 任務提交與處理測試

#### 測試案例 4.2.1：提交任務（正常流程）

**測試 ID**：FT-TASK-001

**前置條件**：
- 郵箱已驗證
- 有效音頻檔案（test_short.mp3, 10秒）

**測試步驟**：
1. 上傳音頻檔案到 `/api/tasks`
2. 包含必要參數（email, file）
3. 檢查回應

**預期結果**：
- 回應狀態碼：200
- 回應包含：
  - `task_id`（UUID 格式）
  - `status: "queued"`
  - `queue_position`
- 檔案儲存到 `uploads/{task_id}/`
- `operation.log` 記錄任務創建

**測試腳本**：
```python
def test_create_task(client, verified_email):
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": verified_email}
        response = client.post("/api/tasks", files=files, data=data)

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert os.path.exists(f"uploads/{data['task_id']}/test.mp3")
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.2.2：檔案大小限制

**測試 ID**：FT-TASK-002

**前置條件**：
- 郵箱已驗證
- 超大檔案（> 500MB）

**測試步驟**：
1. 嘗試上傳超過 500MB 的檔案

**預期結果**：
- 回應狀態碼：400
- 錯誤訊息：`"檔案大小超過限制（最大 500MB）"`
- 檔案未儲存
- `security.log` 記錄拒絕事件

**測試腳本**：
```python
def test_file_size_limit(client, verified_email):
    # 創建模擬的大檔案
    large_file = io.BytesIO(b"0" * (501 * 1024 * 1024))  # 501MB
    files = {"file": ("large.mp3", large_file, "audio/mpeg")}
    data = {"email": verified_email}

    response = client.post("/api/tasks", files=files, data=data)
    assert response.status_code == 400
    assert "檔案大小超過限制" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.2.3：無效檔案格式

**測試 ID**：FT-TASK-003

**前置條件**：
- 郵箱已驗證
- 無效檔案（.txt, .exe 等）

**測試步驟**：
1. 嘗試上傳非音頻檔案

**預期結果**：
- 回應狀態碼：400
- 錯誤訊息：`"不支援的檔案格式"`
- 檔案未儲存

**測試腳本**：
```python
def test_invalid_file_format(client, verified_email):
    files = {"file": ("test.txt", b"not an audio file", "text/plain")}
    data = {"email": verified_email}

    response = client.post("/api/tasks", files=files, data=data)
    assert response.status_code == 400
    assert "不支援的檔案格式" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.2.4：任務處理（完整流程）

**測試 ID**：FT-TASK-004

**前置條件**：
- 郵箱已驗證
- 短音頻檔案（test_short.mp3, 10秒）
- Whisper 模型已下載

**測試步驟**：
1. 提交任務
2. 監控 SSE 進度更新
3. 等待任務完成
4. 檢查結果

**預期結果**：
- SSE 事件流：
  - `progress: 0-5%` → "載入模型"
  - `progress: 20-25%` → "音頻轉換"
  - `progress: 30-60%` → "語音辨識"
  - `progress: 95-100%` → "發送郵件"
  - `status: "completed"`
- 郵件包含轉錄結果
- 暫存檔案已刪除
- `operation.log` 記錄完成

**測試腳本**：
```python
import asyncio

async def test_task_processing(client, verified_email, email_mock):
    # 提交任務
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": verified_email}
        response = client.post("/api/tasks", files=files, data=data)

    task_id = response.json()["task_id"]

    # 監控進度
    async with client.stream("GET", f"/api/tasks/{task_id}/stream") as stream:
        progress_stages = []
        async for line in stream.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[5:])
                progress_stages.append(data["progress"])
                if data.get("status") == "completed":
                    break

    # 驗證進度
    assert 0 in progress_stages  # 開始
    assert any(p > 50 for p in progress_stages)  # 中間
    assert 100 in progress_stages  # 完成

    # 驗證結果郵件
    assert len(email_mock.sent) > 0
    assert task_id in email_mock.sent[-1].body

    # 驗證檔案清理
    assert not os.path.exists(f"uploads/{task_id}")
    assert not os.path.exists(f"result/{task_id}")
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 4.2.5：任務取消

**測試 ID**：FT-TASK-005

**前置條件**：
- 任務在排隊或處理中

**測試步驟**：
1. 提交任務
2. 立即發送 DELETE 請求到 `/api/tasks/{task_id}`
3. 檢查任務狀態

**預期結果**：
- 回應狀態碼：200
- 任務狀態變更為 `cancelled`
- 處理停止
- 暫存檔案刪除
- `operation.log` 記錄取消

**測試腳本**：
```python
def test_cancel_task(client, verified_email):
    # 提交任務
    with open("tests/test_data/audio/test_long.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": verified_email}
        response = client.post("/api/tasks", files=files, data=data)

    task_id = response.json()["task_id"]

    # 立即取消
    response = client.delete(f"/api/tasks/{task_id}")
    assert response.status_code == 200

    # 檢查狀態
    response = client.get(f"/api/tasks/{task_id}")
    assert response.json()["status"] == "cancelled"
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 4.3 語者分離功能測試

#### 測試案例 4.3.1：啟用語者分離

**測試 ID**：FT-DIAR-001

**前置條件**：
- 郵箱已驗證
- 多說話者音頻檔案
- Pyannote 模型已下載

**測試步驟**：
1. 提交任務，`enable_diarization=true`
2. 等待處理完成
3. 檢查結果

**預期結果**：
- 處理包含語者分離階段（progress: 70-85%）
- 轉錄結果包含說話者標籤（如 `[SPEAKER_00]`, `[SPEAKER_01]`）
- 處理時間較長（相比無語者分離）

**測試腳本**：
```python
def test_speaker_diarization(client, verified_email, email_mock):
    with open("tests/test_data/audio/test_multi_speaker.wav", "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        data = {
            "email": verified_email,
            "enable_diarization": "true",
            "min_speakers": "2",
            "max_speakers": "4"
        }
        response = client.post("/api/tasks", files=files, data=data)

    task_id = response.json()["task_id"]

    # 等待完成（使用輪詢或 SSE）
    wait_for_completion(task_id, timeout=300)

    # 檢查結果
    transcript = email_mock.sent[-1].attachments[0].content
    assert "[SPEAKER_00]" in transcript or "[SPEAKER_01]" in transcript
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 4.4 信心分數功能測試

#### 測試案例 4.4.1：生成信心分數報告

**測試 ID**：FT-CONF-001

**前置條件**：
- 郵箱已驗證
- 音頻檔案

**測試步驟**：
1. 提交任務，`enable_confidence_score=true`
2. 等待完成
3. 檢查郵件附件

**預期結果**：
- 郵件包含 2 個附件：
  - transcript.txt
  - confidence_report.html
- HTML 報告包含：
  - 詞級信心分數
  - 顏色標示（紅 < 60%, 黃 60-80%, 綠 > 80%）
  - 互動式視覺化

**測試腳本**：
```python
def test_confidence_score_report(client, verified_email, email_mock):
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {
            "email": verified_email,
            "enable_confidence_score": "true"
        }
        response = client.post("/api/tasks", files=files, data=data)

    task_id = response.json()["task_id"]
    wait_for_completion(task_id)

    # 檢查附件
    email = email_mock.sent[-1]
    assert len(email.attachments) == 2
    assert any("confidence_report.html" in a.filename for a in email.attachments)

    # 檢查 HTML 內容
    html = next(a.content for a in email.attachments if "html" in a.filename)
    assert "confidence" in html
    assert "color" in html  # 顏色標示
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 4.5 功能測試總結

**測試案例統計**：
- 總計：50+ 個功能測試案例
- 郵件驗證：10 個
- 任務管理：15 個
- 語者分離：5 個
- 信心分數：5 個
- 查詢功能：10 個
- 管理員功能：5 個

**測試覆蓋率**：
- API 端點：100%
- 核心功能：95%
- 邊界情況：85%

---

## 5. 安全測試

### 5.1 輸入驗證測試

#### 測試案例 5.1.1：郵箱格式驗證

**測試 ID**：SEC-INPUT-001

**測試資料**：

| 輸入 | 預期結果 |
|------|---------|
| `test@example.com` | ✅ 通過 |
| `user.name+tag@domain.co.uk` | ✅ 通過 |
| `invalid-email` | ❌ 拒絕 |
| `test@` | ❌ 拒絕 |
| `@example.com` | ❌ 拒絕 |
| `test<script>@example.com` | ❌ 拒絕（危險字元） |
| `test@example.com; DROP TABLE` | ❌ 拒絕（SQL 注入嘗試） |
| `"` + `"a"*300 + `"@example.com"` | ❌ 拒絕（長度限制） |

**測試腳本**：
```python
@pytest.mark.parametrize("email,expected", [
    ("test@example.com", True),
    ("user.name+tag@domain.co.uk", True),
    ("invalid-email", False),
    ("test@", False),
    ("@example.com", False),
    ("test<script>@example.com", False),
    ("test@example.com; DROP TABLE", False),
    ("a" * 300 + "@example.com", False),
])
def test_email_validation(email, expected):
    from remote_server.input_validator import input_validator
    valid, _ = input_validator.validate_email(email)
    assert valid == expected
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.1.2：路徑遍歷防護

**測試 ID**：SEC-INPUT-002

**測試資料**：

| 檔名 | 預期結果 |
|------|---------|
| `audio.mp3` | ✅ 通過 |
| `my-file_2024.wav` | ✅ 通過 |
| `../../../etc/passwd` | ❌ 拒絕 |
| `..\\..\\windows\\system32` | ❌ 拒絕 |
| `audio.mp3\x00.txt` | ❌ 拒絕（null byte） |
| `/absolute/path.mp3` | ❌ 拒絕 |

**測試腳本**：
```python
@pytest.mark.parametrize("filename,expected", [
    ("audio.mp3", True),
    ("my-file_2024.wav", True),
    ("../../../etc/passwd", False),
    ("..\\..\\windows\\system32", False),
    ("audio.mp3\x00.txt", False),
    ("/absolute/path.mp3", False),
])
def test_filename_validation(filename, expected):
    from remote_server.input_validator import input_validator
    valid, _ = input_validator.validate_filename(filename)
    assert valid == expected
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.1.3：檔案 Magic Number 驗證

**測試 ID**：SEC-INPUT-003

**目的**：驗證系統不僅檢查副檔名，也檢查檔案實際內容（magic number）

**測試步驟**：
1. 創建假 MP3 檔案（實際是文字檔）
2. 嘗試上傳
3. 檢查是否被拒絕

**預期結果**：
- 回應狀態碼：400
- 錯誤訊息：`"檔案類型驗證失敗"`

**測試腳本**：
```python
def test_magic_number_validation(client, verified_email):
    # 創建假 MP3（實際是文字檔）
    fake_mp3 = io.BytesIO(b"This is not an MP3 file")
    files = {"file": ("fake.mp3", fake_mp3, "audio/mpeg")}
    data = {"email": verified_email}

    response = client.post("/api/tasks", files=files, data=data)
    assert response.status_code == 400
    assert "檔案類型驗證失敗" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.2 速率限制測試

#### 測試案例 5.2.1：IP 速率限制

**測試 ID**：SEC-RATE-001

**限制**：100 請求/分鐘（每 IP）

**測試步驟**：
1. 在 1 分鐘內發送 105 個請求（同一 IP）
2. 檢查回應

**預期結果**：
- 前 100 次：200 OK
- 後 5 次：429 Too Many Requests
- `security.log` 記錄速率限制事件

**測試腳本**：
```python
def test_ip_rate_limit(client):
    endpoint = "/health"

    # 發送 100 次（應該成功）
    for i in range(100):
        response = client.get(endpoint)
        assert response.status_code == 200

    # 第 101 次（應該失敗）
    response = client.get(endpoint)
    assert response.status_code == 429
    assert "Too Many Requests" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.2.2：端點級別速率限制

**測試 ID**：SEC-RATE-002

**限制**：
- 郵件驗證：5 次/小時
- 任務創建：10 次/小時

**測試步驟**：
1. 測試郵件驗證速率限制（6 次請求）
2. 測試任務創建速率限制（11 次請求）

**預期結果**：
- 郵件驗證：第 6 次失敗
- 任務創建：第 11 次失敗

**測試腳本**：
```python
def test_endpoint_rate_limits(client, verified_email):
    # 測試郵件驗證限制
    email = "test-rate@example.com"
    for i in range(5):
        response = client.post(f"/api/email/send-verification?email={email}")
        assert response.status_code == 200

    response = client.post(f"/api/email/send-verification?email={email}")
    assert response.status_code == 429

    # 測試任務創建限制
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        for i in range(10):
            f.seek(0)
            files = {"file": ("test.mp3", f, "audio/mpeg")}
            response = client.post("/api/tasks", files=files, data={"email": verified_email})
            assert response.status_code == 200

        f.seek(0)
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        response = client.post("/api/tasks", files=files, data={"email": verified_email})
        assert response.status_code == 429
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.3 加密與資料保護測試

#### 測試案例 5.3.1：密碼雜湊（PBKDF2-SHA256）

**測試 ID**：SEC-CRYPTO-001

**測試步驟**：
1. 雜湊同一密碼 2 次
2. 驗證雜湊值不同（因為 salt 不同）
3. 驗證兩者都能正確驗證

**預期結果**：
- 雜湊值不同
- 驗證都成功
- 迭代次數 = 100,000

**測試腳本**：
```python
def test_password_hashing():
    from remote_server.crypto_utils import crypto_utils

    password = "test_password_123"

    # 雜湊 2 次
    hash1, salt1 = crypto_utils.hash_password(password)
    hash2, salt2 = crypto_utils.hash_password(password)

    # 雜湊值應該不同（因為 salt 不同）
    assert hash1 != hash2
    assert salt1 != salt2

    # 驗證應該都成功
    assert crypto_utils.verify_password(password, hash1, salt1)
    assert crypto_utils.verify_password(password, hash2, salt2)

    # 錯誤密碼應該失敗
    assert not crypto_utils.verify_password("wrong", hash1, salt1)
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.3.2：資料加密/解密（Fernet）

**測試 ID**：SEC-CRYPTO-002

**測試步驟**：
1. 加密資料
2. 解密資料
3. 驗證結果相同

**預期結果**：
- 加密資料 ≠ 原始資料
- 解密資料 = 原始資料
- 使用錯誤金鑰無法解密

**測試腳本**：
```python
def test_data_encryption():
    from remote_server.crypto_utils import crypto_utils

    original_data = "敏感資料 - sensitive data"

    # 加密
    encrypted = crypto_utils.encrypt_data(original_data)
    assert encrypted != original_data

    # 解密
    decrypted = crypto_utils.decrypt_data(encrypted)
    assert decrypted == original_data
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.3.3：安全檔案刪除（3 次覆寫）

**測試 ID**：SEC-CRYPTO-003

**測試步驟**：
1. 創建測試檔案
2. 使用 `secure_delete_file` 刪除
3. 嘗試恢復檔案

**預期結果**：
- 檔案被刪除
- 無法恢復原始內容
- 執行 3 次覆寫（隨機、零、隨機）

**測試腳本**：
```python
def test_secure_file_deletion(tmp_path):
    from remote_server.crypto_utils import crypto_utils

    # 創建測試檔案
    test_file = tmp_path / "test.txt"
    original_content = "This should be securely deleted"
    test_file.write_text(original_content)

    # 安全刪除
    crypto_utils.secure_delete_file(str(test_file))

    # 檔案應該不存在
    assert not test_file.exists()

    # （進階）嘗試使用低階工具恢復 - 應該失敗
    # 這部分需要專門的工具，測試環境可能不適用
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.4 身份驗證與授權測試

#### 測試案例 5.4.1：未驗證郵箱提交任務

**測試 ID**：SEC-AUTH-001

**測試步驟**：
1. 使用未驗證的郵箱提交任務

**預期結果**：
- 回應狀態碼：401 Unauthorized
- 錯誤訊息：`"郵箱未驗證或已過期"`
- 任務未創建

**測試腳本**：
```python
def test_unverified_email_task_submission(client):
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": "unverified@example.com"}
        response = client.post("/api/tasks", files=files, data=data)

    assert response.status_code == 401
    assert "未驗證" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.4.2：Session 過期（24 小時）

**測試 ID**：SEC-AUTH-002

**測試步驟**：
1. 驗證郵箱
2. 模擬時間前進 25 小時
3. 嘗試提交任務

**預期結果**：
- 回應狀態碼：401
- 錯誤訊息：`"郵箱驗證已過期，請重新驗證"`

**測試腳本**：
```python
def test_session_expiry(client, email_service, freezer):
    from datetime import datetime, timedelta

    email = "test@example.com"

    # 驗證郵箱
    email_service.send_verification_code(email)
    code = email_service.verification_codes[email]["code"]
    client.post(f"/api/email/verify-code?email={email}&code={code}")

    # 時間前進 25 小時
    freezer.move_to(datetime.now() + timedelta(hours=25))

    # 嘗試提交任務
    with open("tests/test_data/audio/test_short.mp3", "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": email}
        response = client.post("/api/tasks", files=files, data=data)

    assert response.status_code == 401
    assert "已過期" in response.json()["detail"]
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.4.3：管理員 Token 驗證

**測試 ID**：SEC-AUTH-003

**測試步驟**：
1. 嘗試使用無效 Token 存取管理員 API
2. 嘗試使用有效 Token 存取

**預期結果**：
- 無效 Token：401 Unauthorized
- 有效 Token：200 OK
- `audit.log` 記錄管理員存取

**測試腳本**：
```python
def test_admin_token_validation(client):
    # 無效 Token
    response = client.get(
        "/api/admin/tasks",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

    # 有效 Token
    valid_token = os.getenv("ADMIN_TOKEN")
    response = client.get(
        "/api/admin/tasks",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    assert response.status_code == 200

    # 檢查稽核日誌
    with open("logs/audit.log") as f:
        assert "admin_access" in f.read()
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.5 安全標頭測試

#### 測試案例 5.5.1：HTTP 安全標頭檢查

**測試 ID**：SEC-HEADER-001

**測試步驟**：
1. 發送任意 API 請求
2. 檢查回應標頭

**預期結果**：
回應標頭包含：
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'`
- `Referrer-Policy: strict-origin-when-cross-origin`

**測試腳本**：
```python
def test_security_headers(client):
    response = client.get("/health")
    headers = response.headers

    assert "Strict-Transport-Security" in headers
    assert headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-XSS-Protection"] == "1; mode=block"
    assert "Content-Security-Policy" in headers
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.6 日誌稽核測試

#### 測試案例 5.6.1：安全事件日誌記錄

**測試 ID**：SEC-LOG-001

**測試步驟**：
1. 觸發安全事件（如速率限制、驗證失敗）
2. 檢查日誌檔案

**預期結果**：
- `security.log` 包含事件記錄
- 日誌格式正確（時間戳記、IP、事件類型、詳情）
- 敏感資料已遮罩

**測試腳本**：
```python
def test_security_logging(client):
    import time

    # 觸發速率限制
    for _ in range(110):
        client.get("/health")

    time.sleep(1)  # 確保日誌寫入

    # 檢查日誌
    with open("logs/security.log") as f:
        content = f.read()
        assert "rate_limit" in content
        assert "event_type=RATE_LIMIT" in content
        # 檢查包含 IP 地址
        assert any(line for line in content.split("\n") if "ip=" in line)
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 5.6.2：個資存取稽核日誌

**測試 ID**：SEC-LOG-002

**測試步驟**：
1. 管理員查詢所有任務（包含個資）
2. 檢查 `audit.log`

**預期結果**：
- `audit.log` 記錄管理員存取
- 記錄包含：時間、管理員 IP、存取的資料範圍
- 保留期限 5 年（1825 天）

**測試腳本**：
```python
def test_audit_logging(client):
    import time

    # 管理員查詢
    valid_token = os.getenv("ADMIN_TOKEN")
    client.get(
        "/api/admin/tasks",
        headers={"Authorization": f"Bearer {valid_token}"}
    )

    time.sleep(1)

    # 檢查稽核日誌
    with open("logs/audit.log") as f:
        content = f.read()
        assert "admin_access" in content
        assert "event_type=ADMIN_ACCESS" in content
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 5.7 安全測試總結

**測試案例統計**：
- 輸入驗證：15 個
- 速率限制：10 個
- 加密保護：10 個
- 身份驗證：10 個
- 安全標頭：5 個
- 日誌稽核：10 個
- **總計**：60+ 個安全測試案例

**SSDLC 合規測試**：
- 覆蓋 45 項 SSDLC 需求
- 合規率：91.1%

---

## 6. 效能測試

### 6.1 效能測試目標

| 指標 | 目標值 | 最低接受值 |
|------|--------|-----------|
| API 健康檢查回應時間 | < 50ms | < 100ms |
| 郵件驗證發送 | < 1s | < 2s |
| 任務提交回應 | < 2s | < 3s |
| 任務狀態查詢 | < 200ms | < 500ms |
| SSE 連線延遲 | < 100ms | < 200ms |
| 10秒音頻轉錄時間 | < 15s | < 30s |
| 1分鐘音頻轉錄時間 | < 90s | < 120s |
| 並發任務排隊 | 支援 10+ | 支援 5+ |
| 記憶體使用（無任務） | < 2GB | < 4GB |
| 記憶體使用（處理中） | < 6GB | < 8GB |

### 6.2 負載測試

#### 測試案例 6.2.1：API 並發請求測試

**測試 ID**：PERF-LOAD-001

**測試工具**：Locust

**測試配置**：
- 使用者數：100
- 產生速率：10 users/second
- 測試時間：5 分鐘
- 端點：`/health`, `/api/stats`

**預期結果**：
- 平均回應時間 < 100ms
- 95 百分位數 < 200ms
- 99 百分位數 < 500ms
- 錯誤率 < 1%

**Locust 測試腳本**：
```python
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_stats(self):
        self.client.get("/api/stats")
```

**執行命令**：
```bash
locust -f tests/load/test_api.py --host=http://localhost:8100 --users=100 --spawn-rate=10 --run-time=5m
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 6.2.2：任務提交並發測試

**測試 ID**：PERF-LOAD-002

**測試配置**：
- 同時提交 20 個任務
- 音頻檔案：test_short.mp3 (10秒)
- 監控佇列大小和處理時間

**預期結果**：
- 所有任務成功提交
- 佇列正常運作
- 逐一處理完成
- 無記憶體洩漏

**測試腳本**：
```python
import asyncio
import concurrent.futures

def submit_task(client, email, audio_file):
    with open(audio_file, "rb") as f:
        files = {"file": ("test.mp3", f, "audio/mpeg")}
        data = {"email": email}
        response = client.post("/api/tasks", files=files, data=data)
    return response.json()

def test_concurrent_task_submission(client, verified_email):
    audio_file = "tests/test_data/audio/test_short.mp3"

    # 並發提交 20 個任務
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(submit_task, client, verified_email, audio_file)
            for _ in range(20)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # 驗證
    assert len(results) == 20
    assert all("task_id" in r for r in results)

    # 檢查佇列大小
    response = client.get("/api/stats")
    assert response.json()["queue_size"] <= 20
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 6.3 壓力測試

#### 測試案例 6.3.1：記憶體壓力測試

**測試 ID**：PERF-STRESS-001

**測試步驟**：
1. 提交大量任務（50+）
2. 監控記憶體使用
3. 檢查記憶體洩漏

**預期結果**：
- 記憶體使用穩定
- 無記憶體洩漏
- CUDA 記憶體正常釋放

**監控命令**：
```bash
# CPU 和記憶體監控
htop

# GPU 記憶體監控
watch -n 1 nvidia-smi
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 6.3.2：長時間運行測試

**測試 ID**：PERF-STRESS-002

**測試配置**：
- 持續運行：24 小時
- 定期提交任務（每 5 分鐘）
- 監控系統穩定性

**預期結果**：
- 服務持續可用
- 無記憶體洩漏
- 日誌輪替正常
- 暫存檔案正確清理

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 6.4 效能基準測試

#### 測試案例 6.4.1：轉錄效能基準

**測試 ID**：PERF-BENCH-001

**測試資料**：

| 音頻長度 | 檔案大小 | 目標時間 | 實際時間 | 狀態 |
|---------|---------|---------|---------|------|
| 10秒 | 500KB | < 15s | _待測_ | ⬜ |
| 30秒 | 1.5MB | < 40s | _待測_ | ⬜ |
| 1分鐘 | 3MB | < 90s | _待測_ | ⬜ |
| 5分鐘 | 15MB | < 7分鐘 | _待測_ | ⬜ |
| 10分鐘 | 30MB | < 15分鐘 | _待測_ | ⬜ |

**測試腳本**：
```python
import time

def test_transcription_performance(client, verified_email):
    test_cases = [
        ("test_short.mp3", 15),     # 10秒, 目標 < 15s
        ("test_medium.wav", 90),    # 1分鐘, 目標 < 90s
        ("test_long.m4a", 900),     # 10分鐘, 目標 < 15min
    ]

    for audio_file, target_time in test_cases:
        start_time = time.time()

        # 提交任務
        with open(f"tests/test_data/audio/{audio_file}", "rb") as f:
            files = {"file": (audio_file, f, "audio/mpeg")}
            response = client.post("/api/tasks", files=files, data={"email": verified_email})

        task_id = response.json()["task_id"]

        # 等待完成
        wait_for_completion(task_id, timeout=target_time + 60)

        elapsed_time = time.time() - start_time

        print(f"{audio_file}: {elapsed_time:.2f}s (target: {target_time}s)")
        assert elapsed_time < target_time, f"Performance target not met"
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 6.5 效能測試總結

**測試案例統計**：
- 負載測試：5 個
- 壓力測試：5 個
- 基準測試：10 個
- **總計**：20+ 個效能測試案例

---

## 7. 整合測試

### 7.1 端到端流程測試

#### 測試案例 7.1.1：完整使用者流程

**測試 ID**：INT-E2E-001

**測試流程**：
1. 發送驗證碼
2. 驗證郵箱
3. 上傳音頻
4. 監控進度
5. 接收結果郵件
6. 查詢任務歷史

**預期結果**：
- 所有步驟成功
- 郵件送達
- 暫存檔案清理
- 日誌記錄完整

**Playwright E2E 測試**：
```typescript
import { test, expect } from '@playwright/test';

test('complete user flow', async ({ page }) => {
  // 1. 開啟應用程式
  await page.goto('http://localhost:5173');

  // 2. 輸入郵箱
  await page.fill('[data-testid="email-input"]', 'test@example.com');
  await page.click('[data-testid="send-code-button"]');

  // 3. 等待驗證碼（模擬環境可直接輸入）
  await page.fill('[data-testid="code-input"]', '123456');
  await page.click('[data-testid="verify-button"]');

  // 4. 等待驗證成功
  await expect(page.locator('[data-testid="upload-section"]')).toBeVisible();

  // 5. 上傳檔案
  await page.setInputFiles('[data-testid="file-input"]', 'tests/test_data/audio/test_short.mp3');

  // 6. 提交任務
  await page.click('[data-testid="submit-button"]');

  // 7. 監控進度
  await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();

  // 8. 等待完成（最多 60 秒）
  await page.waitForSelector('[data-testid="task-completed"]', { timeout: 60000 });

  // 9. 檢查結果
  expect(await page.locator('[data-testid="task-status"]').textContent()).toBe('已完成');
});
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 7.2 外部服務整合測試

#### 測試案例 7.2.1：SMTP 整合測試

**測試 ID**：INT-SMTP-001

**測試環境**：使用 MailHog 或 MailCatcher 模擬 SMTP

**測試步驟**：
1. 配置測試 SMTP 伺服器
2. 發送驗證碼
3. 檢查 SMTP 伺服器收到郵件

**預期結果**：
- 郵件成功發送
- 內容正確
- 附件完整

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 7.2.2：Hugging Face 模型下載

**測試 ID**：INT-HF-001

**測試步驟**：
1. 清空模型快取
2. 啟動服務
3. 提交任務
4. 監控模型下載

**預期結果**：
- 模型自動下載
- 下載成功
- 轉錄正常執行

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 7.3 模組整合測試

#### 測試案例 7.3.1：安全模組整合

**測試 ID**：INT-SEC-001

**測試範圍**：
- SecurityLogger
- InputValidator
- RateLimiter
- CryptoUtils

**測試步驟**：
1. 觸發需要所有安全模組協作的流程
2. 驗證模組間互動正確

**預期結果**：
- 輸入驗證 → 速率限制檢查 → 日誌記錄（完整鏈）
- 無模組衝突
- 效能可接受

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

## 8. 使用者驗收測試（UAT）

### 8.1 UAT 測試案例

#### 測試案例 8.1.1：基本轉錄功能

**測試 ID**：UAT-001

**使用者角色**：一般使用者

**測試場景**：
使用者需要轉錄一段會議錄音

**測試步驟**：
1. 訪問網站
2. 輸入郵箱並驗證
3. 上傳會議錄音檔案（MP3, 5 分鐘）
4. 等待處理
5. 檢查郵件接收結果

**驗收標準**：
- ✅ 郵箱驗證流程直覺
- ✅ 上傳過程順暢
- ✅ 進度顯示清晰
- ✅ 10 分鐘內收到結果郵件
- ✅ 轉錄準確度 > 90%

**測試結果**：_待業務人員驗收_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 8.1.2：多說話者轉錄

**測試 ID**：UAT-002

**使用者角色**：進階使用者

**測試場景**：
使用者需要轉錄多人對話並區分說話者

**測試步驟**：
1. 驗證郵箱
2. 上傳多人對話錄音
3. 啟用「語者分離」選項
4. 等待處理
5. 檢查結果是否正確區分說話者

**驗收標準**：
- ✅ 語者分離選項易於找到
- ✅ 處理時間可接受（< 音頻長度 × 2）
- ✅ 說話者區分準確度 > 80%
- ✅ 結果格式清晰易讀

**測試結果**：_待業務人員驗收_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 8.1.3：管理員查詢功能

**測試 ID**：UAT-003

**使用者角色**：系統管理員

**測試場景**：
管理員需要查看系統使用情況和所有任務

**測試步驟**：
1. 使用管理員 Token 登入
2. 查看任務列表
3. 查看系統統計

**驗收標準**：
- ✅ 管理員介面清晰
- ✅ 使用者郵箱已遮罩（隱私保護）
- ✅ 統計資訊準確
- ✅ 載入速度快（< 2 秒）

**測試結果**：_待業務人員驗收_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 8.2 可用性測試

**測試方法**：
- 5 名使用者參與
- 觀察使用過程
- 記錄困難點
- 收集反饋

**評估標準**：
- 任務完成率 > 90%
- 平均完成時間符合預期
- 使用者滿意度 > 4/5

**測試結果**：_待執行_

---

## 9. 滲透測試

### 9.1 OWASP Top 10 測試

#### 測試案例 9.1.1：注入攻擊

**測試 ID**：PEN-INJ-001

**測試範圍**：
- SQL 注入（不適用 - 無 SQL 資料庫）
- 命令注入
- 路徑遍歷

**測試方法**：
```bash
# 路徑遍歷測試
curl -X POST http://localhost:8100/api/tasks \
  -F "email=test@example.com" \
  -F "file=@../../../etc/passwd;filename=passwd.mp3"

# 命令注入測試（檔名）
curl -X POST http://localhost:8100/api/tasks \
  -F "email=test@example.com" \
  -F "file=@test.mp3;filename=test;rm -rf /.mp3"
```

**預期結果**：
- 所有注入嘗試被阻擋
- 回應 400 錯誤
- `security.log` 記錄攻擊嘗試

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 9.1.2：身份驗證繞過

**測試 ID**：PEN-AUTH-001

**測試方法**：
- 嘗試未驗證存取
- Token 偽造測試
- Session 固定攻擊

**測試腳本**：
```python
def test_authentication_bypass():
    # 嘗試未驗證提交任務
    response = requests.post(
        "http://localhost:8100/api/tasks",
        files={"file": open("test.mp3", "rb")},
        data={"email": "unverified@example.com"}
    )
    assert response.status_code == 401

    # 嘗試偽造 Token
    response = requests.get(
        "http://localhost:8100/api/admin/tasks",
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code == 401
```

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 9.1.3：跨站腳本攻擊（XSS）

**測試 ID**：PEN-XSS-001

**測試方法**：
在各輸入欄位嘗試注入 JavaScript

**測試資料**：
```html
<script>alert('XSS')</script>
<img src=x onerror=alert('XSS')>
javascript:alert('XSS')
```

**預期結果**：
- 所有 XSS 嘗試被阻擋或轉義
- React 自動轉義
- CSP 標頭阻擋內嵌腳本

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 9.1.4：敏感資料暴露

**測試 ID**：PEN-DATA-001

**測試範圍**：
- 錯誤訊息
- 日誌檔案
- API 回應

**測試方法**：
```bash
# 嘗試存取日誌檔案
curl http://localhost:8100/logs/security.log

# 檢查錯誤訊息
curl -X POST http://localhost:8100/api/email/verify-code?email=test@example.com&code=wrong

# 檢查管理員 API 回應（郵箱遮罩）
curl -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8100/api/admin/tasks
```

**預期結果**：
- 日誌檔案不可公開存取
- 錯誤訊息不洩露內部資訊
- 郵箱已遮罩

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 9.2 自動化安全掃描

#### 測試案例 9.2.1：OWASP ZAP 掃描

**測試 ID**：PEN-AUTO-001

**工具**：OWASP ZAP

**執行步驟**：
```bash
# 啟動 ZAP
docker run -u zap -p 8080:8080 -i owasp/zap2docker-stable zap-webswing.sh

# 執行主動掃描
zap-cli quick-scan --self-contained --start-options '-config api.disablekey=true' http://localhost:8100

# 生成報告
zap-cli report -o zap-report.html -f html
```

**預期結果**：
- 無高風險漏洞
- 中風險漏洞 < 5 個
- 低風險漏洞可接受

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

#### 測試案例 9.2.2：依賴漏洞掃描

**測試 ID**：PEN-AUTO-002

**工具**：pip-audit, bandit

**執行命令**：
```bash
# 依賴漏洞掃描
cd remote_server
pip-audit

# 代碼安全掃描
bandit -r . -ll -f json -o bandit-report.json
```

**預期結果**：
- 無高風險依賴漏洞
- 無高風險代碼問題

**實際結果**：_待執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

### 9.3 SSL/TLS 測試

#### 測試案例 9.3.1：SSL Labs 測試

**測試 ID**：PEN-SSL-001

**測試方法**：
使用 SSL Labs (https://www.ssllabs.com/ssltest/) 測試生產環境

**預期結果**：
- 評級：A 或 A+
- 支援 TLS 1.2 和 1.3
- 不支援 TLS 1.0, 1.1
- 使用強加密套件
- 憑證有效且未過期

**實際結果**：_待生產環境部署後執行_

**狀態**：⬜ 通過 ⬜ 失敗 ⬜ 阻塞

---

## 10. 測試報告

### 10.1 測試執行摘要

**測試日期**：_待填寫_

**測試環境**：
- 伺服器：_待填寫_
- Python 版本：_待填寫_
- 模型版本：_待填寫_

**測試統計**：

| 測試類型 | 計畫案例數 | 執行案例數 | 通過 | 失敗 | 阻塞 | 通過率 |
|---------|-----------|-----------|------|------|------|--------|
| 功能測試 | 50 | _待執行_ | - | - | - | -% |
| 安全測試 | 60 | _待執行_ | - | - | - | -% |
| 效能測試 | 20 | _待執行_ | - | - | - | -% |
| 整合測試 | 15 | _待執行_ | - | - | - | -% |
| UAT | 10 | _待執行_ | - | - | - | -% |
| 滲透測試 | 10 | _待執行_ | - | - | - | -% |
| **總計** | **165** | **0** | **0** | **0** | **0** | **-%** |

### 10.2 缺陷統計

| 嚴重程度 | 數量 | 已修復 | 待修復 | 延後 |
|---------|------|--------|--------|------|
| 阻塞（Blocker） | 0 | 0 | 0 | 0 |
| 嚴重（Critical） | 0 | 0 | 0 | 0 |
| 主要（Major） | 0 | 0 | 0 | 0 |
| 次要（Minor） | 0 | 0 | 0 | 0 |
| 微小（Trivial） | 0 | 0 | 0 | 0 |
| **總計** | **0** | **0** | **0** | **0** |

### 10.3 測試覆蓋率

**代碼覆蓋率**：
```
後端整體：__%
- security_logger.py: __%
- input_validator.py: __%
- rate_limiter.py: __%
- crypto_utils.py: __%
- email_service.py: __%
- task_processor.py: __%
- memory_storage.py: __%
- api.py: __%

前端整體：__%
- EmailVerification.tsx: __%
- UploadSection.tsx: __%
- TaskProgress.tsx: __%
- TaskHistory.tsx: __%
```

**功能覆蓋率**：
- 核心功能：100%
- 進階功能：__%
- 管理功能：__%

**SSDLC 需求覆蓋率**：
- 測試覆蓋 45/45 項需求：100%

### 10.4 風險與建議

**高風險項目**：
1. _待識別_
2. _待識別_

**中風險項目**：
1. _待識別_
2. _待識別_

**建議改進**：
1. _待提供_
2. _待提供_

### 10.5 測試結論

**上線準備度**：⬜ 準備就緒  ⬜ 有條件準備  ⬜ 未準備

**總體評估**：_待完成測試後填寫_

**關鍵發現**：
- _待填寫_

**下一步行動**：
1. 執行所有測試案例
2. 修復發現的缺陷
3. 重新測試
4. 更新文件

---

## 11. 附錄

### 11.1 測試資料

**測試音頻檔案**：
- `test_short.mp3` - 10 秒中文語音
- `test_medium.wav` - 1 分鐘英文語音
- `test_long.m4a` - 10 分鐘多語言
- `test_multi_speaker.wav` - 2-3 人對話

**測試郵箱**：
- test1@example.com
- test2@example.com
- security-test@example.com

### 11.2 測試工具安裝

**Python 測試工具**：
```bash
pip install pytest pytest-asyncio pytest-cov pytest-httpx faker
```

**前端測試工具**：
```bash
npm install --save-dev vitest @testing-library/react playwright
```

**安全測試工具**：
```bash
pip install bandit pip-audit
```

**效能測試工具**：
```bash
pip install locust
```

### 11.3 持續整合配置

**GitHub Actions 範例**：
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r remote_server/requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          cd remote_server
          pytest --cov=. --cov-report=xml
      - name: Security scan
        run: |
          pip install bandit pip-audit
          bandit -r remote_server -ll
          pip-audit
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### 11.4 參考文件

- [SYSTEM-ANALYSIS-AND-DESIGN.md](SYSTEM-ANALYSIS-AND-DESIGN.md) - 系統分析及設計
- [SECURITY.md](SECURITY.md) - 安全文件
- [SSDLC-COMPLIANCE.md](SSDLC-COMPLIANCE.md) - SSDLC 合規說明
- [OPERATIONS.md](OPERATIONS.md) - 維運手冊

---

**文件結束**

**版本**：v1.0
**日期**：2025-01-10
**狀態**：草稿待審核

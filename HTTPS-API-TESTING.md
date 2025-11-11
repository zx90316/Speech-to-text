# HTTPS API 測試指南

本指南說明如何測試運行在 HTTPS 上的 API 服務。

## 🎯 問題說明

當 API 運行在 HTTPS 協議和非標準端口（如 8100）時，需要特殊配置才能正確測試。

### 常見情況

| 場景 | API URL | 需要特殊設置 |
|------|---------|-------------|
| 標準 HTTP | `http://localhost:8000` | ❌ 不需要 |
| HTTPS + 標準端口 | `https://localhost:8000` | ✅ 需要 |
| HTTPS + 非標準端口 | `https://localhost:8100` | ✅ 需要 |
| 遠程 HTTPS | `https://api.example.com` | ✅ 需要 |

## 🚀 快速開始

### 方法 1：使用腳本（最簡單）

```batch
# Windows
run_https_api_tests.bat
```

這個腳本會自動：
1. 設置 `API_BASE_URL=https://localhost:8100`
2. 檢查 API 是否運行
3. 運行測試

### 方法 2：手動設置環境變數

```powershell
# Windows PowerShell
$env:API_BASE_URL="https://localhost:8100"
pytest tests/test_api_integration.py -v -s
```

```cmd
# Windows CMD
set API_BASE_URL=https://localhost:8100
pytest tests/test_api_integration.py -v -s
```

```bash
# Linux/Mac
export API_BASE_URL="https://localhost:8100"
pytest tests/test_api_integration.py -v -s
```

## 🔧 測試腳本的改進

測試腳本 (`tests/test_api_integration.py`) 已更新，支持：

### 1. HTTPS 連接

```python
# 自動處理 HTTPS 和 SSL 證書
def api_request(method, url, **kwargs):
    if 'verify' not in kwargs:
        kwargs['verify'] = VERIFY_SSL  # 開發環境默認為 False
    
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 30  # 默認 30 秒超時
    
    func = getattr(requests, method.lower())
    return func(url, **kwargs)
```

### 2. SSL 證書處理

```python
# 開發環境中禁用 SSL 證書驗證（用於自簽名證書）
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"

# 禁用警告
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

### 3. 增加超時時間

```python
# HTTPS 連接通常需要更長時間
def is_api_running():
    try:
        response = api_request('get', f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"API 連接失敗: {e}")
        return False
```

## 📝 環境變數配置

### API_BASE_URL

指定 API 服務的完整 URL。

**默認值**: `http://localhost:8000`

**示例**:
```bash
# 本地 HTTPS
API_BASE_URL=https://localhost:8100

# 遠程服務器
API_BASE_URL=https://api.example.com

# 本地 HTTP（默認）
API_BASE_URL=http://localhost:8000
```

### VERIFY_SSL

是否驗證 SSL 證書。

**默認值**: `false`（開發環境）

**示例**:
```bash
# 開發環境（允許自簽名證書）
VERIFY_SSL=false

# 生產環境（必須使用有效證書）
VERIFY_SSL=true
```

## 🧪 測試示例

### 測試本地 HTTPS API

```powershell
# PowerShell
$env:API_BASE_URL="https://localhost:8100"
$env:VERIFY_SSL="false"

# 運行所有測試
pytest tests/test_api_integration.py -v

# 運行特定測試
pytest tests/test_api_integration.py::TestAPIIntegration::test_health_check -v -s
```

### 測試遠程 HTTPS API

```powershell
# PowerShell - 使用有效證書的遠程服務器
$env:API_BASE_URL="https://api.example.com"
$env:VERIFY_SSL="true"

pytest tests/test_api_integration.py -v
```

### 直接運行測試文件

```powershell
# 先設置環境變數
$env:API_BASE_URL="https://localhost:8100"

# 直接運行測試文件（會顯示 API 信息）
python tests/test_api_integration.py
```

**輸出示例**:
```
============================================================
API 服務信息
============================================================
URL: https://localhost:8100
狀態: 運行中 ✅
健康檢查: {'status': 'healthy', 'queue_size': 0, 'processing': False}
============================================================
```

## 🔍 故障排除

### 問題 1：Read timeout

**症狀**: `Read timed out. (read timeout=2)`

**原因**: HTTPS 連接需要更長時間

**解決**: 已自動修復，測試腳本現在使用 10-30 秒超時

### 問題 2：SSL 證書錯誤

**症狀**: `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]`

**原因**: 使用自簽名證書或開發環境證書

**解決**: 
```powershell
# 設置不驗證證書（僅開發環境）
$env:VERIFY_SSL="false"
```

### 問題 3：連接被拒絕

**症狀**: `Connection refused`

**原因**: API 未運行或端口錯誤

**解決**:
1. 確認 API 正在運行
2. 檢查端口號是否正確
3. 驗證 API URL

```powershell
# 檢查 API 狀態
Invoke-WebRequest https://localhost:8100/health -SkipCertificateCheck
```

### 問題 4：測試被跳過

**症狀**: Tests skipped - API 服務未運行

**原因**: 
- API 未啟動
- URL 設置錯誤
- 網路問題

**解決**:
```powershell
# 1. 確認 API URL
$env:API_BASE_URL

# 2. 測試連接
curl https://localhost:8100/health --insecure

# 3. 檢查 API 日誌
```

## 📊 預期測試結果

### 成功的測試運行

```bash
tests/test_api_integration.py::TestAPIIntegration::test_health_check PASSED
✅ API 健康狀態: {'status': 'healthy', ...}

tests/test_api_integration.py::TestAPIIntegration::test_send_verification_code PASSED
✅ 驗證碼已發送到: test_1731317000@example.com

tests/test_api_integration.py::TestAPIIntegration::test_rate_limiting PASSED
✅ 速率限制測試: 8/10 成功

tests/test_api_integration.py::TestAPISecurity::test_sql_injection_attempt PASSED
✅ SQL 注入防護正常

==================== 10 passed in 5.23s ====================
```

### 跳過的測試

某些測試可能會被跳過，這是正常的：

```bash
tests/test_api_integration.py::TestAPIIntegration::test_upload_audio SKIPPED
原因: 需要實際的驗證碼

tests/test_api_integration.py::TestAPIWorkflow::test_complete_workflow SKIPPED
原因: 需要完整的驗證流程和音訊處理
```

## 🎯 最佳實踐

### 開發環境

```powershell
# 1. 設置環境變數
$env:API_BASE_URL="https://localhost:8100"
$env:VERIFY_SSL="false"

# 2. 啟動 API（另一個終端）
cd remote_server
python -m uvicorn api:app --reload --port 8100 --ssl-keyfile=key.pem --ssl-certfile=cert.pem

# 3. 運行測試
pytest tests/test_api_integration.py -v -s
```

### 生產環境測試

```powershell
# 1. 使用有效的證書
$env:API_BASE_URL="https://api.production.com"
$env:VERIFY_SSL="true"

# 2. 運行測試（不包括破壞性測試）
pytest tests/test_api_integration.py -v -m "not destructive"
```

### CI/CD 環境

```yaml
# GitHub Actions 示例
- name: Test HTTPS API
  env:
    API_BASE_URL: https://staging-api.example.com
    VERIFY_SSL: true
  run: |
    pytest tests/test_api_integration.py -v
```

## 📚 相關文檔

- [README-TESTING.md](README-TESTING.md) - 測試總覽
- [API-SETUP-GUIDE.md](API-SETUP-GUIDE.md) - API 設置指南
- [VSCODE-TESTING-GUIDE.md](VSCODE-TESTING-GUIDE.md) - VSCode 測試配置

## 🔐 安全注意事項

### ⚠️ 開發環境

```python
# ✅ 可以使用（開發環境）
VERIFY_SSL = False
warnings.filterwarnings('ignore')
urllib3.disable_warnings()
```

### ⛔ 生產環境

```python
# ❌ 絕不要在生產環境使用
VERIFY_SSL = False  # 安全風險！

# ✅ 生產環境必須使用
VERIFY_SSL = True
# 使用有效的 SSL 證書
```

## 💡 提示

1. **HTTPS 比 HTTP 慢** - 增加超時時間
2. **自簽名證書** - 開發環境中禁用驗證
3. **端口差異** - 確認 API 端口設置
4. **環境隔離** - 使用不同的 URL 測試不同環境

---

**最後更新**: 2025-11-11  
**版本**: 1.0.0

需要幫助？請查看 [API-SETUP-GUIDE.md](API-SETUP-GUIDE.md) 獲取更多信息。


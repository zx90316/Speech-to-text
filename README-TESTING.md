# 測試文件說明

本專案包含完整的 pytest 測試套件，涵蓋所有核心模組和功能。

## 📋 目錄

- [安裝測試環境](#安裝測試環境)
- [運行測試](#運行測試)
- [測試結構](#測試結構)
- [測試覆蓋率](#測試覆蓋率)
- [測試標記](#測試標記)
- [撰寫新測試](#撰寫新測試)
- [持續整合](#持續整合)

## 🚀 安裝測試環境

### 1. 安裝測試依賴

```bash
# 安裝測試需求
pip install -r requirements-test.txt

# 或者只安裝主要的測試工具
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

### 2. 設置環境變數

測試會自動設置必要的環境變數，但你也可以創建 `.env.test` 文件：

```bash
# .env.test
ENCRYPTION_KEY=test_encryption_key_32_bytes_long_123456789012=
EMAIL_HASH_SALT=test_salt_for_testing
ADMIN_TOKEN=test_admin_token_for_testing
SMTP_SERVER=smtp.test.com
SMTP_PORT=587
SMTP_USERNAME=test@test.com
SMTP_PASSWORD=test_password
FROM_EMAIL=test@test.com
```

## 🧪 運行測試

### 基本測試命令

```bash
# 運行所有測試
pytest

# 運行並顯示詳細輸出
pytest -v

# 運行特定測試文件
pytest tests/test_crypto_utils.py

# 運行特定測試函數
pytest tests/test_crypto_utils.py::TestCryptoUtils::test_hash_password
```

### 測試覆蓋率

```bash
# 運行測試並生成覆蓋率報告
pytest --cov=remote_server --cov-report=html --cov-report=term

# 查看 HTML 覆蓋率報告
# 報告位於: htmlcov/index.html
```

### 並行測試

```bash
# 使用多個 CPU 核心並行運行測試
pytest -n auto

# 使用 4 個進程
pytest -n 4
```

### 標記測試

```bash
# 只運行單元測試
pytest -m unit

# 只運行整合測試
pytest -m integration

# 跳過耗時測試
pytest -m "not slow"

# 只運行安全測試
pytest -m security

# 只運行 API 測試
pytest -m api
```

## 📁 測試結構

```
tests/
├── __init__.py                    # 測試套件初始化
├── conftest.py                    # pytest 配置和共用 fixtures
├── test_crypto_utils.py          # 加密工具測試
├── test_input_validator.py       # 輸入驗證測試
├── test_rate_limiter.py          # 速率限制測試
├── test_security_logger.py       # 安全日誌測試
├── test_memory_storage.py        # 記憶體存儲測試
├── test_email_service.py         # 郵件服務測試
└── test_api.py                   # API 端點測試
```

## 📊 測試覆蓋率

### 當前覆蓋率目標

- **總體覆蓋率**: ≥ 80%
- **核心模組**: ≥ 90%
- **安全模組**: 100%

### 查看覆蓋率報告

```bash
# 生成 HTML 報告
pytest --cov=remote_server --cov-report=html

# 生成終端機報告
pytest --cov=remote_server --cov-report=term-missing

# 生成 XML 報告（用於 CI/CD）
pytest --cov=remote_server --cov-report=xml
```

### 覆蓋率報告位置

- HTML 報告: `htmlcov/index.html`
- XML 報告: `coverage.xml`
- 終端機報告: 直接顯示在命令列

## 🏷️ 測試標記

專案使用以下 pytest 標記來分類測試：

| 標記 | 說明 | 使用場景 |
|------|------|---------|
| `unit` | 單元測試 | 測試單一函數或類的行為 |
| `integration` | 整合測試 | 測試多個組件的協作 |
| `slow` | 耗時測試 | 執行時間 > 1 秒的測試 |
| `security` | 安全測試 | 測試安全相關功能 |
| `api` | API 測試 | 測試 REST API 端點 |

### 使用標記

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    assert 1 + 1 == 2

@pytest.mark.slow
@pytest.mark.integration
def test_complex_workflow():
    # 複雜的測試...
    pass
```

## ✍️ 撰寫新測試

### 測試命名規範

- 測試文件: `test_*.py`
- 測試類: `Test*`
- 測試函數: `test_*`

### 測試結構

```python
"""
測試模組說明
"""
import pytest
from pathlib import Path

# 導入要測試的模組
from your_module import YourClass


class TestYourClass:
    """測試 YourClass 類"""

    def test_basic_functionality(self):
        """測試基本功能"""
        obj = YourClass()
        result = obj.method()
        assert result == expected_value

    def test_edge_case(self):
        """測試邊界情況"""
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)

    @pytest.mark.slow
    def test_performance(self):
        """測試性能"""
        # 性能測試...
        pass
```

### 使用 Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """創建測試數據"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """使用 fixture 的測試"""
    assert sample_data["key"] == "value"
```

### Mock 和 Patch

```python
from unittest.mock import patch, MagicMock

@patch('module.function')
def test_with_mock(mock_function):
    """使用 mock 的測試"""
    mock_function.return_value = "mocked value"
    result = call_function_that_uses_function()
    assert result == "mocked value"
```

### 異步測試

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """測試異步函數"""
    result = await async_function()
    assert result is not None
```

## 🔧 測試配置

### pytest.ini

主要配置項：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: 單元測試
    integration: 整合測試
    slow: 執行時間較長的測試
    security: 安全相關測試
    api: API 端點測試
```

### conftest.py

共用的 fixtures 和配置：

- `temp_dir`: 臨時目錄
- `sample_audio_file`: 測試音訊文件
- `valid_email`: 有效的測試郵箱
- `mock_smtp_server`: 模擬 SMTP 服務器
- 更多 fixtures 請查看 `tests/conftest.py`

## 🔄 持續整合

### GitHub Actions 範例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        pytest --cov=remote_server --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 🐛 除錯測試

### 顯示詳細輸出

```bash
# 顯示 print 輸出
pytest -s

# 顯示詳細的失敗信息
pytest -vv

# 在第一個失敗時停止
pytest -x

# 進入 pdb 除錯器
pytest --pdb
```

### 只運行失敗的測試

```bash
# 記錄失敗的測試
pytest --lf

# 先運行失敗的測試，然後運行其他測試
pytest --ff
```

## 📈 測試報告

### HTML 報告

```bash
# 生成 HTML 測試報告
pytest --html=report.html --self-contained-html
```

### JUnit XML 報告

```bash
# 生成 JUnit XML 報告（用於 CI/CD）
pytest --junitxml=junit.xml
```

## 🔒 安全測試

專案包含專門的安全測試，涵蓋：

- 加密和雜湊函數
- 輸入驗證和清理
- SQL 注入防護
- XSS 防護
- CSRF 防護
- 速率限制
- 會話管理

運行安全測試：

```bash
pytest -m security
```

## 📝 測試檢查清單

撰寫新功能時，確保：

- [ ] 撰寫單元測試（覆蓋率 ≥ 90%）
- [ ] 撰寫整合測試（如果需要）
- [ ] 測試正常情況
- [ ] 測試邊界情況
- [ ] 測試錯誤處理
- [ ] 測試安全性（如適用）
- [ ] 更新文檔
- [ ] 所有測試通過
- [ ] 覆蓋率達標

## 🤝 貢獻

撰寫測試時請遵循：

1. **單一責任**: 每個測試只測試一個功能點
2. **獨立性**: 測試之間不應相互依賴
3. **可重複性**: 測試結果應該一致
4. **清晰命名**: 測試名稱應該描述測試內容
5. **文檔化**: 使用 docstring 說明測試目的

## 📚 參考資源

- [pytest 官方文檔](https://docs.pytest.org/)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio 文檔](https://pytest-asyncio.readthedocs.io/)
- [Python Mock 文檔](https://docs.python.org/3/library/unittest.mock.html)

## ❓ 常見問題

### Q: 測試執行很慢，如何加速？

A: 使用並行測試：`pytest -n auto`

### Q: 如何跳過某些測試？

A: 使用 `@pytest.mark.skip` 或 `@pytest.mark.skipif`

### Q: 如何測試異步代碼？

A: 使用 `@pytest.mark.asyncio` 標記和 `pytest-asyncio` 套件

### Q: 測試覆蓋率不足怎麼辦？

A: 查看覆蓋率報告，針對未覆蓋的代碼補充測試

### Q: 如何測試需要外部服務的代碼？

A: 使用 Mock 或建立測試用的 stub 服務

## 📧 聯絡

如有測試相關問題，請：

1. 查看本文檔
2. 查看測試代碼範例
3. 提交 Issue

---

**最後更新**: 2025-01-11
**版本**: 1.0.0


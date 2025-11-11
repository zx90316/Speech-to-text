# VSCode/Cursor 測試功能使用指南

本指南說明如何在 VSCode 或 Cursor 中使用內建的測試功能來運行和除錯 pytest 測試。

## 📋 目錄

- [安裝和配置](#安裝和配置)
- [測試瀏覽器使用](#測試瀏覽器使用)
- [運行測試](#運行測試)
- [除錯測試](#除錯測試)
- [查看覆蓋率](#查看覆蓋率)
- [快捷鍵](#快捷鍵)
- [常見問題](#常見問題)

## 🚀 安裝和配置

### 1. 安裝必要的擴充功能

在 VSCode/Cursor 中安裝以下擴充功能：

**必裝：**
- **Python** (ms-python.python) - Python 語言支援
- **Pylance** (ms-python.vscode-pylance) - Python 語言伺服器

**建議安裝：**
- **Coverage Gutters** (ryanluker.vscode-coverage-gutters) - 顯示測試覆蓋率
- **Python Test Explorer** (littlefoxteam.vscode-python-test-adapter) - 測試瀏覽器
- **Test Explorer UI** (hbenl.vscode-test-explorer) - 測試介面

### 2. 自動配置

專案已包含以下配置文件：
- `.vscode/settings.json` - VSCode 設定
- `.vscode/launch.json` - 除錯配置
- `.vscode/tasks.json` - 任務配置
- `.vscode/extensions.json` - 推薦擴充功能

VSCode 會自動讀取這些配置。

### 3. 手動啟用測試功能

如果測試功能未自動啟用：

1. 打開命令面板：`Ctrl+Shift+P` (Windows/Linux) 或 `Cmd+Shift+P` (Mac)
2. 輸入：`Python: Configure Tests`
3. 選擇：`pytest`
4. 選擇測試目錄：`tests`

## 🧪 測試瀏覽器使用

### 打開測試瀏覽器

**方法 1：側邊欄**
- 點擊左側活動欄的「測試」圖標（燒杯圖標）🧪

**方法 2：命令面板**
- `Ctrl+Shift+P` → `Testing: Focus on Test Explorer View`

### 測試瀏覽器功能

測試瀏覽器會顯示所有測試的樹狀結構：

```
tests/
├── test_crypto_utils.py
│   ├── TestCryptoUtils
│   │   ├── test_singleton_pattern ✓
│   │   ├── test_hash_password ✓
│   │   └── ...
├── test_input_validator.py
│   └── ...
└── ...
```

**圖標說明：**
- ✓ 綠色勾：測試通過
- ✗ 紅色叉：測試失敗
- ○ 灰色圓：未運行
- ⟳ 轉動：測試運行中

## ▶️ 運行測試

### 在測試瀏覽器中運行

1. **運行所有測試**
   - 點擊頂部的 ▶️ 圖標

2. **運行特定測試文件**
   - 點擊文件旁的 ▶️ 圖標

3. **運行特定測試類**
   - 展開文件，點擊類旁的 ▶️ 圖標

4. **運行單一測試**
   - 展開到測試函數，點擊旁邊的 ▶️ 圖標

### 在編輯器中運行

當您打開測試文件時，每個測試函數上方會出現 **Run Test** 和 **Debug Test** 連結：

```python
# Run Test | Debug Test
def test_example():
    assert True
```

點擊即可運行或除錯該測試。

### 使用快捷鍵運行

- `Ctrl+; A` - 運行所有測試（Windows/Linux）
- `Cmd+; A` - 運行所有測試（Mac）

### 使用命令面板

1. `Ctrl+Shift+P` → `Testing: Run All Tests`
2. 或選擇其他運行選項：
   - `Testing: Run Test at Cursor` - 運行游標處的測試
   - `Testing: Rerun Failed Tests` - 重新運行失敗的測試
   - `Testing: Rerun Last Run` - 重新運行上次的測試

### 使用任務運行

1. `Ctrl+Shift+P` → `Tasks: Run Task`
2. 選擇任務：
   - **運行所有測試**
   - **運行測試（包含覆蓋率）**
   - **運行單元測試**
   - **運行安全測試**
   - **運行快速測試**

## 🐛 除錯測試

### 方法 1：測試瀏覽器

1. 在測試瀏覽器中，右鍵點擊測試
2. 選擇「Debug Test」

### 方法 2：編輯器中的連結

點擊測試函數上方的 **Debug Test** 連結

### 方法 3：使用除錯配置

1. 打開測試文件
2. 按 `F5` 或點擊左側的「運行和除錯」圖標
3. 選擇除錯配置：
   - **Python: 當前測試檔案** - 除錯當前文件的所有測試
   - **Python: 除錯當前測試函數** - 除錯游標處的測試
   - **Python: 所有測試** - 除錯所有測試
   - **Python: 安全測試** - 除錯安全相關測試

### 設置斷點

1. 在代碼行號左側點擊，出現紅點
2. 運行除錯模式，程序會在斷點處暫停

### 除錯控制

- `F5` - 繼續
- `F10` - 單步跳過
- `F11` - 單步進入
- `Shift+F11` - 單步跳出
- `Shift+F5` - 停止除錯

## 📊 查看覆蓋率

### 安裝 Coverage Gutters 擴充功能

確保已安裝 `Coverage Gutters` 擴充功能。

### 生成覆蓋率報告

**方法 1：使用任務**
1. `Ctrl+Shift+P` → `Tasks: Run Task`
2. 選擇「運行測試（包含覆蓋率）」

**方法 2：使用除錯配置**
1. 選擇「Python: 測試（包含覆蓋率）」配置
2. 按 `F5` 運行

**方法 3：使用命令列**
```bash
pytest --cov=remote_server --cov-report=xml --cov-report=html
```

### 在編輯器中顯示覆蓋率

1. 生成覆蓋率報告後
2. `Ctrl+Shift+P` → `Coverage Gutters: Display Coverage`
3. 編輯器左側會顯示覆蓋率：
   - 🟢 綠色：代碼已覆蓋
   - 🔴 紅色：代碼未覆蓋
   - 🟡 黃色：部分覆蓋

### 查看 HTML 覆蓋率報告

**方法 1：使用任務**
1. `Ctrl+Shift+P` → `Tasks: Run Task`
2. 選擇「打開覆蓋率報告」

**方法 2：手動打開**
- 打開 `htmlcov/index.html` 文件

### 覆蓋率狀態欄

Coverage Gutters 會在狀態欄顯示：
- `Coverage: XX%` - 當前文件覆蓋率
- 點擊可切換顯示/隱藏覆蓋率

## ⌨️ 快捷鍵

### 測試相關

| 功能 | Windows/Linux | Mac |
|------|--------------|-----|
| 運行所有測試 | `Ctrl+; A` | `Cmd+; A` |
| 運行游標處測試 | `Ctrl+; C` | `Cmd+; C` |
| 除錯游標處測試 | `Ctrl+; Ctrl+C` | `Cmd+; Cmd+C` |
| 顯示測試輸出 | `Ctrl+; O` | `Cmd+; O` |
| 開啟測試瀏覽器 | 無預設 | 無預設 |

### 除錯相關

| 功能 | 快捷鍵 |
|------|--------|
| 開始除錯 | `F5` |
| 單步跳過 | `F10` |
| 單步進入 | `F11` |
| 單步跳出 | `Shift+F11` |
| 繼續 | `F5` |
| 停止 | `Shift+F5` |
| 重新啟動 | `Ctrl+Shift+F5` |

### 自訂快捷鍵

1. `Ctrl+K Ctrl+S` 打開鍵盤快捷鍵設定
2. 搜尋「Testing」
3. 設定您喜歡的快捷鍵

## 🎨 測試輸出面板

### 查看測試輸出

1. `Ctrl+Shift+P` → `Testing: Show Output`
2. 或點擊測試瀏覽器頂部的「...」→「Show Output」

### 輸出內容

- 測試運行狀態
- 失敗的測試詳情
- print 輸出
- 錯誤堆疊追蹤

## 🔧 進階功能

### 自動運行測試

在 `.vscode/settings.json` 中已配置：
```json
{
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

保存文件時會自動重新發現測試。

### 測試過濾

在測試瀏覽器中：
1. 點擊頂部的篩選圖標 🔍
2. 選擇：
   - Show All Tests - 顯示所有
   - Show Failed Tests - 只顯示失敗
   - Show Passed Tests - 只顯示通過

### 測試標記篩選

運行特定標記的測試：
1. 使用任務：
   - 「運行單元測試」（`-m unit`）
   - 「運行安全測試」（`-m security`）
   - 「運行快速測試」（`-m "not slow"`）

## 📝 測試文件範本

### 快速創建測試

使用程式碼片段（Snippets）：

1. 在測試文件中輸入 `test` 並按 `Tab`
2. 或輸入 `testclass` 創建測試類

您也可以在 `.vscode/python.code-snippets` 中自訂片段。

## 🔍 常見問題

### Q1: 測試不顯示在測試瀏覽器中

**解決方法：**
1. 確認已安裝 pytest：`pip install pytest`
2. 重新載入視窗：`Ctrl+Shift+P` → `Reload Window`
3. 手動配置：`Ctrl+Shift+P` → `Python: Configure Tests`
4. 檢查 `.vscode/settings.json` 中的 `python.testing.pytestEnabled` 是否為 `true`

### Q2: 測試運行失敗，顯示模組找不到

**解決方法：**
1. 確認 PYTHONPATH 已設定（在 `.vscode/settings.json` 中）
2. 在終端機中測試：
   ```bash
   python -m pytest tests/test_crypto_utils.py -v
   ```
3. 確認已安裝測試依賴：
   ```bash
   pip install -r requirements-test.txt
   ```

### Q3: 覆蓋率不顯示

**解決方法：**
1. 確認已安裝 Coverage Gutters 擴充功能
2. 生成覆蓋率報告：
   ```bash
   pytest --cov=remote_server --cov-report=xml
   ```
3. 手動啟用：`Ctrl+Shift+P` → `Coverage Gutters: Display Coverage`
4. 檢查 `coverage.xml` 文件是否存在

### Q4: 除錯時無法進入外部模組

**解決方法：**
在 `.vscode/launch.json` 中設定：
```json
{
    "justMyCode": false
}
```

### Q5: 測試運行很慢

**解決方法：**
1. 使用快速測試任務（跳過 slow 標記）
2. 只運行需要的測試
3. 使用測試過濾功能
4. 考慮使用並行測試：
   ```bash
   pytest -n auto
   ```

## 💡 提示和技巧

### 1. 快速導航到測試

- `Ctrl+P` 輸入 `test_` 快速打開測試文件
- `Ctrl+Shift+O` 在當前文件中搜尋測試函數

### 2. 查看測試歷史

測試瀏覽器會保存上次運行結果，即使重新載入也會保留。

### 3. 使用監視模式

安裝 `pytest-watch`：
```bash
pip install pytest-watch
```

在終端機運行：
```bash
ptw tests/
```

### 4. 整合 Git

- 只測試修改的文件
- 在提交前運行測試
- 使用 Git Lens 查看測試覆蓋率變化

### 5. 團隊協作

- 共享 `.vscode` 配置
- 使用相同的擴充功能
- 統一測試運行標準

## 📚 延伸閱讀

- [VSCode Python Testing 官方文檔](https://code.visualstudio.com/docs/python/testing)
- [pytest 文檔](https://docs.pytest.org/)
- [Coverage.py 文檔](https://coverage.readthedocs.io/)

## 🎬 快速開始檢查清單

- [ ] 安裝 Python 和 Pylance 擴充功能
- [ ] 安裝測試依賴：`pip install -r requirements-test.txt`
- [ ] 打開測試瀏覽器（點擊側邊欄的測試圖標）
- [ ] 運行所有測試（點擊頂部的 ▶️）
- [ ] 查看測試結果
- [ ] 嘗試除錯單一測試
- [ ] 生成並查看覆蓋率報告
- [ ] 設定您喜歡的快捷鍵

---

**祝測試愉快！** 🎉

有任何問題，請參考 README-TESTING.md 或專案文檔。


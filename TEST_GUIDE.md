# 測試指南 - 前端與後端整合測試

本指南幫助您測試完整的語音轉文字系統。

## 🧪 測試前準備

### 1. 啟動後端服務

```bash
cd remote_server
# Windows
start_api.bat
# Linux/Mac
./start_api.sh
```

**確認後端運行：**
- 訪問 http://localhost:8000
- 應該看到: `{"service": "Whisper 語音轉文字 API", "version": "2.0.0", "status": "running"}`

### 2. 啟動前端服務

```bash
cd frontend
# Windows
START.bat
# Linux/Mac
npm run dev
```

**確認前端運行：**
- 訪問 http://localhost:5173
- 應該看到現代化的深色主題界面

## 🔍 功能測試清單

### ✅ 測試 1: 基本上傳和轉錄

**步驟：**
1. 打開前端界面 http://localhost:5173
2. 準備一個測試音訊檔案（MP3/WAV）
3. 拖放或點擊上傳音訊檔案
4. 取消勾選「語者分離」（加快測試速度）
5. 點擊「開始轉錄」

**預期結果：**
- ✓ 成功提交任務並獲得任務 ID
- ✓ 顯示任務進度面板
- ✓ 進度條從 0% 開始增長
- ✓ 顯示當前處理階段
- ✓ 最終達到 100% 完成

**Chrome DevTools 檢查：**
1. 打開開發者工具（F12）
2. Network 標籤：
   - 查看 `POST /api/tasks` 請求（上傳）
   - 查看 `GET /api/tasks/{id}/stream` SSE 連接
3. Console 標籤：
   - 應無錯誤訊息

---

### ✅ 測試 2: SSE 即時進度推送

**步驟：**
1. 提交一個較長的音訊檔案（> 1 分鐘）
2. 觀察進度面板的更新

**預期結果：**
- ✓ 進度條平滑增長
- ✓ 階段文字即時更新
- ✓ 部分轉錄結果逐步顯示（如果可用）
- ✓ 佇列位置正確顯示

**Chrome DevTools 檢查：**
1. Network 標籤 → EventStream：
   - 持續接收 `data:` 消息
   - 消息格式為 JSON
   - 包含 `status`, `progress`, `current_stage`

---

### ✅ 測試 3: 語者分離功能

**步驟：**
1. 上傳一個多人對話的音訊檔案
2. 勾選「啟用語者分離」
3. 提交並等待完成

**預期結果：**
- ✓ 進度顯示「執行語者分離」階段
- ✓ 部分結果包含 `speaker` 欄位（如 SPEAKER_00）
- ✓ 下載結果包含語者標記

---

### ✅ 測試 4: 任務取消

**步驟：**
1. 提交一個任務
2. 在處理過程中點擊「取消任務」
3. 確認取消

**預期結果：**
- ✓ 任務狀態變為「已取消」
- ✓ 進度停止更新
- ✓ SSE 連接關閉

**Chrome DevTools 檢查：**
- Network: `DELETE /api/tasks/{id}` 請求成功（200）

---

### ✅ 測試 5: 結果下載

**步驟：**
1. 等待一個任務完成
2. 點擊「下載結果」按鈕
3. 點擊「下載原始 ASR」按鈕

**預期結果：**
- ✓ 成功下載兩個 TXT 文件
- ✓ 檔案包含轉錄文字
- ✓ 語者分離結果包含語者標記

**Chrome DevTools 檢查：**
- Network: `GET /api/tasks/{id}/download` 請求成功
- Response Type: `text/plain`

---

### ✅ 測試 6: 任務歷史

**步驟：**
1. 提交 2-3 個任務
2. 查看右側的任務歷史面板
3. 點擊歷史任務項目

**預期結果：**
- ✓ 顯示所有提交過的任務
- ✓ 顯示正確的狀態（完成/處理中/失敗）
- ✓ 點擊任務可查看詳情
- ✓ 已完成任務顯示下載按鈕

**Chrome DevTools 檢查：**
- Network: `GET /api/my-tasks` 請求成功
- Response 包含任務陣列

---

### ✅ 測試 7: 服務統計

**步驟：**
1. 觀察頂部的服務統計區域
2. 提交任務時觀察變化

**預期結果：**
- ✓ 「正在處理」數量正確
- ✓ 「排隊中」數量正確
- ✓ 服務狀態正確（處理中/閒置）
- ✓ 每 5 秒自動更新

---

### ✅ 測試 8: 多任務佇列

**步驟：**
1. 快速提交 3 個任務
2. 觀察佇列處理

**預期結果：**
- ✓ 任務按順序處理
- ✓ 每個任務顯示正確的佇列位置
- ✓ 第一個任務開始處理，其他等待
- ✓ 任務依次完成

---

### ✅ 測試 9: 錯誤處理

**步驟：**
1. 上傳不支援的檔案格式（如 .txt）
2. 停止後端服務
3. 嘗試上傳檔案

**預期結果：**
- ✓ 顯示錯誤提示
- ✓ 不會崩潰
- ✓ 可以重試

---

### ✅ 測試 10: 響應式設計

**步驟：**
1. 在 Chrome DevTools 中開啟設備模擬器
2. 測試不同螢幕尺寸：
   - 桌面（1920x1080）
   - 平板（768x1024）
   - 手機（375x667）

**預期結果：**
- ✓ 布局自適應
- ✓ 按鈕和文字可讀
- ✓ 所有功能正常

---

## 🎨 Chrome DevTools 深度測試

### Performance 測試

1. 打開 Performance 標籤
2. 開始錄製
3. 完成一次上傳和轉錄流程
4. 停止錄製

**檢查項目：**
- ✓ 無長時間卡頓
- ✓ FPS 保持流暢
- ✓ 無記憶體洩漏

### Network 測試

1. 打開 Network 標籤
2. 執行完整流程

**檢查項目：**
- ✓ 所有請求成功（綠色）
- ✓ SSE 連接持續開啟
- ✓ 檔案上傳大小正確
- ✓ 下載響應正確

### Console 測試

**檢查項目：**
- ✓ 無錯誤訊息
- ✓ 無警告訊息
- ✓ API 調用日誌清晰

### Application 測試

1. 打開 Application 標籤
2. 查看 Storage

**檢查項目：**
- ✓ SessionStorage/LocalStorage 使用正常（如有）
- ✓ 無敏感資訊洩漏

---

## 📊 API 直接測試

### 使用 curl 測試

```bash
# 健康檢查
curl http://localhost:8000/health

# 服務統計
curl http://localhost:8000/api/stats

# 上傳任務
curl -X POST "http://localhost:8000/api/tasks?enable_diarization=false" \
  -F "file=@test.mp3"

# 查詢狀態（替換 TASK_ID）
curl http://localhost:8000/api/tasks/TASK_ID

# SSE 串流（替換 TASK_ID）
curl -N http://localhost:8000/api/tasks/TASK_ID/stream

# 查看歷史
curl http://localhost:8000/api/my-tasks
```

### 使用 Swagger UI

訪問 http://localhost:8000/docs

1. 展開每個 API 端點
2. 點擊 "Try it out"
3. 填寫參數
4. 執行測試

---

## ✅ 測試完成檢查清單

- [ ] 基本上傳和轉錄
- [ ] SSE 即時進度推送
- [ ] 語者分離功能
- [ ] 任務取消
- [ ] 結果下載
- [ ] 任務歷史
- [ ] 服務統計
- [ ] 多任務佇列
- [ ] 錯誤處理
- [ ] 響應式設計
- [ ] Performance 正常
- [ ] Network 請求正常
- [ ] Console 無錯誤
- [ ] API 直接調用正常

---

## 🐛 常見測試問題

### 前端無法連接後端

**檢查：**
1. 後端是否運行（http://localhost:8000）
2. 前端代理配置（vite.config.ts）
3. CORS 設定

**解決：**
```bash
# 重啟後端
cd remote_server
python api.py

# 重啟前端
cd frontend
npm run dev
```

### SSE 連接斷開

**檢查：**
1. 瀏覽器 Console 是否有錯誤
2. Network 標籤的 EventStream
3. 後端是否正常運行

### 進度不更新

**檢查：**
1. 任務是否真的在處理（查看後端日誌）
2. SSE 連接是否正常
3. 資料庫更新是否正常

---

## 📝 測試報告範本

```markdown
## 測試日期：2025-10-01

### 環境
- OS: Windows 10
- Python: 3.10
- Node.js: 18.0
- Browser: Chrome 118

### 測試結果
- [ ] 所有基本功能正常
- [ ] SSE 即時更新正常
- [ ] 語者分離功能正常
- [ ] 多任務處理正常
- [ ] 響應式設計正常

### 發現問題
1. 無

### 建議改進
1. 可以添加音訊預覽功能
2. 可以支援更多檔案格式
```

---

**開始測試吧！** 🧪

有任何問題請查看後端日誌或前端 Console。


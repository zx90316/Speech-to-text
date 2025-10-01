# Whisper 語音轉文字完整系統 - 快速開始指南

完整的 Whisper 語音轉文字系統，包含後端 API 和現代化前端界面。

## 📋 系統架構

```
專案根目錄/
├── remote_server/          # 後端 API 服務
│   ├── api.py             # FastAPI 主應用
│   ├── database.py        # SQLite 資料庫管理
│   ├── task_processor.py  # 任務處理邏輯
│   └── ...
└── frontend/              # 前端 Web 應用
    ├── src/              # React 源代碼
    └── ...
```

## 🚀 快速啟動（10 分鐘開始使用）

### 步驟 1: 環境準備

**系統需求：**
- Python 3.8+
- Node.js 16+
- CUDA（推薦，用於 GPU 加速）
- FFmpeg

### 步驟 2: 設定後端

```bash
# 1. 進入後端目錄
cd remote_server

# 2. 創建 .env 檔案
# 複製以下內容並填入您的 Hugging Face Token
echo HUGGINGFACE_TOKEN=your_token_here > .env

# 3. 啟動後端服務（會自動安裝依賴）
# Windows:
start_api.bat

# Linux/Mac:
chmod +x start_api.sh
./start_api.sh
```

**獲取 Hugging Face Token：**
1. 訪問 https://huggingface.co/settings/tokens
2. 創建新 Token（Read 權限即可）
3. 同意 pyannote/speaker-diarization-community-1 模型使用條款

### 步驟 3: 啟動前端

```bash
# 開啟新的終端視窗

# 1. 進入前端目錄
cd frontend

# 2. 安裝依賴並啟動（首次會自動安裝）
# Windows:
START.bat

# Linux/Mac:
npm install
npm run dev
```

### 步驟 4: 開始使用

1. 打開瀏覽器訪問: **http://localhost:5173**
2. 上傳音訊檔案（支援 MP3, WAV, M4A, FLAC）
3. 選擇是否啟用語者分離
4. 點擊「開始轉錄」
5. 即時查看處理進度
6. 完成後下載轉錄結果

## 📊 功能展示

### 後端 API 功能

- ✅ **多格式支援**: MP3, WAV, M4A, FLAC
- ✅ **語者分離**: 自動識別不同說話者
- ✅ **任務佇列**: 自動排隊處理多個任務
- ✅ **進度推送**: SSE 即時推送處理進度
- ✅ **任務管理**: 查詢、取消、下載
- ✅ **歷史記錄**: 根據 IP 查詢歷史任務
- ✅ **SQLite 存儲**: 持久化任務資訊

### 前端界面功能

- ✅ **拖放上傳**: 支援拖放或點擊上傳
- ✅ **即時進度**: 動態進度條和狀態顯示
- ✅ **部分結果**: 處理過程中顯示已轉錄內容
- ✅ **任務歷史**: 查看所有提交過的任務
- ✅ **一鍵下載**: 快速下載轉錄結果
- ✅ **服務監控**: 即時顯示佇列和處理狀態
- ✅ **現代化 UI**: 深色主題，流暢動畫
- ✅ **響應式設計**: 支援各種屏幕尺寸

## 🌐 訪問地址

| 服務 | 地址 | 說明 |
|------|------|------|
| 前端界面 | http://localhost:5173 | Web 用戶界面 |
| 後端 API | http://localhost:8000 | REST API 服務 |
| API 文檔 | http://localhost:8000/docs | Swagger UI 文檔 |
| 健康檢查 | http://localhost:8000/health | 服務健康狀態 |

## 📖 使用流程

### 方式一：使用前端界面（推薦）

1. **上傳檔案**
   - 拖放或點擊選擇音訊檔案
   - 選擇是否啟用語者分離
   - 點擊「開始轉錄」

2. **監控進度**
   - 查看即時進度條和處理階段
   - 觀察佇列位置
   - 查看部分轉錄結果

3. **獲取結果**
   - 下載最終轉錄結果（含語者資訊）
   - 下載原始 ASR 結果
   - 查看任務歷史記錄

### 方式二：直接調用 API

```bash
# 1. 提交任務
curl -X POST "http://localhost:8000/api/tasks?enable_diarization=true" \
  -F "file=@audio.mp3"

# 返回: {"task_id": "xxx", "status": "pending", ...}

# 2. 查詢狀態
curl "http://localhost:8000/api/tasks/{task_id}"

# 3. 即時進度（SSE）
curl "http://localhost:8000/api/tasks/{task_id}/stream"

# 4. 下載結果
curl -O "http://localhost:8000/api/tasks/{task_id}/download?file_type=transcript"

# 5. 查看歷史
curl "http://localhost:8000/api/my-tasks"
```

## 🔧 進階配置

### 調整後端設置

編輯 `remote_server/task_processor.py`：

```python
# 降低記憶體使用
compute_type="int8"  # 預設是 "float16"

# 使用不同的模型
model = WhisperModel("openai/whisper-large-v3", ...)
```

### 調整前端 API 地址

編輯 `frontend/vite.config.ts`：

```typescript
proxy: {
  '/api': {
    target: 'http://your-backend-address:8000',
    changeOrigin: true,
  }
}
```

## 📁 結果文件位置

```
remote_server/
├── result/                  # 轉錄結果
│   └── {task_id}/
│       ├── transcript_raw.txt              # 原始 ASR 結果
│       └── transcript_with_speakers.txt    # 含語者分離結果
├── uploads/                 # 上傳的音訊檔案
│   └── {task_id}/
│       └── original_audio.mp3
└── tasks.db                 # SQLite 資料庫
```

## 🎯 處理流程說明

### 後端處理階段

1. **初始化**（0-5%）
   - 載入 Whisper 模型
   - 載入語者分離模型（如啟用）

2. **音訊預處理**（20-25%）
   - 轉換為 WAV 格式
   - 重採樣至 16kHz

3. **語音轉文字**（30-60%）
   - 使用 Faster-Whisper 進行 ASR
   - 繁簡轉換

4. **語者分離**（70-85%，可選）
   - 識別不同說話者
   - 整合語者資訊

5. **完成**（100%）
   - 保存結果文件
   - 更新資料庫狀態

### 前端顯示

- 即時顯示當前處理階段
- 動態更新進度百分比
- 顯示佇列位置
- 展示部分轉錄結果

## 💡 使用技巧

### 1. 提高處理速度
- 使用 GPU（CUDA）可提升 10 倍速度
- 關閉語者分離可減少 30-40% 處理時間
- 使用較短的音訊片段

### 2. 獲得最佳效果
- 使用清晰的音訊源
- 推薦 16kHz 採樣率
- 避免背景噪音
- 啟用語者分離適合多人對話

### 3. 管理任務
- 可以同時提交多個任務（自動排隊）
- 隨時取消不需要的任務
- 查看歷史記錄追蹤所有任務

## 🐛 故障排除

### 後端問題

**模型下載失敗**
```bash
# 設定 Hugging Face 鏡像（中國用戶）
export HF_ENDPOINT=https://hf-mirror.com
```

**CUDA 記憶體不足**
```python
# 降低計算精度
compute_type="int8"
```

**FFmpeg 找不到**
- 確保 FFmpeg 在系統 PATH 中
- 或將 ffmpeg-7.1.1-full_build-shared 放在專案根目錄

### 前端問題

**無法連接 API**
- 確認後端運行在 http://localhost:8000
- 檢查防火牆設置
- 查看瀏覽器控制台錯誤

**上傳失敗**
- 檢查檔案格式（MP3, WAV, M4A, FLAC）
- 檢查檔案大小限制
- 查看後端日誌

**進度不更新**
- 確認瀏覽器支援 EventSource API
- 檢查 SSE 連接狀態
- 重新整理頁面

## 📚 更多資源

- **API 完整文檔**: http://localhost:8000/docs
- **後端 README**: remote_server/README.md
- **前端 README**: frontend/README.md
- **API 使用範例**: remote_server/example_usage.py

## 🙏 致謝

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [BELLE Whisper](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

## 📝 授權

本專案僅供學習和研究使用。

---

**開始使用吧！** 🎉

如有問題，請查看 API 文檔或檢查日誌輸出。


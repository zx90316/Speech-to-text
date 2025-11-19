# Whisper 語音轉文字 API 服務

[English](README_EN.md) | 繁體中文

基於 Faster-Whisper 和 Pyannote 的完整語音轉文字 API 服務，包含現代化 React 前端與音訊預處理功能。

## ✨ 功能特色

### 核心功能
- 🎯 **高品質轉錄**：使用 Belle-Whisper-Large-V3 中文模型
- 👥 **語者分離**：可選的多人對話識別，支援設定語者數量
- 📊 **即時進度**：使用 SSE (Server-Sent Events) 推送處理進度與部分結果
- 🎨 **信心度視覺化**：詞級信心分數與互動式 HTML 視覺化
- 📁 **任務管理**：完整的任務生命週期管理
- 🔍 **歷史查詢**：根據 IP 查詢過往提交的任務
- 💾 **結果儲存**：自動組織並儲存轉錄結果
- 🚀 **非同步處理**：後台任務佇列，不阻塞請求

### 音訊預處理
- 🎵 **降噪處理**：FFT 降噪演算法
- 🔊 **音量正規化**：峰值或 LUFS 正規化
- 🔇 **靜音移除**：可配置閾值的靜音檢測
- 🎤 **人聲增強**：語音頻段增強
- 🔔 **迴聲消除**：去除點擊聲與迴聲
- 🎛️ **頻率均衡**：3 頻段 EQ 調整
- 🎚️ **動態壓縮**：動態範圍壓縮
- ⚡ **速度調整**：保持音調的速度變化
- 🎹 **音調調整**：半音調整
- 🔄 **取樣率轉換**：支援多種取樣率

### 前端介面
- 💻 **現代化 UI**：React + TypeScript + Vite
- 📤 **拖放上傳**：支援拖放檔案上傳
- 🎬 **時間範圍選擇**：可選擇音訊片段進行轉錄
- 📊 **即時預覽**：處理過程中顯示部分結果
- 📜 **任務歷史**：批次管理與查詢
- 🔧 **進階參數**：VAD 敏感度、計算類型等
- 🎯 **智慧音訊播放器**：根據檔案大小自動選擇播放器
- 🛠️ **管理介面**：管理員儀表板與批次操作

## 🚀 快速開始

### 1. 環境需求

**後端：**
- Python 3.8+
- CUDA（推薦，用於 GPU 加速）
- FFmpeg（音訊處理）
- Hugging Face Token（用於語者分離模型）

**前端：**
- Node.js 16+
- npm 或 yarn

### 2. 後端設定

#### 安裝依賴

```bash
# 進入後端目錄
cd backend

# 創建虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

#### 設定環境變數

在 `backend/` 目錄下創建 `.env` 檔案：

```env
HUGGINGFACE_TOKEN=your_huggingface_token_here
ADMIN_TOKEN=your_admin_token_here
```

獲取 Hugging Face Token：前往 [Hugging Face Settings](https://huggingface.co/settings/tokens)

#### 啟動後端服務

```bash
cd backend
python api.py
```

後端 API 將在 `http://localhost:8000` 啟動，可訪問以下文檔：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. 前端設定

```bash
# 進入前端目錄
cd frontend

# 安裝依賴
npm install

# 啟動開發伺服器
npm run dev
```

前端將在 `http://localhost:5173` 啟動，自動代理 API 請求到後端。

#### 生產環境建置

```bash
cd frontend
npm run build
npm run preview
```

## 📖 API 使用說明

### 完整 API 文檔

啟動服務後訪問：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要端點

#### 1. 提交轉錄任務

```bash
POST /api/tasks
```

**基本參數：**
- `file`: 音訊檔案（支援 mp3, wav, m4a, flac）
- `enable_diarization`: 是否啟用語者分離（預設: true）
- `start_time`: 開始時間（秒，可選）
- `end_time`: 結束時間（秒，可選）
- `language`: 語言代碼（如 zh, en, ja，可選）
- `task`: 任務類型（transcribe 或 translate）
- `model`: Whisper 模型名稱
- `compute_type`: 計算類型（float32, int8, float16）

**進階參數：**
- `vad_onset`: VAD 語音檢測敏感度（0-1，預設 0.5）
- `vad_offset`: VAD 語音結束閾值（0-1，預設 0.363）
- `min_speakers`: 最小語者數（可選）
- `max_speakers`: 最大語者數（可選）
- `enable_confidence_score`: 啟用信心度分數（預設 false）

**範例：**
```bash
curl -X POST "http://localhost:8000/api/tasks?enable_diarization=true" \
  -F "file=@audio.mp3"
```

**回應：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "queue_position": 1,
  "message": "任務已提交，正在排隊處理"
}
```

#### 2. 查詢任務狀態

```bash
GET /api/tasks/{task_id}
```

**範例：**
```bash
curl "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000"
```

**回應：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "audio.mp3",
  "status": "processing",
  "progress": 45.5,
  "current_stage": "語音轉文字 (已處理 25 個片段)",
  "queue_position": 0,
  "enable_diarization": true,
  "created_at": "2025-10-01T10:30:00",
  "started_at": "2025-10-01T10:31:00",
  "completed_at": null,
  "error_message": null,
  "has_result": false
}
```

#### 3. 即時進度串流（SSE）

```bash
GET /api/tasks/{task_id}/stream
```

**範例（JavaScript）：**
```javascript
const eventSource = new EventSource(`http://localhost:8000/api/tasks/${taskId}/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('進度:', data.progress, '%');
  console.log('狀態:', data.current_stage);
  
  if (data.status === 'completed') {
    console.log('任務完成！');
    eventSource.close();
  }
};

eventSource.onerror = (error) => {
  console.error('連線錯誤:', error);
  eventSource.close();
};
```

**串流資料格式：**
```json
{
  "status": "processing",
  "progress": 45.5,
  "current_stage": "語音轉文字 (已處理 25 個片段)",
  "queue_position": 0,
  "error_message": null,
  "timestamp": "2025-10-01T10:32:15.123456",
  "partial_result": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "大家好",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

#### 4. 取消任務

```bash
DELETE /api/tasks/{task_id}
```

**範例：**
```bash
curl -X DELETE "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000"
```

#### 5. 下載轉錄結果

```bash
GET /api/tasks/{task_id}/download?file_type=transcript
```

**參數：**
- `file_type`:
  - `transcript`: 最終結果（含語者資訊）
  - `raw`: 原始 ASR 轉錄結果
  - `confidence_html`: 信心度視覺化 HTML（需啟用 enable_confidence_score）

**範例：**
```bash
# 下載最終結果
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=transcript"

# 下載原始 ASR 結果
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=raw"

# 下載信心度視覺化 HTML
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=confidence_html"
```

#### 6. 查詢我的任務歷史

```bash
GET /api/my-tasks?limit=50
```

**範例：**
```bash
curl "http://localhost:8000/api/my-tasks?limit=10"
```

**回應：**
```json
{
  "client_ip": "192.168.1.100",
  "total": 10,
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "audio1.mp3",
      "status": "completed",
      "progress": 100.0,
      "enable_diarization": true,
      "created_at": "2025-10-01T10:30:00",
      "completed_at": "2025-10-01T10:35:00",
      "has_result": true
    }
  ]
}
```

#### 7. 服務統計資訊

```bash
GET /api/stats
```

**回應：**
```json
{
  "queue_size": 3,
  "processing_count": 1,
  "is_processing": true,
  "total_waiting": 4
}
```

### 音訊預處理 API

#### 8. 提交預處理任務

```bash
POST /api/preprocess
```

**參數：**
- `file`: 音訊檔案
- `config`: JSON 格式的預處理配置

**配置範例：**
```json
{
  "enable_denoise": true,
  "denoise_strength": 0.5,
  "enable_normalize": true,
  "normalize_type": "peak",
  "target_level": -3.0,
  "enable_silence_removal": true,
  "silence_threshold": -50.0,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.5,
  "enable_mono": true,
  "enable_resample": true,
  "target_sample_rate": 16000
}
```

#### 9. 查詢預處理狀態

```bash
GET /api/preprocess/{preprocess_id}
```

#### 10. 下載預處理音訊

```bash
GET /api/preprocess/{preprocess_id}/download?file_type=processed
```

**參數：**
- `file_type`: `original` 或 `processed`

## 📁 目錄結構

```
Speech-to-text/
├── backend/              # 後端服務
│   ├── api.py                 # FastAPI 主應用
│   ├── database.py            # SQLite 資料庫管理
│   ├── task_processor.py      # 轉錄任務處理邏輯
│   ├── audio_preprocessor.py  # 音訊預處理引擎
│   ├── preprocess_processor.py # 預處理任務處理器
│   ├── requirements.txt       # Python 依賴
│   ├── .env                   # 環境變數（需自行創建）
│   ├── tasks.db               # SQLite 資料庫（自動創建）
│   ├── uploads/               # 上傳的音訊檔案
│   │   └── {task_id}/
│   ├── result/                # 轉錄結果
│   │   └── {task_id}/
│   │       ├── transcript_raw.txt
│   │       ├── transcript_with_speakers.txt
│   │       └── confidence_visualization.html
│   └── preprocessed/          # 預處理檔案
│       └── {preprocess_id}/
│           ├── original_*.mp3
│           └── processed_*.wav
│
├── frontend/                   # 前端應用
│   ├── src/
│   │   ├── ui/
│   │   │   └── App.tsx        # 主應用組件
│   │   ├── components/        # React 組件
│   │   ├── pages/             # 頁面組件
│   │   ├── api.ts             # API 客戶端
│   │   └── types.ts           # TypeScript 類型定義
│   ├── package.json
│   └── vite.config.ts
│
├── CLAUDE.md                   # Claude Code 指南
├── README.md                   # 本檔案
└── README_EN.md               # English README
```

## 🔄 任務狀態流程

```
pending → processing → completed
                    ↘ failed
                    ↘ canceled
```

- **pending**: 任務已提交，等待處理
- **processing**: 正在處理中
- **completed**: 處理完成
- **failed**: 處理失敗
- **canceled**: 已取消

## 🎯 處理階段

1. **載入模型**（0-5%）
2. **轉換音訊格式**（20-25%）
3. **語音轉文字 (ASR)**（30-60%）
4. **語者分離**（70-85%，如啟用）
5. **整合語者資訊**（85-95%，如啟用）
6. **處理完成**（100%）

## 💡 使用建議

### 轉錄相關
1. **首次使用**：第一次運行會自動下載模型，需要一些時間和儲存空間
2. **GPU 加速**：強烈建議使用 CUDA 加速，處理速度可提升 10 倍以上
3. **記憶體需求**：建議至少 8GB RAM，啟用語者分離需要更多記憶體
4. **計算類型選擇**：
   - `float32`：最高品質，記憶體需求大，適合 GPU 充足時使用
   - `int8`：平衡選項，記憶體需求中等，品質良好
   - `float16`：記憶體需求小，適合 VRAM 有限的 GPU
5. **並發處理**：目前為單任務處理，多個任務會排隊執行

### 音訊預處理
1. **音訊格式**：建議使用 16kHz 單聲道 WAV 以獲得最佳轉錄效果
2. **預處理順序**：建議先進行音訊預處理再進行轉錄，可提升識別準確度
3. **降噪強度**：從 0.3-0.5 開始調整，過高可能影響音質
4. **靜音移除**：閾值建議設為 -50dB 以上，避免誤刪正常語音
5. **A/B 對比**：使用前端介面可即時對比原始與處理後的音訊

### 前端使用
1. **大檔案處理**：系統會自動為大檔案選擇原生音訊播放器
2. **時間範圍選擇**：可在音訊播放器中選擇特定片段進行轉錄
3. **信心度視覺化**：啟用後可下載互動式 HTML，查看詞級信心分數

## 🐛 常見問題

### 模型下載失敗
```bash
# 設定 Hugging Face 鏡像（中國用戶）
export HF_ENDPOINT=https://hf-mirror.com
```

### FFmpeg 找不到
確保 FFmpeg 已正確安裝並加入系統 PATH，或將 `ffmpeg-7.1.1-full_build-shared` 資料夾放在專案根目錄

### CUDA 記憶體不足
在前端選擇較低的計算類型：
- 選擇 `int8` 或 `float16` 計算類型
- 關閉語者分離功能
- 使用較小的 Whisper 模型

### 前端無法連接後端
確認：
1. 後端服務已啟動在 `http://localhost:8000`
2. 前端開發伺服器使用 `npm run dev` 啟動
3. Vite 代理配置正確（已預設配置）

### 預處理後音質變差
調整參數：
- 降低降噪強度（denoise_strength）
- 調整靜音移除閾值（silence_threshold）
- 關閉不需要的濾鏡

## 🛠️ 技術架構

### 後端技術
- **FastAPI**: 現代化 Python Web 框架
- **Faster-Whisper**: 高效能 Whisper 推理引擎
- **Pyannote Audio**: 語者分離模型
- **FFmpeg**: 音訊處理工具
- **SQLite**: 任務資料庫
- **Server-Sent Events**: 即時進度推送

### 前端技術
- **React 18**: UI 框架
- **TypeScript**: 類型安全
- **Vite**: 快速建置工具
- **Axios**: HTTP 客戶端
- **Lucide React**: 圖示庫

### 主要模型
- **Whisper**: CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32
- **Diarization**: pyannote/speaker-diarization-community-1
- **Text Conversion**: OpenCC (s2twp)

## 📝 授權

本專案僅供學習和研究使用。

## 🙏 致謝

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [BELLE-2/Belle-whisper-large-v3-zh-punct](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)


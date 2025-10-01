# Whisper 語音轉文字 API 服務

基於 Faster-Whisper 和 Pyannote 的完整語音轉文字 API 服務，支援語者分離功能。

## ✨ 功能特色

- 🎯 **高品質轉錄**：使用 Belle-Whisper-Large-V3 中文模型
- 👥 **語者分離**：可選的多人對話識別
- 📊 **即時進度**：使用 SSE (Server-Sent Events) 推送處理進度
- 📁 **任務管理**：完整的任務生命週期管理
- 🔍 **歷史查詢**：根據 IP 查詢過往提交的任務
- 💾 **結果儲存**：自動組織並儲存轉錄結果
- 🚀 **非同步處理**：後台任務佇列，不阻塞請求

## 🚀 快速開始

### 1. 環境需求

- Python 3.8+
- CUDA（推薦，用於 GPU 加速）
- FFmpeg（音訊處理）
- Hugging Face Token（用於語者分離模型）

### 2. 安裝依賴

```bash
# Windows
start_api.bat

# Linux/Mac
chmod +x start_api.sh
./start_api.sh
```

或手動安裝：

```bash
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

### 3. 設定環境變數

創建 `.env` 檔案：

```env
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

獲取 Token：前往 [Hugging Face](https://huggingface.co/settings/tokens)

### 4. 啟動服務

```bash
python api.py
```

服務將在 `http://localhost:8000` 啟動

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

**參數：**
- `file`: 音訊檔案（支援 mp3, wav, m4a, flac）
- `enable_diarization`: 是否啟用語者分離（預設: true）

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

**範例：**
```bash
# 下載最終結果
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=transcript"

# 下載原始 ASR 結果
curl -O "http://localhost:8000/api/tasks/550e8400-e29b-41d4-a716-446655440000/download?file_type=raw"
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

## 📁 目錄結構

```
remote_server/
├── api.py                    # FastAPI 主應用
├── database.py               # SQLite 資料庫管理
├── task_processor.py         # 任務處理邏輯
├── requirements.txt          # Python 依賴
├── start_api.bat            # Windows 啟動腳本
├── start_api.sh             # Linux/Mac 啟動腳本
├── .env                     # 環境變數（需自行創建）
├── tasks.db                 # SQLite 資料庫（自動創建）
├── uploads/                 # 上傳的音訊檔案
│   └── {task_id}/
│       └── original_audio.mp3
└── result/                  # 轉錄結果
    └── {task_id}/
        ├── transcript_raw.txt              # 原始 ASR 結果
        └── transcript_with_speakers.txt    # 含語者分離結果
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

1. **首次使用**：第一次運行會自動下載模型，需要一些時間
2. **GPU 加速**：強烈建議使用 CUDA 加速，處理速度可提升 10 倍以上
3. **記憶體需求**：建議至少 8GB RAM，啟用語者分離需要更多
4. **音訊格式**：建議使用 16kHz 單聲道 WAV 以獲得最佳效果
5. **並發處理**：目前為單任務處理，多個任務會排隊執行

## 🐛 常見問題

### 模型下載失敗
```bash
# 設定 Hugging Face 鏡像（中國用戶）
export HF_ENDPOINT=https://hf-mirror.com
```

### FFmpeg 找不到
確保 FFmpeg 已正確安裝並加入系統 PATH，或將 `ffmpeg-7.1.1-full_build-shared` 資料夾放在專案根目錄

### CUDA 記憶體不足
```python
# 在 task_processor.py 中調整
compute_type="int8"  # 改為 int8 降低記憶體使用
```

## 📝 授權

本專案僅供學習和研究使用。

## 🙏 致謝

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [BELLE-2/Belle-whisper-large-v3-zh-punct](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct)


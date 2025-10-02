# 音訊預處理功能使用指南

## 🎯 功能概述

音訊預處理功能可在轉錄前優化音訊品質，提升語音辨識準確度。支援降噪、音量正規化、人聲增強等多種處理選項。

## 🚀 快速開始

### 方法一：使用前端介面

1. **啟動服務**
   ```bash
   # 後端
   cd Speech-to-text
   .venv/Scripts/python.exe remote_server/api_preprocess_only.py

   # 前端
   cd frontend
   npm run dev
   ```

2. **使用預處理**
   - 在主頁點擊「預處理」按鈕
   - 上傳音訊檔案
   - 調整預處理參數
   - 點擊「開始預處理」
   - 使用播放器對比原始與處理後音訊

### 方法二：使用 API

```python
import requests
import json

# 預處理配置
config = {
    "enable_denoise": True,           # 降噪
    "denoise_strength": 0.5,          # 降噪強度
    "enable_normalize": True,         # 音量正規化
    "normalize_type": "peak",         # 正規化類型
    "target_level": -3.0,             # 目標電平
    "enable_vocal_enhancement": True, # 人聲增強
    "enable_mono": True               # 轉單聲道
}

# 上傳檔案
with open('audio.mp3', 'rb') as f:
    files = {'file': ('audio.mp3', f, 'audio/mpeg')}
    params = {'config': json.dumps(config)}

    response = requests.post(
        'http://localhost:8000/api/preprocess',
        files=files,
        params=params
    )

    result = response.json()
    preprocess_id = result['preprocess_id']

    # 下載處理後的音訊
    download_url = f"http://localhost:8000/api/preprocess/{preprocess_id}/download?file_type=processed"
```

## 🎛️ 常用預處理方案

### 方案 1：通話錄音清理
適用於電話錄音、視訊會議等場景

```json
{
  "enable_denoise": true,
  "denoise_strength": 0.7,
  "enable_normalize": true,
  "normalize_type": "lufs",
  "target_level": -16.0,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.6,
  "enable_compression": true,
  "compression_ratio": 6.0,
  "enable_mono": true
}
```

### 方案 2：環境噪音去除
適用於街頭採訪、戶外錄音等

```json
{
  "enable_denoise": true,
  "denoise_strength": 0.8,
  "enable_eq": true,
  "eq_low_gain": -6,
  "eq_mid_gain": 3,
  "eq_high_gain": -3,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.7,
  "enable_mono": true
}
```

### 方案 3：音量不穩定修正
適用於音量忽大忽小的錄音

```json
{
  "enable_normalize": true,
  "normalize_type": "lufs",
  "target_level": -16.0,
  "enable_compression": true,
  "compression_ratio": 8.0,
  "compression_threshold": -20.0,
  "enable_silence_removal": true,
  "silence_threshold": -40.0,
  "min_silence_duration": 0.5
}
```

### 方案 4：會議錄音優化
適用於會議室錄音、演講等

```json
{
  "enable_denoise": true,
  "denoise_strength": 0.5,
  "enable_echo_removal": true,
  "enable_normalize": true,
  "normalize_type": "lufs",
  "target_level": -16.0,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.5,
  "enable_silence_removal": true,
  "silence_threshold": -45.0,
  "enable_mono": true
}
```

### 方案 5：低品質音訊救援
適用於老舊錄音、低品質音訊

```json
{
  "enable_denoise": true,
  "denoise_strength": 0.9,
  "enable_echo_removal": true,
  "enable_eq": true,
  "eq_low_gain": -8,
  "eq_mid_gain": 6,
  "eq_high_gain": -6,
  "enable_vocal_enhancement": true,
  "enhancement_strength": 0.8,
  "enable_compression": true,
  "compression_ratio": 10.0,
  "enable_normalize": true,
  "normalize_type": "lufs",
  "target_level": -14.0,
  "enable_mono": true
}
```

## 📊 參數調整建議

### 降噪強度 (denoise_strength)
- **0.3-0.5**：輕微降噪，保留更多細節
- **0.5-0.7**：中度降噪，平衡噪音與音質
- **0.7-1.0**：強力降噪，可能影響音質

### 音量正規化類型
- **peak**：快速，適合一般用途
- **lufs**：專業響度標準，適合廣播、視訊

### EQ 調整
- **低頻 (80-300Hz)**：控制背景噪音、轟鳴聲
- **中頻 (300-3000Hz)**：人聲主要頻段
- **高頻 (3000-8000Hz)**：清晰度、齒音

### 動態壓縮比
- **2:1 - 4:1**：溫和壓縮
- **4:1 - 8:1**：中度壓縮
- **8:1 - 20:1**：強力壓縮（可能不自然）

## 🎬 使用流程範例

### 完整預處理 + 轉錄工作流程

```python
import requests
import json

# 1. 預處理音訊
preprocess_config = {
    "enable_denoise": True,
    "denoise_strength": 0.6,
    "enable_normalize": True,
    "normalize_type": "lufs",
    "target_level": -16.0,
    "enable_vocal_enhancement": True,
    "enable_mono": True
}

with open('meeting.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/preprocess',
        files={'file': ('meeting.mp3', f)},
        params={'config': json.dumps(preprocess_config)}
    )

    preprocess_id = response.json()['preprocess_id']
    print(f"預處理完成，ID: {preprocess_id}")

# 2. 下載處理後的音訊
processed_audio = requests.get(
    f'http://localhost:8000/api/preprocess/{preprocess_id}/download',
    params={'file_type': 'processed'}
)

with open('meeting_processed.wav', 'wb') as f:
    f.write(processed_audio.content)

# 3. 使用處理後的音訊進行轉錄
# （需要完整的 Whisper API）
with open('meeting_processed.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/tasks',
        files={'file': ('meeting_processed.wav', f)},
        params={'enable_diarization': True}
    )

    task_id = response.json()['task_id']
    print(f"轉錄任務已提交，ID: {task_id}")
```

## 💡 最佳實踐

### 1. 先測試小片段
在處理長音訊前，先用前 30 秒測試不同預處理參數。

### 2. 逐步調整
不要一次開啟所有功能，逐步測試每個選項的效果。

### 3. A/B 對比
使用前端播放器仔細對比原始與處理後音訊，確保沒有過度處理。

### 4. 保存設定
記錄有效的預處理配置，建立自己的預設檔案庫。

### 5. 注意音質損失
過度的降噪或壓縮可能導致音質下降，影響辨識準確度。

## ⚠️ 注意事項

1. **處理時間**：長音訊處理可能需要數分鐘
2. **磁碟空間**：處理後的檔案會暫存在伺服器，記得清理
3. **音訊品質**：垃圾進垃圾出，預處理無法創造不存在的資訊
4. **參數平衡**：過度處理可能適得其反

## 🔗 相關文件

- [完整實作說明](PREPROCESS_IMPLEMENTATION.md)
- [API 文件](http://localhost:8000/docs)
- [TODO 功能清單](TODO.md)

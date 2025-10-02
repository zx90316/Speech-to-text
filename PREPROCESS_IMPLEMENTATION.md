# 音訊預處理功能實作總結

## 📋 實作完成項目

### 1. 後端核心功能 ✅

**檔案：** [remote_server/audio_preprocessor.py](remote_server/audio_preprocessor.py)

實作了完整的音訊預處理引擎，包含以下功能：

#### 基礎處理
- ✅ **降噪處理**：使用 FFT Denoising (afftdn) 演算法
  - 可調整降噪強度（0.0-1.0）
  - 自動套用高通濾波器去除低頻噪音

- ✅ **音量正規化**
  - 峰值歸一化 (Peak Normalization)
  - 響度歸一化 (LUFS Normalization)
  - 可自訂目標電平

- ✅ **靜音片段移除**
  - 可調整靜音閾值 (dB)
  - 可設定最小靜音持續時間
  - 支援開始和結束靜音移除

#### 增強功能
- ✅ **人聲增強**
  - 針對語音頻段 (300-3000Hz) 進行增強
  - 可調整增強強度

- ✅ **迴聲消除**
  - 使用 adeclick 去除點擊聲
  - 使用 adeclip 去除削波失真

- ✅ **頻率均衡器 (EQ)**
  - 低頻控制 (80-300Hz)
  - 中頻控制 (300-3000Hz)
  - 高頻控制 (3000-8000Hz)
  - 每個頻段 ±12dB 可調

#### 進階處理
- ✅ **速度調整**
  - 支援 0.5x - 2.0x 速度調整
  - 保持音調不變 (使用 atempo)

- ✅ **音調調整**
  - 支援 ±12 半音調整
  - 使用 asetrate + atempo 組合

- ✅ **動態範圍壓縮**
  - 可調壓縮比 (1:1 - 20:1)
  - 可調閾值
  - 自動 attack/release 參數

- ✅ **格式轉換**
  - 立體聲轉單聲道
  - 取樣率轉換 (16kHz/44.1kHz/48kHz)

### 2. 後端 API 端點 ✅

**檔案：** [remote_server/api.py](remote_server/api.py)

新增以下 RESTful API 端點：

- `POST /api/preprocess` - 提交音訊預處理任務
- `GET /api/preprocess/{id}/download` - 下載預處理音訊（原始/處理後）
- `GET /api/preprocess/{id}/info` - 獲取預處理詳細資訊
- `DELETE /api/preprocess/{id}` - 刪除預處理檔案

### 3. 前端整合 ✅

#### API 客戶端
**檔案：** [frontend/src/api.ts](frontend/src/api.ts)

新增預處理 API 方法：
- `preprocessAudio()` - 上傳並預處理音訊
- `downloadPreprocessedAudio()` - 取得下載 URL
- `getPreprocessInfo()` - 查詢預處理資訊
- `deletePreprocess()` - 刪除預處理檔案

#### 預處理組件
**檔案：** [frontend/src/components/AudioPreprocessor.tsx](frontend/src/components/AudioPreprocessor.tsx)

實作完整的預處理使用者介面：
- ✅ 檔案選擇與上傳
- ✅ 多項預處理參數設定（滑桿、選單）
- ✅ A/B 音訊對比播放
- ✅ 原始/處理後音訊資訊顯示
- ✅ 即時錯誤提示

**樣式檔案：** [frontend/src/components/AudioPreprocessor.css](frontend/src/components/AudioPreprocessor.css)

#### 主應用整合
**檔案：** [frontend/src/ui/App.tsx](frontend/src/ui/App.tsx)

- ✅ 新增「預處理」導航按鈕
- ✅ 新增預處理頁面視圖模式
- ✅ 整合預處理組件至應用架構

**樣式更新：** [frontend/src/styles/main.css](frontend/src/styles/main.css)

## 🧪 測試驗證

### 測試檔案
1. [test_preprocess.py](test_preprocess.py) - 預處理模組單元測試
2. [test_api_preprocess.py](test_api_preprocess.py) - API 整合測試
3. [remote_server/api_preprocess_only.py](remote_server/api_preprocess_only.py) - 獨立預處理 API 測試版

### 測試結果 ✅
```
上傳檔案: 範例.mp3
預處理成功！
預處理 ID: cb010099-5a17-4730-98f4-7f0d74a66f65

原始音訊資訊:
  時長: 2.59 秒
  取樣率: 48000 Hz
  聲道: 2

處理後音訊資訊:
  時長: 2.66 秒
  取樣率: 16000 Hz
  聲道: 1

套用的濾鏡: afftdn=nr=48:nf=-25, highpass=f=200, pan=mono|c0=0.5*c0+0.5*c1, volume=-3.0dB
```

✅ **所有核心功能測試通過**

## 📁 檔案結構

```
Speech-to-text/
├── remote_server/
│   ├── audio_preprocessor.py      # 預處理核心引擎
│   ├── api.py                      # 主 API（含預處理端點）
│   ├── api_preprocess_only.py     # 測試用獨立 API
│   └── preprocessed/               # 預處理結果儲存目錄
│
├── frontend/
│   ├── src/
│   │   ├── api.ts                 # API 客戶端（含預處理方法）
│   │   ├── components/
│   │   │   ├── AudioPreprocessor.tsx
│   │   │   └── AudioPreprocessor.css
│   │   ├── ui/
│   │   │   └── App.tsx            # 主應用（含預處理頁面）
│   │   └── styles/
│   │       └── main.css           # 全域樣式
│
└── test_*.py                       # 測試檔案
```

## 🚀 使用方式

### 啟動後端（測試版）
```bash
cd Speech-to-text
.venv/Scripts/python.exe remote_server/api_preprocess_only.py
```

### 啟動前端
```bash
cd frontend
npm run dev
```

### 使用預處理功能
1. 在主頁點擊「預處理」按鈕
2. 選擇音訊檔案
3. 調整預處理參數
4. 點擊「開始預處理」
5. 使用 A/B 播放器對比原始與處理後音訊
6. 可下載處理後的音訊檔案

## 🎛️ 預處理參數說明

| 參數 | 說明 | 範圍 |
|------|------|------|
| `enable_denoise` | 啟用降噪 | bool |
| `denoise_strength` | 降噪強度 | 0.0-1.0 |
| `enable_normalize` | 啟用音量正規化 | bool |
| `normalize_type` | 正規化類型 | 'peak' / 'lufs' |
| `target_level` | 目標電平 | -20~0 dB (peak) / -30~-10 LUFS |
| `enable_silence_removal` | 移除靜音 | bool |
| `silence_threshold` | 靜音閾值 | -60~-20 dB |
| `min_silence_duration` | 最小靜音時長 | 0.1~2.0 秒 |
| `enable_vocal_enhancement` | 人聲增強 | bool |
| `enhancement_strength` | 增強強度 | 0.0-1.0 |
| `enable_echo_removal` | 迴聲消除 | bool |
| `enable_eq` | 啟用 EQ | bool |
| `eq_low_gain` | 低頻增益 | -12~12 dB |
| `eq_mid_gain` | 中頻增益 | -12~12 dB |
| `eq_high_gain` | 高頻增益 | -12~12 dB |
| `enable_speed_change` | 速度調整 | bool |
| `speed_factor` | 速度倍率 | 0.5-2.0 |
| `enable_compression` | 動態壓縮 | bool |
| `compression_ratio` | 壓縮比 | 1.0-20.0 |
| `enable_mono` | 轉單聲道 | bool |
| `enable_resample` | 重新取樣 | bool |
| `target_sample_rate` | 目標取樣率 | 16000/44100/48000 Hz |

## 🔄 後續整合建議

### 1. 與轉錄任務整合
可在 `UploadSection` 組件中新增「先預處理」選項，讓用戶在轉錄前先進行音訊預處理。

### 2. 預處理預設檔案
實作預處理設定檔儲存功能，讓用戶可以儲存常用的預處理配置。

### 3. 進階功能
- 整合 Demucs/Spleeter 進行人聲分離
- 新增音訊品質評估 (PESQ/STOI)
- 實作波形圖視覺化對比

### 4. 效能優化
- 實作預處理進度回報 (SSE)
- 支援批次預處理
- 快取常用預處理結果

## 📝 注意事項

1. **FFmpeg 依賴**：預處理功能依賴 FFmpeg，確保專案根目錄有 `ffmpeg-7.1.1-full_build-shared` 資料夾

2. **檔案管理**：預處理檔案儲存在 `remote_server/preprocessed/` 目錄，建議定期清理

3. **音訊格式**：輸出格式為 WAV，取樣率預設 16kHz 單聲道（適合語音辨識）

4. **處理時間**：預處理時間取決於音訊長度和啟用的濾鏡數量，長音訊可能需要較長時間

## ✅ 完成狀態

- [x] 後端預處理引擎
- [x] 後端 API 端點
- [x] 前端 API 封裝
- [x] 前端預處理組件
- [x] A/B 對比播放
- [x] 主應用整合
- [x] 功能測試驗證

**實作完成日期：** 2025-10-02

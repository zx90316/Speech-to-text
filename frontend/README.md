# Whisper 語音轉文字 - 前端應用

現代化的 Whisper 語音轉文字服務前端界面。

## ✨ 功能特色

- 📤 **拖放上傳**: 支援拖放或點擊上傳音訊檔案
- 📊 **即時進度**: 使用 SSE 即時顯示轉錄進度
- 👥 **語者分離**: 可選的多人對話識別
- 📝 **部分結果**: 處理過程中顯示已完成的轉錄片段
- 📚 **任務歷史**: 查看所有提交過的任務
- ⬇️ **快速下載**: 一鍵下載轉錄結果
- 🎨 **現代化 UI**: 深色主題，流暢動畫
- 📱 **響應式設計**: 支援桌面和移動設備

## 🚀 快速開始

### 環境需求

- Node.js 16+
- npm 或 yarn

### 安裝依賴

```bash
npm install
```

### 啟動開發服務器

```bash
npm run dev
```

前端將在 `http://localhost:5173` 啟動

### 構建生產版本

```bash
npm run build
```

構建產物將輸出到 `dist` 目錄

### 預覽生產版本

```bash
npm run preview
```

## 🔧 配置

### API 代理

前端通過 Vite 代理連接到後端 API（配置在 `vite.config.ts`）：

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

如果您的後端 API 運行在不同的地址，請修改此配置。

## 📁 項目結構

```
frontend/
├── src/
│   ├── components/          # React 組件
│   │   ├── UploadSection.tsx      # 上傳區域
│   │   ├── TaskProgress.tsx       # 任務進度顯示
│   │   ├── TaskHistory.tsx        # 任務歷史列表
│   │   └── ServiceStats.tsx       # 服務統計
│   ├── styles/              # 樣式文件
│   │   └── main.css              # 主樣式
│   ├── ui/                  # UI 相關
│   │   └── App.tsx               # 主應用組件
│   ├── api.ts               # API 客戶端
│   ├── types.ts             # TypeScript 類型定義
│   └── main.tsx             # 應用入口
├── index.html               # HTML 模板
├── package.json             # 依賴配置
├── tsconfig.json            # TypeScript 配置
└── vite.config.ts           # Vite 配置
```

## 🎨 技術棧

- **React 18**: UI 框架
- **TypeScript**: 類型安全
- **Vite**: 構建工具
- **Axios**: HTTP 客戶端
- **Lucide React**: 圖標庫
- **EventSource API**: SSE 進度推送

## 🔌 API 集成

前端與後端 API 的集成點：

### 1. 提交任務

```typescript
POST /api/tasks
FormData: { file: File }
Query: { enable_diarization: boolean }
```

### 2. 即時進度（SSE）

```typescript
GET /api/tasks/{taskId}/stream
EventSource: 持續接收進度更新
```

### 3. 查詢狀態

```typescript
GET /api/tasks/{taskId}
```

### 4. 下載結果

```typescript
GET /api/tasks/{taskId}/download?file_type=transcript
```

### 5. 任務歷史

```typescript
GET /api/my-tasks?limit=50
```

## 📝 使用說明

### 上傳檔案

1. 點擊或拖放音訊檔案到上傳區域
2. 選擇是否啟用語者分離
3. 點擊「開始轉錄」按鈕

### 監控進度

- 任務提交後會自動顯示進度面板
- 進度條和狀態會即時更新
- 可以查看部分轉錄結果（如果可用）
- 可以隨時取消正在進行的任務

### 查看歷史

- 右側面板顯示所有歷史任務
- 點擊任務項目查看詳情
- 已完成的任務可以直接下載結果

## 🐛 調試

### 使用瀏覽器開發者工具

1. 打開瀏覽器開發者工具（F12）
2. 查看 Network 標籤監控 API 請求
3. 查看 Console 標籤查看日誌
4. EventStream 標籤可以查看 SSE 連接

### 常見問題

**問題：無法連接到 API**
- 確認後端服務是否運行在 `http://localhost:8000`
- 檢查 `vite.config.ts` 中的代理配置

**問題：上傳失敗**
- 檢查檔案格式（支援 MP3, WAV, M4A, FLAC）
- 查看瀏覽器控制台的錯誤訊息

**問題：進度不更新**
- 檢查 SSE 連接是否正常
- 確認瀏覽器支援 EventSource API

## 📄 License

此項目僅供學習和研究使用。

# 遠端部署指南 - 前後端分離

本文件說明如何將前端和後端部署在不同設備上。

## 架構圖

```
┌─────────────────────┐                    ┌─────────────────────┐
│     前端設備 A       │    HTTP 請求       │     後端設備 B       │
│  (瀏覽器/開發機)     │ ─────────────────▶ │   (GPU 運算機)       │
│                     │                    │                     │
│  Vite Dev Server    │                    │  FastAPI + Whisper  │
│  Port: 5173         │                    │  Port: 8100         │
└─────────────────────┘                    └─────────────────────┘
```

## 快速開始

### 後端設備 B（GPU 運算機）

1. **啟動後端服務**
   ```batch
   cd backend
   START_REMOTE.bat
   ```

2. **或手動設定環境變數**
   ```batch
   set API_HOST=0.0.0.0
   set API_PORT=8100
   set CORS_ORIGINS=*
   python api.py
   ```

3. **防火牆設定**
   ```batch
   netsh advfirewall firewall add rule name="Whisper API" dir=in action=allow protocol=tcp localport=8100
   ```

### 前端設備 A

1. **修改 `START_REMOTE.bat` 中的後端 IP**
   ```batch
   set VITE_BACKEND_URL=http://後端IP:8100
   ```

2. **啟動前端**
   ```batch
   cd frontend
   START_REMOTE.bat
   ```

---

## 環境變數說明

### 後端環境變數

| 變數名稱 | 預設值 | 說明 |
|---------|-------|------|
| `API_HOST` | `0.0.0.0` | 監聽地址。`0.0.0.0` 接受所有來源，`127.0.0.1` 只接受本機 |
| `API_PORT` | `8100` | API 服務端口 |
| `CORS_ORIGINS` | `*` | 允許的前端來源。`*` = 全部，或指定如 `http://192.168.1.50:5173` |
| `INTERNAL_SYSTEM` | `true` | 內部系統模式。`true` = 純 HTTP 運行（跳過 HSTS 強制 HTTPS） |
| `ENABLE_DOCS` | `true` | 是否啟用 API 文件 (/docs, /redoc) |

### 前端環境變數

| 變數名稱 | 預設值 | 說明 |
|---------|-------|------|
| `VITE_BACKEND_URL` | `http://localhost:8100` | 後端 URL（透過 Vite Proxy 轉發） |
| `VITE_API_URL` | `/api` | 直接 API URL（生產環境 build 使用） |

---

## 兩種連接模式

### 模式 1：透過 Vite Proxy（開發環境推薦）

```
瀏覽器 → Vite (localhost:5173) → Proxy → 後端 (192.168.1.100:8100)
                  /api 請求轉發
```

**設定方式：**
- 前端：設定 `VITE_BACKEND_URL=http://後端IP:8100`
- 優點：不需要處理 CORS，瀏覽器只看到同一來源

### 模式 2：直接連接（生產環境推薦）

```
瀏覽器 → 靜態檔案服務 (前端)
    ↓
    └──→ 直接請求 → 後端 (192.168.1.100:8100/api)
```

**設定方式：**
- 前端：設定 `VITE_API_URL=http://後端IP:8100/api`
- 後端：設定 `CORS_ORIGINS` 為前端的來源
- 優點：Build 後可部署為純靜態檔案

---

## 安全建議

### 開發環境
- `CORS_ORIGINS=*` 允許所有來源（方便測試）

### 生產環境
- 指定具體的前端來源：`CORS_ORIGINS=https://your-frontend.com`
- 關閉 API 文件：`ENABLE_DOCS=false`
- 使用 HTTPS 和反向代理（如 Nginx）
- 設定強壯的 `ADMIN_TOKEN`

---

## 常見問題

### Q: 瀏覽器顯示 CORS 錯誤
**A:** 確認後端的 `CORS_ORIGINS` 設定正確，包含前端的完整 URL（含端口）。

### Q: 無法連接到後端
**A:** 
1. 確認後端 `API_HOST=0.0.0.0`
2. 檢查防火牆是否允許端口 8100
3. 確認兩台設備在同一網段或有正確路由

### Q: SSE 串流進度不更新
**A:** 確認 `VITE_API_URL` 或 `VITE_BACKEND_URL` 設定正確，且後端可訪問。

### Q: 如何確認後端正常運行？
**A:** 在瀏覽器訪問 `http://後端IP:8100/health` 或 `http://後端IP:8100/docs`


import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // 載入環境變數（使用 '.' 代表當前目錄）
  const env = loadEnv(mode, '.', '')
  
  // API 目標地址（從環境變數讀取，支援遠端後端）
  const apiTarget = env.VITE_BACKEND_URL || 'http://localhost:8100'

  console.log(`[Vite] API Proxy Target: ${apiTarget}`)

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',  // 監聽所有介面，支援區域網路訪問
      port: 5173,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,  // 修改請求標頭中的 Host 為目標 URL
          secure: false,  // 允許自簽名證書（開發環境）
        }
      }
    },
    // 預覽伺服器設定（npm run preview）
    preview: {
      host: '0.0.0.0',
      port: 4173,
    }
  }
})



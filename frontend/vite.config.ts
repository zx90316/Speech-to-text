import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'

// SSL 證書路徑（可以通過環境變數覆蓋）
const sslKeyPath = process.env.SSL_KEY_PATH || 'C:/nginx/ssl/server-key.pem'
const sslCertPath = process.env.SSL_CERT_PATH || 'C:/nginx/ssl/server-cert.pem'

// 檢查 SSL 證書是否存在
const hasSSL = fs.existsSync(sslKeyPath) && fs.existsSync(sslCertPath)

// 根據環境變數或證書存在性決定使用 HTTP 還是 HTTPS
const useHTTPS = process.env.VITE_USE_HTTPS === 'true' || hasSSL

// API 目標地址
const apiTarget = process.env.VITE_BACKEND_URL || 'https://localhost:8100'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // 如果有 SSL 證書且啟用 HTTPS，則使用 HTTPS
    ...(useHTTPS && hasSSL ? {
      https: {
        key: fs.readFileSync(sslKeyPath),
        cert: fs.readFileSync(sslCertPath),
      }
    } : {}),
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        secure: false, // 允許自簽證書
        ws: true, // 支持 WebSocket 和 SSE
        // 配置 proxy 事件處理，避免連接錯誤
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            // ===== 重要：轉發客戶端真實 IP =====
            // 獲取客戶端 IP（支援多層代理）
            const clientIp =
              // 1. 檢查是否已有 X-Forwarded-For（來自上層代理）
              req.headers['x-forwarded-for'] ||
              // 2. 檢查 X-Real-IP
              req.headers['x-real-ip'] ||
              // 3. 使用直接連接的 IP
              req.socket.remoteAddress ||
              req.connection.remoteAddress ||
              'unknown';

            // 設置 X-Forwarded-For 標頭
            if (typeof clientIp === 'string') {
              // 如果已有 X-Forwarded-For，追加當前 IP
              const existingForwarded = req.headers['x-forwarded-for'];
              if (existingForwarded) {
                proxyReq.setHeader('X-Forwarded-For', existingForwarded);
              } else {
                proxyReq.setHeader('X-Forwarded-For', clientIp);
              }

              // 設置 X-Real-IP（Nginx 風格）
              proxyReq.setHeader('X-Real-IP', clientIp.split(',')[0].trim());
            }

            // 設置其他轉發標頭
            proxyReq.setHeader('X-Forwarded-Proto', req.socket.encrypted ? 'https' : 'http');
            proxyReq.setHeader('X-Forwarded-Host', req.headers.host || 'localhost:5173');

            // SSE 請求需要特殊處理
            if (req.url?.includes('/stream')) {
              proxyReq.setHeader('Connection', 'keep-alive');
              proxyReq.setHeader('Cache-Control', 'no-cache');
            }
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            // 確保 SSE 響應正確處理
            if (req.url?.includes('/stream')) {
              proxyRes.headers['connection'] = 'keep-alive';
              proxyRes.headers['cache-control'] = 'no-cache';
            }
          });
        },
      }
    }
  },
})



import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API 目標地址
const apiTarget = process.env.VITE_BACKEND_URL || 'http://localhost:8100'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget
      }
    }
  },
})



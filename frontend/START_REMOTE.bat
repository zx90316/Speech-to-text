chcp 65001
@echo off
echo ================================================
echo Whisper 語音轉文字 - 前端啟動（遠端後端模式）
echo ================================================
echo.

cd /d "%~dp0"

REM 檢查 node_modules 是否存在
if not exist "node_modules\" (
    echo 首次運行，正在安裝依賴...
    echo.
    call npm install
    echo.
)

REM 設定後端 URL（請修改為您的後端 IP）
REM ================================================
REM 方式 1：透過 Vite Proxy 轉發（開發環境推薦）
REM 設定 VITE_BACKEND_URL 讓 Vite 轉發 /api 請求到後端
set VITE_BACKEND_URL=http://localhost:8100

REM 方式 2：直接連接後端（生產環境推薦）
REM 設定 VITE_API_URL 讓前端直接訪問後端 API
REM set VITE_API_URL=http://192.168.80.24:8100/api
REM ================================================

echo.
echo [設定資訊]
echo   前端地址: http://localhost:5173
echo   後端地址: %VITE_BACKEND_URL%
if defined VITE_API_URL echo   API 直連: %VITE_API_URL%
echo.
echo 請確保後端 API 已在目標設備上運行
echo.

npm run dev

pause


chcp 65001
@echo off
echo ================================================
echo Whisper 語音轉文字 - 前端啟動
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

echo 正在啟動開發服務器...
echo 前端地址: http://localhost:5173
echo.
echo 請確保後端 API 已在 http://localhost:8100 運行
echo.

npm run dev

pause


@echo off
chcp 65001 > nul
echo ========================================
echo   Backend V2 - Qwen ASR 服務啟動
echo ========================================
echo.

cd /d "%~dp0"

echo 啟動虛擬環境...
call ..\.venv\Scripts\activate.bat

echo.
echo 啟動伺服器 (Port 8100)...
echo API 文件: http://localhost:8100/docs
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8100 --reload

pause

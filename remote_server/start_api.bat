@echo off
echo ================================================
echo Whisper 語音轉文字 API 服務啟動中...
echo ================================================
echo.

cd /d "%~dp0"

REM 檢查虛擬環境是否存在
if not exist ".venv\" (
    echo 未找到虛擬環境，正在創建...
    python -m venv .venv
    echo 虛擬環境創建完成
    echo.
)

REM 啟動虛擬環境
call .venv\Scripts\activate.bat

REM 安裝或更新依賴
echo 檢查並安裝依賴套件...
pip install -r requirements.txt
echo.

REM 檢查 .env 檔案
if not exist ".env" (
    echo 警告: 未找到 .env 檔案
    echo 請創建 .env 檔案並設定 HUGGINGFACE_TOKEN
    echo.
    echo 範例:
    echo HUGGINGFACE_TOKEN=your_token_here
    echo.
    pause
)

REM 啟動 API 服務
echo ================================================
echo 正在啟動 API 服務...
echo API 地址: http://localhost:8000
echo API 文檔: http://localhost:8000/docs
echo ================================================
echo.

python api.py

pause


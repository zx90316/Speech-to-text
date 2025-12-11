chcp 65001
@echo off
echo ================================================
echo Whisper 語音轉文字 - 後端啟動（支援遠端前端）
echo ================================================
echo.

cd /d "%~dp0"
cd ..

REM 啟用虛擬環境
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate
) else (
    echo [警告] 未找到虛擬環境，請先建立 .venv
    pause
    exit /b 1
)

cd backend

REM ================================================
REM 後端設定
REM ================================================

REM API 監聽設定
set API_HOST=0.0.0.0
set API_PORT=8100

REM CORS 設定 - 允許的前端來源
REM 使用 * 允許所有來源（開發環境）
REM 或指定特定來源，多個來源用逗號分隔
REM 例如: set CORS_ORIGINS=http://192.168.1.50:5173,http://frontend.example.com
set CORS_ORIGINS=*

REM 內部系統模式 - 不強制 HTTPS（跳過 HSTS 標頭）
set INTERNAL_SYSTEM=true

REM ================================================

echo.
echo [後端設定]
echo   監聽地址: %API_HOST%:%API_PORT%
echo   CORS 來源: %CORS_ORIGINS%
echo   內部系統: %INTERNAL_SYSTEM% (不強制 HTTPS)
echo.
echo [防火牆提醒]
echo   請確保 Windows 防火牆已允許端口 %API_PORT%
echo   或執行: netsh advfirewall firewall add rule name="Whisper API" dir=in action=allow protocol=tcp localport=%API_PORT%
echo.

python api.py

pause


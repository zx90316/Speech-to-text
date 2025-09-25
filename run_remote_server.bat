chcp 65001
@ECHO OFF
:: 這個 .bat 檔必須放在專案根目錄下才能正常運作

ECHO 正在啟動虛擬環境...

:: 啟動在同目錄下的 .venv 虛擬環境
CALL .\.venv\Scripts\activate.bat

ECHO 虛擬環境已啟動，正在執行 Python 腳本...
ECHO.

:: 執行同目錄下的 main.py
python .\remote_server\remote_inference_server.py

ECHO.
ECHO 腳本執行完畢。
PAUSE
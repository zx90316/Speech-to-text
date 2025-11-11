@echo off
REM Windows 批次腳本 - 運行測試
REM 
REM 使用方法:
REM   run_tests.bat          - 運行所有測試
REM   run_tests.bat unit     - 只運行單元測試
REM   run_tests.bat coverage - 運行測試並生成覆蓋率

echo ========================================
echo 🧪 運行 pytest 測試套件
echo ========================================
echo.

if "%1"=="" (
    echo 運行所有測試...
    python run_tests.py --mode all --verbose
) else if "%1"=="unit" (
    echo 運行單元測試...
    python run_tests.py --mode unit --verbose
) else if "%1"=="integration" (
    echo 運行整合測試...
    python run_tests.py --mode integration --verbose
) else if "%1"=="security" (
    echo 運行安全測試...
    python run_tests.py --mode security --verbose
) else if "%1"=="api" (
    echo 運行 API 測試...
    python run_tests.py --mode api --verbose
) else if "%1"=="fast" (
    echo 運行快速測試...
    python run_tests.py --mode fast --verbose
) else if "%1"=="coverage" (
    echo 運行測試並生成覆蓋率報告...
    python run_tests.py --mode coverage --verbose
) else (
    echo 未知的測試模式: %1
    echo.
    echo 可用模式:
    echo   all         - 所有測試
    echo   unit        - 單元測試
    echo   integration - 整合測試
    echo   security    - 安全測試
    echo   api         - API 測試
    echo   fast        - 快速測試（跳過耗時測試）
    echo   coverage    - 測試 + 覆蓋率報告
)

pause


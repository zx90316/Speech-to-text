@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===== Speech-to-text �@��w��/�Ұ� =====
REM �Ϊk�G
REM   ���������ΰ���G�۰ʧP�_�O�_�ݭn�w�� �� �M��ҰʩҦ��A��
REM   run.bat install  �u���w�ˡ]�إ� venv�B�w�ˮM�󵥡^
REM   run.bat start    �ȱҰʡ]���]�w�w�˦n�^

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM �i��G�Y�n�Ұ� Celery�A�N�U�C�ܼƧאּ 1�]�ݦۦ�Ұ� Redis�^
set START_CELERY=

call :detect_python || goto :error

if /I "%1"=="install" goto :install
if /I "%1"=="start" goto :start

REM �۰ʧP�_�O�_�ݭn�w��
if not exist .venv\Scripts\python.exe goto :install
if exist frontend\package.json if not exist frontend\node_modules goto :install
goto :start

:install
echo [Install] 建立根目錄 venv 與相依...
if not exist .venv\Scripts\python.exe (
  %PY_CMD% -m venv .venv || goto :error
)
call .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
call .venv\Scripts\pip.exe install -r backend\requirements.txt || goto :error

echo [Install] 遠端伺服器相依...
call .venv\Scripts\pip.exe install -r remote_server\requirements.txt || goto :error

if exist frontend\package.json (
  echo [Install] �e�� npm �ۨ�...
  call :detect_npm || goto :skip_frontend
  pushd frontend
  call npm install || (popd & goto :error)
  popd
) else (
  echo [Install] �䤣�� frontend\package.json�A���L�e�ݦw��
)

if /I "%1"=="install" goto :end

:start
echo [Start] �Ұʻ��� Whisper ���A���]8001�^...
start "remote_server" cmd /c "cd /d remote_server && call ..\.venv\Scripts\python.exe -m uvicorn remote_inference_server:app --host 0.0.0.0 --port 8001 | cat"

if not "%START_CELERY%"=="" (
  echo [Start] �Ұ� Celery worker...
  start "celery" cmd /c "cd /d backend && call ..\.venv\Scripts\celery.exe -A app.celery_app.celery_app worker --loglevel=INFO | cat"
)

echo [Start] �Ұʫ�� FastAPI�]8000�^...
start "backend" cmd /c "cd /d backend && call ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload | cat"

if exist frontend\package.json (
  echo [Start] �Ұʫe�ݡ]Vite, �w�] 5173�^...
  start "frontend" cmd /c "cd /d frontend && call npm run dev | cat"
) else (
  echo [Start] �䤣��e�ݡA���L
)

echo.
echo �Ҧ��A�Ȥw�Ұʡ]��W�ߵ����^�C
echo ���:   http://localhost:8000
echo �e��:   http://localhost:5173
echo ����ASR: http://localhost:8001/healthz
goto :end

:detect_python
for /f "delims=" %%i in ('where python 2^>nul') do set PY_CMD=python& goto :py_ok
for /f "delims=" %%i in ('where py 2^>nul') do set PY_CMD=py -3.11& goto :py_ok
echo [Error] �䤣�� Python�]�� 3.11�^�C�Цw�˫᭫�աC
exit /b 1
:py_ok
%PY_CMD% --version
exit /b 0

:detect_npm
for /f "delims=" %%i in ('where npm 2^>nul') do set NPM_OK=1& goto :npm_ok
echo [Warn] �䤣�� npm�A�N���L�e�ݨB�J�C
exit /b 1
:npm_ok
exit /b 0

:skip_frontend
echo [Warn] �䤣�� npm�A���L�e�ݦw��
goto :after_frontend

:error
echo.
echo [Error] ����L�{�o�Ϳ��~�A���ˬd�W�z�T���C
exit /b 1

:after_frontend
goto :start

:end
endlocal
exit /b 0



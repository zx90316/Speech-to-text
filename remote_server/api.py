"""
FastAPI 應用主檔案
提供完整的語音轉文字 API 服務
符合 SSDLC 安全要求
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# 導入安全模組
from memory_storage import memory_manager
from email_service import email_service
from task_processor import task_processor
from security_logger import security_logger
from input_validator import input_validator
from rate_limiter import rate_limiter
from crypto_utils import crypto_utils


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # Startup
    # 清理殘留的臨時檔案
    memory_manager.cleanup_all_temporary_files()

    # 啟動任務佇列處理器
    asyncio.create_task(process_queue())

    print("=" * 60)
    print("Whisper 語音轉文字 API 服務已啟動（記憶體模式 + 郵件通知）")
    print("=" * 60)
    yield
    # Shutdown (if needed)


# 創建速率限制器
limiter = Limiter(key_func=get_remote_address)

# 創建 FastAPI 應用
app = FastAPI(
    title="Whisper 語音轉文字 API",
    description="基於 Faster-Whisper 和 Pyannote 的語音轉文字服務，支援語者分離（符合 SSDLC 安全要求）",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None
)

# 註冊速率限制異常處理
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 中介軟體（更安全的配置）
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # 改為白名單
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],  # 限制允許的方法
    allow_headers=["Content-Type", "Authorization"],  # 限制允許的標頭
    max_age=3600,  # 預檢請求緩存時間
)

# 信任主機中介軟體（防止 Host Header 攻擊）
trusted_hosts = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=trusted_hosts
)


# 安全標頭中介軟體
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """添加安全響應標頭"""
    response = await call_next(request)

    # 安全標頭（符合 SSDLC 要求）
    response.headers["X-Content-Type-Options"] = "nosniff"  # 防止 MIME 類型嗅探
    response.headers["X-Frame-Options"] = "DENY"  # 防止點擊劫持
    response.headers["X-XSS-Protection"] = "1; mode=block"  # XSS 保護
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"  # HSTS
    response.headers["Content-Security-Policy"] = "default-src 'self'"  # CSP
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"  # Referrer 策略
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"  # 權限策略

    # 移除可能洩露服務器信息的標頭
    response.headers.pop("Server", None)

    return response


# 創建必要的資料夾
RESULT_DIR = Path(__file__).parent / "result"
UPLOAD_DIR = Path(__file__).parent / "uploads"
RESULT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# 任務佇列
task_queue = asyncio.Queue()
processing = False


def get_client_ip(request: Request) -> str:
    """獲取客戶端 IP 地址"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def process_queue():
    """處理任務佇列"""
    global processing
    while True:
        task_id = await task_queue.get()
        processing = True

        try:
            # 使用 get_task_full() 獲取完整任務資訊（包含檔案路徑）
            task = memory_manager.get_task_full(task_id)
            if not task or task['status'] == 'canceled':
                task_queue.task_done()
                processing = False
                continue

            # 獲取任務檔案路徑（從任務資料中取得）
            upload_path = Path(task['upload_path']) / task['filename']

            # 在後台線程中執行處理，避免阻塞事件循環
            await asyncio.to_thread(
                task_processor.process_task_sync,
                task_id=task_id,
                audio_path=str(upload_path),
                enable_diarization=task['enable_diarization'],
                start_time=task.get('start_time'),
                end_time=task.get('end_time'),
                language=task.get('language'),
                task=task.get('task', 'transcribe'),
                model=task.get('model', 'CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32'),
                vad_onset=task.get('vad_onset', 0.5),
                vad_offset=task.get('vad_offset', 0.363),
                min_speakers=task.get('min_speakers'),
                max_speakers=task.get('max_speakers'),
                enable_confidence_score=task.get('enable_confidence_score', False),
                compute_type=task.get('compute_type', None)
            )

        except Exception as e:
            print(f"處理任務 {task_id} 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            memory_manager.update_task_status(task_id, 'failed', error_message=str(e))

        finally:
            task_queue.task_done()
            processing = False




@app.get("/")
async def root():
    """根路徑"""
    return {
        "service": "Whisper 語音轉文字 API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "queue_size": task_queue.qsize(),
        "processing": processing
    }


@app.post("/api/tasks", summary="提交轉錄任務")
async def create_task(
    request: Request,
    email: str = Query(..., description="已驗證的電子郵件（用於接收結果）"),
    file: UploadFile = File(..., description="音訊檔案 (支援 mp3, wav, m4a, flac)"),
    enable_diarization: bool = Query(True, description="是否啟用語者分離"),
    start_time: Optional[float] = Query(None, ge=0, description="開始時間（秒）"),
    end_time: Optional[float] = Query(None, ge=0, description="結束時間（秒）"),
    language: Optional[str] = Query(None, description="語言代碼（如 zh, en, ja），留空自動偵測"),
    task: str = Query("transcribe", description="任務類型：transcribe（轉錄）或 translate（翻譯成英文）"),
    model: str = Query("CWTchen/Belle-whisper-large-v3-zh-punct-ct2-float32", description="Whisper 模型"),
    # 進階參數
    vad_onset: float = Query(0.5, ge=0, le=1, description="VAD 語音檢測敏感度 (0-1)"),
    vad_offset: float = Query(0.363, ge=0, le=1, description="VAD 語音結束閾值 (0-1)"),
    min_speakers: Optional[int] = Query(None, ge=1, description="最小語者數"),
    max_speakers: Optional[int] = Query(None, ge=1, description="最大語者數"),
    enable_confidence_score: bool = Query(False, description="是否啟用信心分數輸出"),
    compute_type: Optional[str] = Query(None, description="計算類型 (float32, int8, float16)"),
    # LLM 校對參數
    enable_llm_correction: bool = Query(False, description="是否啟用 LLM 文本校對"),
    llm_model: Optional[str] = Query(None, description="LLM 模型 (gemma3:4b, qwen3:4b, gpt-oss:20b)")
):
    """
    提交新的語音轉文字任務

    - **email**: 已驗證的電子郵件地址（必須先完成驗證）
    - **file**: 音訊檔案
    - **enable_diarization**: 是否啟用語者分離功能
    - **start_time**: 音訊開始時間（秒），可選
    - **end_time**: 音訊結束時間（秒），可選
    - **language**: 語言代碼（如 zh, en, ja），留空則自動偵測
    - **task**: transcribe（轉錄）或 translate（翻譯成英文）

    返回任務ID，處理完成後會將結果發送至您的郵箱
    """
    # 驗證郵箱是否已驗證
    if not email_service.is_email_verified(email):
        raise HTTPException(
            status_code=403,
            detail="郵箱未驗證，請先使用 /api/email/send-verification 發送驗證碼"
        )

    # 驗證任務類型
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail="任務類型必須是 transcribe 或 translate")

    # 驗證時間範圍
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise HTTPException(status_code=400, detail="開始時間必須小於結束時間")

    # 檢查檔案類型
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供檔案名稱")

    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的檔案格式。支援的格式: {', '.join(allowed_extensions)}"
        )

    # 生成任務 ID
    task_id = str(uuid.uuid4())

    # 在記憶體中創建任務記錄
    task_data = memory_manager.create_task(
        task_id=task_id,
        email=email,
        filename=file.filename,
        enable_diarization=enable_diarization,
        start_time=start_time,
        end_time=end_time,
        language=language,
        task=task,
        model=model,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        enable_confidence_score=enable_confidence_score,
        compute_type=compute_type,
        enable_llm_correction=enable_llm_correction,
        llm_model=llm_model
    )

    # 保存上傳的檔案到臨時目錄
    upload_path = Path(task_data['upload_path']) / file.filename
    content = await file.read()
    with open(upload_path, 'wb') as f:
        f.write(content)

    # 加入處理佇列
    await task_queue.put(task_id)

    # 計算佇列位置
    queue_position = memory_manager.get_queue_position(task_id)

    return {
        "task_id": task_id,
        "status": "pending",
        "queue_position": queue_position,
        "message": f"任務已提交，正在排隊處理。完成後將發送結果至 {email}"
    }


@app.get("/api/tasks/{task_id}", summary="查詢任務狀態")
async def get_task_status(task_id: str):
    """
    查詢任務的當前狀態

    - **task_id**: 任務ID

    返回任務的詳細資訊，包括進度、狀態、錯誤訊息等
    """
    task = memory_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到該任務")

    # 計算佇列位置
    queue_position = 0
    if task['status'] == 'pending':
        queue_position = memory_manager.get_queue_position(task_id)
    
    return {
        "task_id": task['task_id'],
        "filename": task['filename'],
        "status": task['status'],
        "progress": task['progress'],
        "current_stage": task['current_stage'],
        "queue_position": queue_position,
        "enable_diarization": task['enable_diarization'],
        "created_at": task['created_at'],
        "started_at": task['started_at'],
        "completed_at": task['completed_at'],
        "error_message": task['error_message'],
        "has_result": task['status'] == 'completed'
    }


@app.get("/api/tasks/{task_id}/stream", summary="串流任務進度（SSE）")
async def stream_task_progress(task_id: str):
    """
    使用 Server-Sent Events (SSE) 即時推送任務進度

    - **task_id**: 任務ID

    持續推送任務狀態更新，直到任務完成或失敗
    """
    task = memory_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到該任務")

    async def event_generator():
        """SSE 事件生成器"""
        last_progress = -1
        last_status = None

        while True:
            task = memory_manager.get_task(task_id)
            if not task:
                yield f"event: error\ndata: {{\"message\": \"任務不存在\"}}\n\n"
                break

            # 只在狀態改變時推送
            if task['progress'] != last_progress or task['status'] != last_status:
                last_progress = task['progress']
                last_status = task['status']

                queue_position = 0
                if task['status'] == 'pending':
                    queue_position = memory_manager.get_queue_position(task_id)
                
                event_data = {
                    "status": task['status'],
                    "progress": task['progress'],
                    "current_stage": task['current_stage'],
                    "queue_position": queue_position,
                    "error_message": task['error_message'],
                    "has_result": task['status'] == 'completed',
                    "timestamp": datetime.now().isoformat()
                }

                # 如果有 ASR 進度信息，也一併推送
                if task.get('asr_progress'):
                    event_data['asr_progress'] = task['asr_progress']

                # 如果有部分結果，也一併推送
                if task.get('partial_result'):
                    event_data['partial_result'] = task['partial_result']
                
                import json
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 如果任務已完成或失敗，結束串流
            if task['status'] in ('completed', 'failed', 'canceled'):
                break
            
            # 等待一段時間後再次檢查
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/api/tasks/{task_id}", summary="取消任務")
async def cancel_task(task_id: str, permanent: bool = Query(False, description="是否永久刪除任務及其檔案")):
    """
    取消正在進行或排隊中的任務

    - **task_id**: 任務ID
    - **permanent**: 如果為 True，則永久刪除任務記錄和相關檔案

    注意：已完成的任務無法取消，但可以永久刪除
    """
    task = memory_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到該任務")

    # 如果是永久刪除
    if permanent:
        # 清理任務檔案
        memory_manager.cleanup_task_files(task_id)

        # 從記憶體刪除記錄
        memory_manager.delete_task(task_id)

        return {
            "task_id": task_id,
            "status": "deleted",
            "message": "任務已永久刪除"
        }

    # 否則只是取消任務
    success = memory_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"任務狀態為 {task['status']}，無法取消"
        )

    return {
        "task_id": task_id,
        "status": "canceled",
        "message": "任務已取消"
    }


@app.get("/api/my-tasks", summary="查詢我的任務歷史")
async def get_my_tasks(
    email: str = Query(..., description="已驗證的電子郵件"),
    limit: int = Query(50, ge=1, le=100, description="返回的任務數量上限")
):
    """
    根據電子郵件查詢該用戶提交過的所有任務

    - **email**: 電子郵件地址
    - **limit**: 返回的任務數量（預設 50，最多 100）
    """
    # 驗證郵箱是否已驗證
    if not email_service.is_email_verified(email):
        raise HTTPException(
            status_code=403,
            detail="郵箱未驗證，請先完成驗證"
        )

    tasks = memory_manager.get_tasks_by_email(email, limit=limit)

    # 簡化返回的資訊
    simplified_tasks = []
    for task in tasks:
        simplified_tasks.append({
            "task_id": task['task_id'],
            "filename": task['filename'],
            "status": task['status'],
            "progress": task['progress'],
            "enable_diarization": task['enable_diarization'],
            "created_at": task['created_at'],
            "completed_at": task['completed_at'],
            "has_result": task['status'] == 'completed'
        })

    return {
        "email": email,
        "total": len(simplified_tasks),
        "tasks": simplified_tasks
    }


@app.post("/api/tasks/batch", summary="批量查詢任務")
async def get_tasks_batch(task_ids: List[str]):
    """
    根據任務 ID 列表批量查詢任務資訊
    支援 localStorage 方式追蹤任務

    - **task_ids**: 任務 ID 列表
    """
    if len(task_ids) > 100:
        raise HTTPException(status_code=400, detail="一次最多查詢 100 個任務")

    tasks = []
    for task_id in task_ids:
        task = memory_manager.get_task(task_id)
        if task:
            tasks.append({
                "task_id": task['task_id'],
                "filename": task['filename'],
                "status": task['status'],
                "progress": task['progress'],
                "enable_diarization": task['enable_diarization'],
                "created_at": task['created_at'],
                "completed_at": task['completed_at'],
                "has_result": task['status'] == 'completed'
            })

    return {
        "total": len(tasks),
        "tasks": tasks
    }


@app.get("/api/stats", summary="服務統計資訊")
async def get_stats():
    """
    獲取服務的統計資訊
    """
    processing_count = memory_manager.get_processing_count()
    queue_size = task_queue.qsize()

    return {
        "queue_size": queue_size,
        "processing_count": processing_count,
        "is_processing": processing,
        "total_waiting": queue_size + processing_count
    }


# ==================== 郵件驗證 API ====================

@app.post("/api/email/send-verification", summary="發送郵件驗證碼")
@limiter.limit("5/hour")  # 每小時最多 5 次
async def send_verification_email(
    request: Request,
    email: str = Query(..., description="郵件地址")
):
    """
    發送郵件驗證碼

    - **email**: 郵件地址

    返回發送狀態
    """
    ip_address = get_client_ip(request)

    # 輸入驗證
    is_valid, error = input_validator.validate_email(email)
    if not is_valid:
        security_logger.log_invalid_request(ip_address, "/api/email/send-verification", error)
        raise HTTPException(status_code=400, detail=error)

    # 速率限制檢查
    is_allowed, error = rate_limiter.check_email_verification_rate_limit(email)
    if not is_allowed:
        security_logger.log_rate_limit_exceeded(ip_address, "/api/email/send-verification")
        raise HTTPException(status_code=429, detail=error)

    # 檢查 IP 和郵箱黑名單
    is_blacklisted, remaining = rate_limiter.is_ip_blacklisted(ip_address)
    if is_blacklisted:
        security_logger.log_security_event(
            "BLACKLISTED_IP_ATTEMPT",
            ip_address,
            "warning",
            "Blacklisted IP attempted to send verification code"
        )
        raise HTTPException(
            status_code=403,
            detail=f"IP 已被臨時封禁，請在 {remaining} 秒後再試"
        )

    is_blacklisted, remaining = rate_limiter.is_email_blacklisted(email)
    if is_blacklisted:
        raise HTTPException(
            status_code=403,
            detail=f"郵箱已被臨時封禁，請在 {remaining} 秒後再試"
        )

    # 發送驗證碼
    success = email_service.send_verification_email(email)

    if not success:
        security_logger.log_error(
            "EMAIL_SEND_FAILURE",
            "Failed to send verification email",
            user_id=email,
            details={"ip_address": ip_address}
        )
        raise HTTPException(status_code=500, detail="發送驗證碼失敗，請檢查 SMTP 設定")

    # 記錄日誌
    security_logger.log_verification_code_sent(email, ip_address)

    return {
        "success": True,
        "message": "驗證碼已發送至您的郵箱，有效期 5 分鐘"
    }


@app.post("/api/email/verify-code", summary="驗證郵件驗證碼")
@limiter.limit("10/minute")  # 每分鐘最多 10 次
async def verify_email_code(
    request: Request,
    email: str = Query(..., description="郵件地址"),
    code: str = Query(..., description="驗證碼")
):
    """
    驗證郵件驗證碼

    - **email**: 郵件地址
    - **code**: 6 位數驗證碼

    返回驗證結果
    """
    ip_address = get_client_ip(request)

    # 輸入驗證
    is_valid, error = input_validator.validate_email(email)
    if not is_valid:
        security_logger.log_invalid_request(ip_address, "/api/email/verify-code", error)
        raise HTTPException(status_code=400, detail=error)

    is_valid, error = input_validator.validate_verification_code(code)
    if not is_valid:
        security_logger.log_invalid_request(ip_address, "/api/email/verify-code", error)
        raise HTTPException(status_code=400, detail=error)

    # 檢查黑名單
    is_blacklisted, remaining = rate_limiter.is_email_blacklisted(email)
    if is_blacklisted:
        raise HTTPException(
            status_code=403,
            detail=f"郵箱已被臨時封禁，請在 {remaining} 秒後再試"
        )

    # 驗證驗證碼
    is_valid = email_service.verify_code(email, code)

    if not is_valid:
        # 記錄驗證失敗
        is_banned = rate_limiter.record_verification_failure(email, ip_address)

        security_logger.log_auth_attempt(
            email,
            ip_address,
            success=False,
            reason="Invalid verification code"
        )

        if is_banned:
            security_logger.log_security_event(
                "VERIFICATION_BRUTEFORCE_DETECTED",
                ip_address,
                "warning",
                f"Multiple verification failures for email: {crypto_utils.mask_email(email)}"
            )
            raise HTTPException(
                status_code=403,
                detail="驗證失敗次數過多，已被臨時封禁"
            )

        # 獲取剩餘嘗試次數
        remaining_attempts = rate_limiter.get_remaining_attempts(email, ip_address)
        raise HTTPException(
            status_code=400,
            detail=f"驗證碼無效或已過期（剩餘嘗試次數：{remaining_attempts}）"
        )

    # 驗證成功，重置失敗記錄
    rate_limiter.reset_verification_failures(email, ip_address)

    # 記錄成功日誌
    security_logger.log_auth_attempt(email, ip_address, success=True)

    return {
        "success": True,
        "message": "郵箱驗證成功",
        "email": email
    }


# ==================== 管理者 API ====================

def verify_admin_token(token: str = Query(..., description="管理者 Token")):
    """驗證管理者 Token"""
    admin_token = os.getenv("ADMIN_TOKEN", "admin_secret_token_2024")
    if token != admin_token:
        raise HTTPException(status_code=403, detail="無效的管理者權限")
    return True


@app.get("/api/admin/tasks", summary="管理者 - 獲取所有任務")
async def admin_get_all_tasks(
    _: bool = Depends(verify_admin_token),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    管理者：獲取所有任務（僅佇列資訊，不含任務內容）
    需要提供有效的 admin_token
    """
    tasks = memory_manager.get_all_tasks_summary(limit=limit, offset=offset)
    total = memory_manager.get_total_tasks_count()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "tasks": tasks
    }


@app.get("/api/admin/stats", summary="管理者 - 系統統計")
async def admin_get_stats(_: bool = Depends(verify_admin_token)):
    """
    管理者：獲取詳細的系統統計資訊
    """
    stats = memory_manager.get_stats_summary()
    queue_size = task_queue.qsize()
    processing_count = memory_manager.get_processing_count()

    return {
        **stats,
        "queue_size": queue_size,
        "processing_count": processing_count,
        "is_processing": processing
    }


@app.post("/api/admin/tasks/batch-delete", summary="管理者 - 批量刪除任務")
async def admin_batch_delete_tasks(
    _: bool = Depends(verify_admin_token),
    task_ids: List[str] = Query(..., description="要刪除的任務 ID 列表")
):
    """
    管理者：批量刪除任務及其相關檔案
    """
    deleted_count = 0
    for task_id in task_ids:
        # 清理任務檔案
        if memory_manager.cleanup_task_files(task_id):
            deleted_count += 1

        # 從記憶體刪除
        memory_manager.delete_task(task_id)

    return {
        "deleted_count": deleted_count,
        "message": f"已刪除 {deleted_count} 個任務"
    }


@app.post("/api/admin/cleanup", summary="管理者 - 清理舊任務")
async def admin_cleanup_old_tasks(
    _: bool = Depends(verify_admin_token),
    keep_count: int = Query(100, ge=10, description="保留最近的 N 個已完成任務")
):
    """
    管理者：清理舊的已完成任務，保留最近的 N 個
    """
    deleted_count = memory_manager.cleanup_old_completed_tasks(keep_count=keep_count)

    return {
        "deleted_count": deleted_count,
        "message": f"已清理 {deleted_count} 個舊任務，保留最近 {keep_count} 個"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
        log_level="info"
    )


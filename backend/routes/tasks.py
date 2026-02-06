# -*- coding: utf-8 -*-
"""
任務 API 路由
"""
import os
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from config import UPLOAD_DIR, RESULT_DIR, DEFAULT_QWEN_MODEL
from models import TaskStatus, TaskResponse, TaskListResponse, TranscriptSegment, WordInfo
from storage import storage
from pipeline import pipeline

router = APIRouter(tags=["任務"])

# 執行緒池用於後台處理
executor = ThreadPoolExecutor(max_workers=2)


def _convert_segments(segments):
    """將儲存的 segments 轉換為回應格式"""
    if not segments:
        return None
    
    result = []
    for seg in segments:
        words = None
        if seg.get("words"):
            words = [
                WordInfo(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    probability=w.get("probability", 1.0),
                )
                for w in seg["words"]
            ]
        
        result.append(TranscriptSegment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            speaker=seg.get("speaker"),
            words=words,
        ))
    
    return result


def _task_to_response(task: dict) -> TaskResponse:
    """將任務資料轉換為回應格式"""
    return TaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        current_stage=task["current_stage"],
        error_message=task["error_message"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        filename=task["filename"],
        text=task["text"],
        language=task["detected_language"],
        segments=_convert_segments(task["segments"]),
        has_diarization=task["has_diarization"],
    )


@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音訊檔案"),
    email: Optional[str] = Query(None, description="通知郵件地址（任務完成時發送通知）"),
    enable_diarization: bool = Query(True, description="是否啟用語者分離"),
    enable_timestamps: bool = Query(False, description="是否啟用時間戳輸出"),
    language: Optional[str] = Query(None, description="語言代碼 (zh/en/ja 等)"),
    model: str = Query(DEFAULT_QWEN_MODEL, description="ASR 模型名稱"),
    min_speakers: Optional[int] = Query(None, ge=1, le=20, description="最小語者數"),
    max_speakers: Optional[int] = Query(None, ge=1, le=20, description="最大語者數"),
):
    """
    提交新的轉錄任務
    
    上傳音訊檔案並開始處理。若提供 email，任務完成時會發送通知。
    """
    # 驗證檔案
    if not file.filename:
        raise HTTPException(status_code=400, detail="請提供檔案")
    
    # 檢查副檔名
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.webm', '.mp4'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的檔案格式: {file_ext}。支援的格式: {', '.join(allowed_extensions)}"
        )
    
    # 建立任務
    task_id = storage.create_task(
        filename=file.filename,
        email=email,
        enable_diarization=enable_diarization,
        enable_timestamps=enable_timestamps,
        language=language,
        model=model,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    
    # 儲存檔案
    upload_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    try:
        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        storage.delete_task(task_id)
        raise HTTPException(status_code=500, detail=f"檔案儲存失敗: {str(e)}")
    
    # 更新任務的音訊路徑
    storage.update_task(task_id, audio_path=str(upload_path))
    
    # 在背景執行處理
    def run_pipeline():
        try:
            pipeline.process(task_id)
        finally:
            # 清理上傳的檔案
            if upload_path.exists():
                try:
                    os.remove(upload_path)
                except Exception:
                    pass
    
    background_tasks.add_task(run_pipeline)
    
    task = storage.get_task(task_id)
    return _task_to_response(task)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """查詢任務狀態"""
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    return _task_to_response(task)


@router.get("/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """
    SSE 即時進度推送
    
    持續推送任務狀態更新，直到任務完成或失敗
    """
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    async def event_generator():
        last_progress = -1.0
        
        while True:
            task = storage.get_task(task_id)
            if not task:
                yield {
                    "event": "error",
                    "data": '{"error": "任務不存在"}'
                }
                break
            
            # 只在進度變化時發送
            if task["progress"] != last_progress:
                last_progress = task["progress"]
                
                status = task["status"]
                yield {
                    "event": "progress",
                    "data": f'{{"task_id": "{task_id}", "status": "{status}", "progress": {task["progress"]}, "stage": "{task["current_stage"]}"}}'
                }
            
            # 任務結束
            if task["status"] in ["completed", "failed", "canceled"]:
                response = _task_to_response(task)
                yield {
                    "event": "complete",
                    "data": response.model_dump_json()
                }
                break
            
            await asyncio.sleep(0.5)
    
    return EventSourceResponse(event_generator())


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """取消任務"""
    task = storage.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    if task["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="無法取消已完成或失敗的任務")
    
    success = storage.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="取消失敗")
    
    return {"message": "任務已取消", "task_id": task_id}


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    limit: int = Query(50, ge=1, le=100, description="返回數量限制")
):
    """取得任務列表"""
    tasks = storage.get_all_tasks(limit=limit)
    return TaskListResponse(
        tasks=[_task_to_response(t) for t in tasks],
        total=len(tasks),
    )

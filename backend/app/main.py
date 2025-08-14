from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .storage import TaskStore
from .utils.formatting import generate_srt
from .config import settings
from .tasks import transcribe_remote_task,  transcribe_vertex_task
from .services.transcription_remote import transcribe_with_remote_llm
from .services.transcription_vertex import transcribe_with_vertex_ai


app = FastAPI(title="Speech-to-Text Backend", version="0.1.0")

# CORS
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.post("/api/v1/transcribe")
async def create_transcription_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_choice: Literal["vertex_ai", "remote_llm"] = Query(...),
    start_time: Optional[str] = Query(default=None, description="HH:MM:SS"),
    end_time: Optional[str] = Query(default=None, description="HH:MM:SS"),
    language_code: Optional[str] = Query(default="zh-TW"),
    # 共同參數
    chunk_length: Optional[float] = Query(default=30.0, description="分塊秒數"),
    # Vertex 參數
    prompt: Optional[str] = Query(default=None, description="提示詞"),
    temperature: Optional[float] = Query(default=1.0),
    top_p: Optional[float] = Query(default=0.95),
    max_output_tokens: Optional[int] = Query(default=65535),
    thinking_budget: Optional[int] = Query(default=0),
    safety_off: Optional[bool] = Query(default=True),
):
    if file.content_type is None or not any(
        file.filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".flac")
    ):
        raise HTTPException(status_code=400, detail="不支援的音訊格式，請上傳 wav/mp3/m4a/flac。")

    task_id = str(uuid.uuid4())
    TaskStore.initialize_task(task_id=task_id, model_choice=model_choice, start_time=start_time, end_time=end_time)

    # 將實際工作交給背景執行
    contents = await file.read()
    if settings.use_celery:
        if model_choice == "remote_llm":
            transcribe_remote_task.delay(task_id, contents, start_time, end_time)
        elif model_choice == "vertex_ai":
            transcribe_vertex_task.delay(
                task_id,
                contents,
                start_time,
                end_time,
                language_code,
                prompt,
                temperature,
                top_p,
                max_output_tokens,
                thinking_budget,
                safety_off,
                chunk_length,
            )

    else:
        if model_choice == "remote_llm":
            background_tasks.add_task(
                transcribe_with_remote_llm,
                task_id=task_id,
                raw_bytes=contents,
                start_time=start_time,
                end_time=end_time,
                chunk_length_s=float(chunk_length or 30.0),
            )
        elif model_choice == "vertex_ai":
            background_tasks.add_task(
                transcribe_with_vertex_ai,
                task_id=task_id,
                raw_bytes=contents,
                start_time=start_time,
                end_time=end_time,
                chunk_length_s=float(chunk_length or 30.0),
                language_code=language_code or "zh-TW",
                prompt=prompt,
                temperature=float(temperature or 0),
                top_p=float(top_p or 0.95),
                max_output_tokens=int(max_output_tokens or 65535),
                thinking_budget=int(thinking_budget or 0),
                safety_off=bool(safety_off if safety_off is not None else True),
            )

    return {"task_id": task_id}


@app.websocket("/ws/v1/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            task = TaskStore.get_task(task_id)
            if task is None:
                await websocket.send_json({"status": "failed", "error": "未知的任務 ID"})
                break

            await websocket.send_json(
                {
                    "status": task["status"],
                    "progress": task["progress"],
                    "partial_text": task.get("partial_text", ""),
                    "tokens": task.get("tokens", {"input": 0, "output": 0}),
                    "error": task.get("error", ""),
                    "segments": task.get("segments", []),
                }
            )

            if task["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        # 使用者關閉頁面或連線中斷，這裡不需額外處理
        return


@app.get("/api/v1/result/{task_id}")
async def get_result(
    task_id: str,
    format: Literal["plain", "timestamped", "srt"] = Query("plain"),
):
    task = TaskStore.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="找不到此任務")
    if task["status"] != "completed":
        raise HTTPException(status_code=409, detail="任務尚未完成")

    segments = task.get("segments", [])

    filename = f"transcript_{task_id}.{ 'srt' if format == 'srt' else 'txt'}"

    if format == "plain":
        text = "".join([s.get("text", "") for s in segments])
        content = text or task.get("partial_text", "")
        return Response(
            content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    if format == "timestamped":
        lines = []
        for s in segments:
            start, end, text = s.get("start", 0.0), s.get("end", 0.0), s.get("text", "")
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")
        return Response(
            "\n".join(lines),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    if format == "srt":
        return Response(
            generate_srt(segments),
            media_type="application/x-subrip; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    raise HTTPException(status_code=400, detail="不支援的輸出格式")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/api/v1/cancel/{task_id}")
async def cancel_task(task_id: str):
    task = TaskStore.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="找不到此任務")
    if task.get("status") in ("completed", "failed", "canceled"):
        return {"status": task.get("status")}
    TaskStore.mark_canceled(task_id)
    return {"status": "canceled"}



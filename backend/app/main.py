from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from .storage import TaskStore
from .utils.formatting import generate_srt
from .config import settings
from .tasks import transcribe_remote_task, transcribe_sim_task, transcribe_vertex_task
from .services.transcription_sim import simulate_transcription
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
            transcribe_vertex_task.delay(task_id, contents, start_time, end_time, language_code)
        else:
            transcribe_sim_task.delay(task_id, contents)
    else:
        if model_choice == "remote_llm":
            background_tasks.add_task(
                transcribe_with_remote_llm,
                task_id,
                contents,
                start_time,
                end_time,
            )
        elif model_choice == "vertex_ai":
            background_tasks.add_task(
                transcribe_with_vertex_ai,
                task_id,
                contents,
                start_time,
                end_time,
                30.0,
                language_code or "zh-TW",
            )
        else:
            background_tasks.add_task(simulate_transcription, task_id, contents)

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



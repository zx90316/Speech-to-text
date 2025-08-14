from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse

from .storage import TaskStore
from .utils.formatting import generate_srt
from .services.transcription_sim import simulate_transcription


app = FastAPI(title="Speech-to-Text Backend", version="0.1.0")


@app.post("/api/v1/transcribe")
async def create_transcription_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_choice: Literal["vertex_ai", "remote_llm"] = Query(...),
    start_time: Optional[str] = Query(default=None, description="HH:MM:SS"),
    end_time: Optional[str] = Query(default=None, description="HH:MM:SS"),
):
    if file.content_type is None or not any(
        file.filename.lower().endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".flac")
    ):
        raise HTTPException(status_code=400, detail="不支援的音訊格式，請上傳 wav/mp3/m4a/flac。")

    task_id = str(uuid.uuid4())
    TaskStore.initialize_task(task_id=task_id, model_choice=model_choice, start_time=start_time, end_time=end_time)

    # 將實際工作交給背景執行（此為模擬實作，後續可替換為 Celery 任務）
    contents = await file.read()
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

    if format == "plain":
        text = "".join([s.get("text", "") for s in segments])
        return PlainTextResponse(text or task.get("partial_text", ""))

    if format == "timestamped":
        lines = []
        for s in segments:
            start, end, text = s.get("start", 0.0), s.get("end", 0.0), s.get("text", "")
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")
        return PlainTextResponse("\n".join(lines))

    if format == "srt":
        return PlainTextResponse(generate_srt(segments), media_type="application/x-subrip")

    raise HTTPException(status_code=400, detail="不支援的輸出格式")



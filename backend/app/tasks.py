from __future__ import annotations

import asyncio

from .celery_app import celery_app
from .services.transcription_remote import transcribe_with_remote_llm
from .services.transcription_sim import simulate_transcription
from .storage import save_temp_upload, delete_file_silent, read_file_bytes
from .services.transcription_vertex import transcribe_with_vertex_ai


@celery_app.task(name="transcribe_remote")
def transcribe_remote_task(task_id: str, raw_bytes: bytes, start_time: str | None, end_time: str | None) -> None:
    # 某些 broker/backend 對大型 bytes 支援不佳，可以先落地檔案再讀回
    path = save_temp_upload(raw_bytes, suffix=".bin")
    try:
        data = read_file_bytes(path)
        transcribe_with_remote_llm(
            task_id=task_id,
            raw_bytes=data,
            start_time=start_time,
            end_time=end_time,
        )
    finally:
        delete_file_silent(path)


@celery_app.task(name="transcribe_sim")
def transcribe_sim_task(task_id: str, raw_bytes: bytes) -> None:
    asyncio.run(simulate_transcription(task_id=task_id, raw_bytes=raw_bytes))


@celery_app.task(name="transcribe_vertex")
def transcribe_vertex_task(task_id: str, raw_bytes: bytes, start_time: str | None, end_time: str | None, language_code: str = "zh-TW") -> None:
    path = save_temp_upload(raw_bytes, suffix=".bin")
    try:
        data = read_file_bytes(path)
        transcribe_with_vertex_ai(
            task_id=task_id,
            raw_bytes=data,
            start_time=start_time,
            end_time=end_time,
            language_code=language_code,
        )
    finally:
        delete_file_silent(path)



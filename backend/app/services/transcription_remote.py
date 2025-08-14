from __future__ import annotations

import os
import time
import tempfile
from typing import Optional

import httpx

from ..config import settings
from ..storage import TaskStore
from ..utils.formatting import parse_hhmmss
from ..utils.ffmpeg import (
    ensure_ffmpeg_available,
    ffprobe_duration_seconds,
    ffmpeg_extract_segment_to_wav,
)


def _resolve_time_range(total_seconds: float, start_time_str: Optional[str], end_time_str: Optional[str]) -> tuple[float, float]:
    start_s = parse_hhmmss(start_time_str) if start_time_str else 0.0
    end_s = parse_hhmmss(end_time_str) if end_time_str else total_seconds
    start_s = max(0.0, min(start_s, total_seconds))
    end_s = max(start_s, min(end_s, total_seconds))
    return start_s, end_s


def _iter_offsets(start_s: float, end_s: float, chunk_length_s: float):
    chunk = max(1.0, float(chunk_length_s))
    offset = float(start_s)
    while offset < end_s:
        remain = end_s - offset
        duration = min(chunk, remain)
        yield offset, duration
        offset += duration


def transcribe_with_remote_llm(
    task_id: str,
    raw_bytes: bytes,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    chunk_length_s: float = 30.0,
) -> None:
    try:
        ensure_ffmpeg_available()

        # 將原始 bytes 落地為暫存檔，交由 ffmpeg 處理
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as src:
            src.write(raw_bytes)
            src_path = src.name

        total_seconds = ffprobe_duration_seconds(src_path)
        start_s, end_s = _resolve_time_range(total_seconds, start_time, end_time)
        if end_s - start_s <= 0.0:
            TaskStore.mark_failed(task_id, error_message="音訊長度為 0，請確認檔案或時間區段設定。")
            os.remove(src_path)
            return

        processed = 0.0

        with httpx.Client(base_url=settings.remote_server_url, timeout=httpx.Timeout(120.0)) as client:
            for offset, duration in _iter_offsets(start_s, end_s, chunk_length_s):
                if TaskStore.is_canceled(task_id):
                    TaskStore.mark_failed(task_id, error_message="任務已取消")
                    return
                chunk_wav = ffmpeg_extract_segment_to_wav(src_path, offset_seconds=offset, duration_seconds=duration)
                try:
                    with open(chunk_wav, "rb") as f:
                        files = {"file": ("chunk.wav", f, "audio/wav")}
                        resp = client.post("/transcribe/", files=files)
                    resp.raise_for_status()
                    data = resp.json()
                    text = str(data.get("text", ""))

                    TaskStore.append_segment(task_id, start=offset - start_s, end=offset - start_s + duration, text=text)

                    processed = offset + duration - start_s
                    progress = (processed / (end_s - start_s)) * 100.0
                    TaskStore.update_progress(task_id, progress=progress)
                finally:
                    try:
                        os.remove(chunk_wav)
                    except Exception:
                        pass

                time.sleep(0.05)

        TaskStore.mark_completed(task_id)
    except Exception as e:
        TaskStore.mark_failed(task_id, error_message=str(e))
    finally:
        try:
            os.remove(src_path)  # type: ignore[name-defined]
        except Exception:
            pass



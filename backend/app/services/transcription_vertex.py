from __future__ import annotations

import time
import tempfile
from typing import Optional

from ..storage import TaskStore
from ..utils.formatting import parse_hhmmss
from ..config import settings
from ..utils.ffmpeg import (
    ensure_ffmpeg_available,
    ffprobe_duration_seconds,
    ffmpeg_extract_segment_to_wav,
)
from google import genai
from google.genai import types


def _predict_chunk_with_vertex(task_id: str, wav_bytes: bytes, language_code: str, stream_timeout_s: float = 30.0) -> str:
    client = genai.Client(
        vertexai=True,
        project=settings.vertex_project,
        location=(settings.vertex_location or "global"),
    )

    try:
        audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    except Exception:
        audio_part = types.Part.from_data(data=wav_bytes, mime_type="audio/wav")  # type: ignore

    # 盡量貼近官方範例：加入明確的文字指示，忽略背景音
    contents = [
        types.Content(
            role="user",
            parts=[
                audio_part,
                types.Part.from_text(
                    text=(
                        "Generate a transcription of the audio, only extract speech and ignore background audio. If any part of the speech is in Chinese, please transcribe it using Traditional Chinese characters."
                    )
                ),
            ],
        )
    ]

    # 先嘗試串流；若在 timeout 內沒有結果，改用非串流 fallback
    import threading

    text_chunks: list[str] = []
    caught_error: list[Exception] = []

    def _worker():
        try:
            for chunk in client.models.generate_content_stream(
                model=settings.vertex_genai_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    max_output_tokens=65535,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    ],
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ):
                try:
                    # 嘗試更新輸出 token（若串流回傳可得）
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage is not None and getattr(usage, "total_token_count", None):
                        try:
                            TaskStore.increment_tokens(task_id, input_tokens=int(usage.prompt_token_count), output_tokens=int(usage.candidates_token_count))
                        except Exception as e:
                            print(e)
                            pass

                    if hasattr(chunk, "text") and isinstance(chunk.text, str) and chunk.text:
                        text_chunks.append(chunk.text)
                        TaskStore.update_partial_text(task_id, chunk.text, append=True)
                except Exception as e:  # noqa: BLE001
                    caught_error.append(e)
                    break
        except Exception as e:  # noqa: BLE001
            caught_error.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=stream_timeout_s)

def transcribe_with_vertex_ai(
    task_id: str,
    raw_bytes: bytes,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    chunk_length_s: float = 30.0,
    language_code: str = "zh-TW",
) -> None:
    try:
        ensure_ffmpeg_available()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as src:
            src.write(raw_bytes)
            src_path = src.name

        total_seconds = ffprobe_duration_seconds(src_path)
        start_s = parse_hhmmss(start_time) if start_time else 0.0
        end_s = parse_hhmmss(end_time) if end_time else total_seconds
        start_s = max(0.0, min(start_s, total_seconds))
        end_s = max(start_s, min(end_s, total_seconds))
        if end_s - start_s <= 0.0:
            TaskStore.mark_failed(task_id, error_message="音訊長度為 0，請確認檔案或時間區段設定。")
            return

        processed = 0.0
        offset = start_s
        while offset < end_s:
            duration = min(chunk_length_s, end_s - offset)
            chunk_wav = ffmpeg_extract_segment_to_wav(src_path, offset_seconds=offset, duration_seconds=duration)
            try:
                with open(chunk_wav, "rb") as f:
                    wav_bytes = f.read()
                text = _predict_chunk_with_vertex(task_id, wav_bytes, language_code=language_code)
                TaskStore.append_segment(task_id, start=offset - start_s, end=offset - start_s + duration, text=text)
                processed = (offset + duration) - start_s
                progress = (processed / (end_s - start_s)) * 100.0
                TaskStore.update_progress(task_id, progress=progress)
            finally:
                try:
                    import os
                    os.remove(chunk_wav)
                except Exception:
                    pass
            offset += duration
            time.sleep(0.05)

        TaskStore.mark_completed(task_id)
    except Exception as e:
        TaskStore.mark_failed(task_id, error_message=str(e))
    finally:
        try:
            import os
            os.remove(src_path)  # type: ignore[name-defined]
        except Exception:
            pass



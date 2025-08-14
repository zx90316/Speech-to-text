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


def _predict_chunk_with_vertex(
    task_id: str,
    wav_bytes: bytes,
    language_code: str,
    *,
    stream_timeout_s: float = 30.0,
    prompt: str | None = None,
    temperature: float = 0.7,  # 降低溫度以減少重複
    top_p: float = 0.9,        # 稍微降低top_p
    max_output_tokens: int = 65535,
    thinking_budget: int = 0,
    safety_off: bool = True,
) -> str:
    client = genai.Client(
        vertexai=True,
        project=settings.vertex_project,
        location=(settings.vertex_location or "global"),
    )

    try:
        audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    except Exception:
        audio_part = types.Part.from_data(data=wav_bytes, mime_type="audio/wav")  # type: ignore


    si_text = """Your task is to be an expert transcriptionist.
Provide a direct transcription of the audio file. The output must ONLY contain the transcribed text.
You MUST NOT output any preambles, notes, introductions, or self-references. For example, your output must never start with phrases like "The content of this audio is:", "Here is the transcription:", or in Chinese "這段音訊的內容是：".

Key requirements:
1. Extract speech only. Ignore all background sounds.
2. If any speech is in Chinese, you MUST transcribe it using Traditional Chinese characters.
3. CRITICAL: Avoid repeating the same phrases or words. Ensure the output is clean and non-repetitive.
"""

    # 內容：音訊 + 提示（可自訂）
    contents = [
        types.Content(
            role="user",
            parts=[
                audio_part,
                types.Part.from_text(text='逐字稿：')
                ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0.1,
        frequency_penalty=0.7,  # 關鍵：增加一個正值來懲罰重複
        presence_penalty=0.5,       # 關鍵：增加一個正值來鼓勵新詞彙
        top_p = top_p,
        max_output_tokens = 500,
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
        system_instruction=[types.Part.from_text(text=si_text)],
        #thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget,),
    )
    
    caught_error: list[Exception] = []
    
    try:
        response = client.models.generate_content(
        model = settings.vertex_genai_model,
        contents = contents,
        config = generate_content_config,
        )

        TaskStore.update_partial_text(task_id, response.text, append=True)

        if response and hasattr(response, "usage_metadata") and response.usage_metadata:
            TaskStore.increment_tokens(
                task_id,
                input_tokens=int(getattr(response.usage_metadata, "prompt_token_count", 0)),
                output_tokens=int(getattr(response.usage_metadata, "candidates_token_count", 0)),
            )

        # 返回完整的轉錄文本
        return response.text
        
    except Exception as e:
        caught_error.append(e)
        return ""

def transcribe_with_vertex_ai(
    task_id: str,
    raw_bytes: bytes,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    chunk_length_s: float = 30.0,
    language_code: str = "zh-TW",
    prompt: str | None = None,
    temperature: float = 0,
    top_p: float = 0.95,
    max_output_tokens: int = 65535,
    thinking_budget: int = 0,
    safety_off: bool = True,
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
            # 支援取消
            if TaskStore.is_canceled(task_id):
                TaskStore.mark_failed(task_id, error_message="任務已取消")
                return
            duration = min(chunk_length_s, end_s - offset)
            chunk_wav = ffmpeg_extract_segment_to_wav(src_path, offset_seconds=offset, duration_seconds=duration)
            try:
                # 在串流過程會即時把 token 追加到 partial_text
                # 這裡先記錄呼叫前的 partial_text，若最終 text 為空，會以增量補齊段落文字
                try:
                    previous_partial = str(TaskStore.get_task(task_id).get("partial_text", ""))
                except Exception:
                    previous_partial = ""
                with open(chunk_wav, "rb") as f:
                    wav_bytes = f.read()
                text = _predict_chunk_with_vertex(
                    task_id,
                    wav_bytes,
                    language_code=language_code,
                    stream_timeout_s=30.0,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=thinking_budget,
                    safety_off=safety_off,
                )
                # 檢查是否有有效的轉錄結果
                if text and str(text).strip():
                    # 流式處理已返回完整文本，直接使用
                    TaskStore.append_segment(
                        task_id,
                        start=offset - start_s,
                        end=offset - start_s + duration,
                        text=str(text).strip(),
                    )
                else:
                    # 備用機制：若返回的文本為空，檢查是否有通過流式更新的 partial_text
                    try:
                        current_partial = str(TaskStore.get_task(task_id).get("partial_text", ""))
                        if len(current_partial) > len(previous_partial):
                            # 提取本次處理新增的文本部分
                            new_text = current_partial[len(previous_partial):].strip()
                            if new_text:
                                TaskStore.append_segment(
                                    task_id,
                                    start=offset - start_s,
                                    end=offset - start_s + duration,
                                    text=new_text,
                                )
                    except Exception:
                        pass
                
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



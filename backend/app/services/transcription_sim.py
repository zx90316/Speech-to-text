from __future__ import annotations

import asyncio
import random

from ..storage import TaskStore


async def simulate_transcription(task_id: str, raw_bytes: bytes) -> None:
    try:
        total_chunks = 10
        accumulated = 0.0
        for i in range(total_chunks):
            await asyncio.sleep(0.6)
            # 模擬產出一段結果
            duration = random.uniform(2.0, 6.0)
            start = accumulated
            end = accumulated + duration
            accumulated = end
            text = f"[片段{i+1}] 測試文字。"
            TaskStore.append_segment(task_id, start=start, end=end, text=text)
            TaskStore.update_progress(task_id, progress=((i + 1) / total_chunks) * 100.0)

        TaskStore.mark_completed(task_id)
    except Exception as e:
        TaskStore.mark_failed(task_id, error_message=str(e))



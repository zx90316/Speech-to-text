from __future__ import annotations

import threading
from typing import Dict, Any, Optional


_lock = threading.Lock()
_tasks: Dict[str, Dict[str, Any]] = {}


class TaskStore:
    @staticmethod
    def initialize_task(task_id: str, model_choice: str, start_time: str | None, end_time: str | None) -> None:
        with _lock:
            _tasks[task_id] = {
                "status": "processing",
                "progress": 0.0,
                "partial_text": "",
                "segments": [],
                "meta": {
                    "model_choice": model_choice,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            }

    @staticmethod
    def get_task(task_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            return _tasks.get(task_id)

    @staticmethod
    def append_segment(task_id: str, start: float, end: float, text: str) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["segments"].append({"start": start, "end": end, "text": text})
            task["partial_text"] += text

    @staticmethod
    def update_progress(task_id: str, progress: float) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["progress"] = float(max(0.0, min(100.0, progress)))

    @staticmethod
    def mark_completed(task_id: str) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["status"] = "completed"
            task["progress"] = 100.0

    @staticmethod
    def mark_failed(task_id: str, error_message: str) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["status"] = "failed"
            task["error"] = error_message



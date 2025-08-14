from __future__ import annotations

import threading
from typing import Dict, Any, Optional
import os
import tempfile


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
                "tokens": {"input": 0, "output": 0},
                "canceled": False,
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
            safe_text = "" if text is None else str(text)
            task["segments"].append({"start": start, "end": end, "text": safe_text})

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

    @staticmethod
    def update_partial_text(task_id: str, text: str, *, append: bool = True) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            safe_text = "" if text is None else str(text)
            if append:
                task["partial_text"] += safe_text

    @staticmethod
    def increment_tokens(task_id: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            tokens = task.setdefault("tokens", {"input": 0, "output": 0})
            tokens["input"] = int(tokens.get("input", 0)) + int(max(0, input_tokens))
            tokens["output"] = int(tokens.get("output", 0)) + int(max(0, output_tokens))

    @staticmethod
    def set_tokens(task_id: str, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            tokens = task.setdefault("tokens", {"input": 0, "output": 0})
            if input_tokens is not None:
                tokens["input"] = int(max(0, input_tokens))
            if output_tokens is not None:
                tokens["output"] = int(max(0, output_tokens))

    @staticmethod
    def mark_canceled(task_id: str) -> None:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return
            task["canceled"] = True
            task["status"] = "canceled"

    @staticmethod
    def is_canceled(task_id: str) -> bool:
        with _lock:
            task = _tasks.get(task_id)
            if not task:
                return False
            return bool(task.get("canceled", False))



def save_temp_upload(contents: bytes, suffix: str | None = None) -> str:
    """將上傳檔案 bytes 儲存為臨時檔，回傳檔案路徑。"""
    if not suffix or not suffix.startswith("."):
        suffix = ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        return tmp.name


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def delete_file_silent(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


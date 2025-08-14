from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _append_ffmpeg_path_from_env() -> None:
    raw = os.getenv("FFMPEG_PATH", "")
    if not raw:
        return

    # 支援多個路徑，以系統分隔符號分隔（; on Windows, : on Unix）
    parts = [p for p in raw.split(os.pathsep) if p.strip()]
    if not parts:
        return

    # 專案根目錄（backend/app/utils/ffmpeg.py → 根在 parents[3]）
    project_root = Path(__file__).resolve().parents[3]
    cwd = Path.cwd()

    for p in parts:
        path_obj = Path(p)
        candidates = [path_obj]
        if not path_obj.is_absolute():
            candidates.append(project_root / p)
            candidates.append(cwd / p)

        for cand in candidates:
            if cand.exists():
                cand_str = str(cand)
                if cand_str not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + cand_str
                break


def ensure_ffmpeg_available() -> None:
    _append_ffmpeg_path_from_env()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請安裝或在 .env 設定 FFMPEG_PATH。")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("找不到 ffprobe，請安裝或在 .env 設定 FFMPEG_PATH。")


def ffprobe_duration_seconds(input_path: str) -> float:
    ensure_ffmpeg_available()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    out = result.stdout.decode(errors="ignore").strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def ffmpeg_trim_to_file(
    input_path: str,
    output_suffix: str = ".wav",
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
) -> str:
    ensure_ffmpeg_available()
    with tempfile.NamedTemporaryFile(delete=False, suffix=output_suffix) as tmp:
        out_path = tmp.name

    cmd = ["ffmpeg", "-y"]
    if start_seconds is not None:
        cmd += ["-ss", str(max(0.0, start_seconds))]
    cmd += ["-i", input_path]
    if end_seconds is not None and start_seconds is not None and end_seconds > start_seconds:
        duration = max(0.0, end_seconds - start_seconds)
        cmd += ["-t", str(duration)]
    # 輸出為 wav 以相容 downstream
    cmd += ["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def ffmpeg_extract_segment_to_wav(
    input_path: str,
    offset_seconds: float,
    duration_seconds: float,
) -> str:
    ensure_ffmpeg_available()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, offset_seconds)),
        "-i",
        input_path,
        "-t",
        str(max(0.0, duration_seconds)),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path



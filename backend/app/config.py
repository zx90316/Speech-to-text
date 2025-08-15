from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from pathlib import Path

from dotenv import load_dotenv


root_env = Path(__file__).resolve().parents[2] / ".env"
backend_env = Path(__file__).resolve().parents[1] / ".env"
for dotenv_path in (root_env, backend_env):
    load_dotenv(dotenv_path=dotenv_path, override=True)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    remote_server_url: str = os.getenv("REMOTE_SERVER_URL", "http://localhost:8001")
    vertex_project: str = os.getenv("VERTEX_PROJECT", "vscc-faq")
    vertex_location: str = os.getenv("VERTEX_LOCATION", "global")
    vertex_genai_model: str = os.getenv("VERTEX_GENAI_MODEL", "gemini-2.5-flash-lite")
    use_celery: bool = _parse_bool(os.getenv("USE_CELERY", None), default=False)
    cors_origins: List[str] = None


settings = Settings(
    cors_origins=_parse_csv(os.getenv("CORS_ORIGINS", "http://192.168.80.24:5173,http://localhost:5173,http://127.0.0.1:5173")),
)



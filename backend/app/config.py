from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    remote_server_url: str = os.getenv("REMOTE_SERVER_URL", "http://localhost:8001")
    vertex_project: str = os.getenv("VERTEX_PROJECT", "")


settings = Settings()



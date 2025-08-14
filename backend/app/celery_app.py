from __future__ import annotations

from celery import Celery

from .config import settings


celery_app = Celery(
    "speech_to_text_backend",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)



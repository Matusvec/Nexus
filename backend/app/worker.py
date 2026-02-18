"""Celery worker configuration for background job processing.

Optional upgrade from FastAPI BackgroundTasks to Celery for:
- Horizontal scaling (multiple workers)
- Guaranteed delivery (job survives server restart)
- Priority queues and concurrency control

Usage:
    celery -A app.worker worker --loglevel=info --concurrency=4
"""

from celery import Celery
from app.config import settings

celery_app = Celery("nexus", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,  # for reliability
    worker_prefetch_multiplier=1,
)

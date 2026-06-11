"""Heuristics for background worker availability."""

from datetime import datetime, timedelta, timezone


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def is_background_worker_likely_available(
    db,
    stale_pending_minutes: int = 5,
    stale_processing_minutes: int = 30,
) -> bool:
    """
    Return False when pending or processing jobs look stalled, indicating
    the Railway background worker is likely down.
    """
    now = datetime.now(timezone.utc)
    pending_threshold = now - timedelta(minutes=stale_pending_minutes)
    processing_threshold = now - timedelta(minutes=stale_processing_minutes)

    for job in db.get_pending_jobs(limit=20):
        created = _parse_timestamp(job.get("created_at"))
        if created and created < pending_threshold:
            return False

    for job in db.get_processing_jobs():
        started = _parse_timestamp(job.get("started_at")) or _parse_timestamp(
            job.get("created_at")
        )
        if started and started < processing_threshold:
            return False

    return True

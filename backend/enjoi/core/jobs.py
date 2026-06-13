"""Background job manager: threads + WebSocket progress fan-out.

Long pipeline stages run in daemon threads. Progress updates are throttled and
broadcast to all connected WebSocket clients via asyncio queues; the REST
endpoint /api/jobs/{id} can also be polled.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Callable

from .errors import PipelineError

log = logging.getLogger("enjoi.jobs")

ProgressFn = Callable[[float, str], None]


class Job:
    def __init__(self, job_type: str, project_id: str | None):
        self.id = "j_" + uuid.uuid4().hex[:10]
        self.type = job_type
        self.project_id = project_id
        self.status = "queued"            # queued|running|done|error
        self.progress = 0.0
        self.message = ""
        self.result: Any = None
        self.error: str | None = None
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "project_id": self.project_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
        }


def subprogress(progress: ProgressFn, lo: float, hi: float, prefix: str = "") -> ProgressFn:
    """Map a child stage's 0..1 progress into the [lo, hi] window of the parent."""

    def fn(frac: float, msg: str) -> None:
        frac = min(max(frac, 0.0), 1.0)
        progress(lo + (hi - lo) * frac, (prefix + msg) if prefix else msg)

    return fn


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---- websocket plumbing ------------------------------------------------
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    def _emit(self, job: Job) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        payload = {"type": "job", "job": job.to_dict()}

        def push() -> None:
            for q in list(self._listeners):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

        try:
            self._loop.call_soon_threadsafe(push)
        except RuntimeError:
            pass

    # ---- job lifecycle -----------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def start(
        self,
        job_type: str,
        project_id: str | None,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Job:
        job = Job(job_type, project_id)
        with self._lock:
            self.jobs[job.id] = job

        last_emit = [0.0]

        def progress(frac: float, msg: str) -> None:
            job.progress = min(max(float(frac), 0.0), 1.0)
            job.message = str(msg)
            now = time.time()
            if now - last_emit[0] >= 0.1 or job.progress >= 1.0:
                last_emit[0] = now
                self._emit(job)

        def run() -> None:
            job.status = "running"
            self._emit(job)
            try:
                job.result = fn(*args, progress=progress, **kwargs)
                job.status = "done"
                job.progress = 1.0
                job.message = job.message or "Done"
            except PipelineError as exc:
                job.status = "error"
                job.error = str(exc)
                log.warning("job %s (%s) failed: %s", job.id, job.type, exc)
            except Exception as exc:  # unexpected — log full trace, keep msg readable
                job.status = "error"
                job.error = f"Unexpected error: {exc}"
                log.error("job %s (%s) crashed:\n%s", job.id, job.type, traceback.format_exc())
            self._emit(job)

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return job


manager = JobManager()

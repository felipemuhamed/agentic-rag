import logging
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from app.uploads.models import (
    JobKind,
    JobNotFoundError,
    JobQueueFullError,
    UploadJob,
    utc_now_iso,
)


logger = logging.getLogger(__name__)


class UploadJobManager:
    """Run ingestion work off the request path and expose its status.

    Ingestion loads embedding models and can take minutes for a large manual, so
    the API answers `202 Accepted` and the caller polls the job. Workers default
    to one thread because embedding on CPU is already the bottleneck.
    """

    def __init__(
        self,
        *,
        workers: int = 1,
        max_pending: int = 32,
        max_jobs: int = 200,
        retention_seconds: float = 3_600.0,
        clock: Callable[[], float] = monotonic,
    ):
        if workers < 1 or max_pending < 1 or max_jobs < 1:
            raise ValueError("invalid upload job manager configuration")
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be greater than zero")
        self._workers = workers
        self._max_pending = max_pending
        self._max_jobs = max_jobs
        self._retention_seconds = retention_seconds
        self._clock = clock
        self._jobs: OrderedDict[str, UploadJob] = OrderedDict()
        self._finished_at: dict[str, float] = {}
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def submit(
        self,
        *,
        kind: JobKind,
        filename: str,
        principal: str,
        tenant_id: str | None,
        work: Callable[[], dict[str, Any]],
    ) -> UploadJob:
        job = UploadJob(
            job_id=uuid4().hex,
            kind=kind,
            filename=filename,
            principal=principal,
            tenant_id=tenant_id,
        )
        with self._lock:
            if self._closed:
                raise JobQueueFullError("the ingestion queue is shutting down")
            self._prune()
            pending = sum(
                1
                for stored in self._jobs.values()
                if stored.status in {"queued", "running"}
            )
            if pending >= self._max_pending:
                raise JobQueueFullError("too many ingestion jobs in progress")
            self._jobs[job.job_id] = job
            executor = self._ensure_executor()
            # Snapshot before queueing: a fast job could otherwise finish and
            # make the 202 response describe an already terminal state.
            accepted = job.snapshot()
        executor.submit(self._run, job.job_id, work)
        return accepted

    def get(
        self,
        job_id: str,
        *,
        principal: str,
        tenant_id: str | None,
    ) -> UploadJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.owned_by(
                principal=principal,
                tenant_id=tenant_id,
            ):
                raise JobNotFoundError(job_id)
            return job.snapshot()

    def list(
        self,
        *,
        principal: str,
        tenant_id: str | None,
        limit: int = 50,
    ) -> list[UploadJob]:
        with self._lock:
            self._prune()
            jobs = [
                job.snapshot()
                for job in self._jobs.values()
                if job.owned_by(principal=principal, tenant_id=tenant_id)
            ]
        return list(reversed(jobs))[:limit]

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait)

    def _run(self, job_id: str, work: Callable[[], dict[str, Any]]) -> None:
        self._update(job_id, status="running")
        try:
            result = work()
        except Exception as error:
            logger.warning(
                "Upload ingestion failed",
                extra={"job_id": job_id, "error_type": type(error).__name__},
                exc_info=True,
            )
            self._update(
                job_id,
                status="failed",
                error=str(error) or type(error).__name__,
                error_code=error_code_for(error),
            )
            return
        self._update(job_id, status="succeeded", result=result)

    def _update(self, job_id: str, *, status: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status  # type: ignore[assignment]
            job.updated_at = utc_now_iso()
            for name, value in fields.items():
                setattr(job, name, value)
            if status in {"succeeded", "failed"}:
                self._finished_at[job_id] = self._clock()

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers,
                thread_name_prefix="upload-ingest",
            )
        return self._executor

    def _prune(self) -> None:
        now = self._clock()
        expired = [
            job_id
            for job_id, finished_at in self._finished_at.items()
            if now - finished_at > self._retention_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._finished_at.pop(job_id, None)

        while len(self._jobs) > self._max_jobs:
            for job_id, job in self._jobs.items():
                if job.status in {"succeeded", "failed"}:
                    self._jobs.pop(job_id, None)
                    self._finished_at.pop(job_id, None)
                    break
            else:
                return


def error_code_for(error: Exception) -> str:
    from app.knowledge.ingestion.metadata import UnsupportedDocumentTypeError
    from app.knowledge.ingestion.pdf import PdfExtractionError
    from app.knowledge.ingestion.service import EmptyDocumentError

    if isinstance(error, UnsupportedDocumentTypeError):
        return "unsupported_document_type"
    if isinstance(error, PdfExtractionError):
        return "pdf_extraction_failed"
    if isinstance(error, EmptyDocumentError):
        return "empty_document"
    if isinstance(error, ValueError):
        return "invalid_document"
    return "ingestion_failed"

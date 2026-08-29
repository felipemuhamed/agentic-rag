from threading import Event

import pytest

from app.uploads.jobs import UploadJobManager, error_code_for
from app.uploads.models import JobNotFoundError, JobQueueFullError


def submit(manager: UploadJobManager, work, *, principal="anonymous", tenant=None):
    return manager.submit(
        kind="document",
        filename="guia.md",
        principal=principal,
        tenant_id=tenant,
        work=work,
    )


def wait_for(manager: UploadJobManager, job_id: str, **owner) -> dict:
    manager.shutdown()
    return manager.get(job_id, **owner)


def test_successful_work_is_reported_with_its_result() -> None:
    manager = UploadJobManager()

    job = submit(manager, lambda: {"documentId": "abc", "chunksWritten": 3})
    finished = wait_for(manager, job.job_id, principal="anonymous", tenant_id=None)

    assert job.status == "queued"
    assert finished.status == "succeeded"
    assert finished.result == {"documentId": "abc", "chunksWritten": 3}
    assert finished.error is None


def test_failure_is_reported_with_a_stable_error_code() -> None:
    manager = UploadJobManager()

    def failing() -> dict:
        raise ValueError("documento invalido")

    job = submit(manager, failing)
    finished = wait_for(manager, job.job_id, principal="anonymous", tenant_id=None)

    assert finished.status == "failed"
    assert finished.error == "documento invalido"
    assert finished.error_code == "invalid_document"


def test_jobs_are_only_visible_to_the_principal_and_tenant_that_created_them() -> None:
    manager = UploadJobManager()

    job = submit(manager, lambda: {}, principal="frontend", tenant="public")
    manager.shutdown()

    assert manager.get(job.job_id, principal="frontend", tenant_id="public")
    with pytest.raises(JobNotFoundError):
        manager.get(job.job_id, principal="frontend", tenant_id="privado")
    with pytest.raises(JobNotFoundError):
        manager.get(job.job_id, principal="outro", tenant_id="public")


def test_queue_rejects_new_work_beyond_the_pending_limit() -> None:
    manager = UploadJobManager(max_pending=1)
    release = Event()

    submit(manager, release.wait)
    try:
        with pytest.raises(JobQueueFullError):
            submit(manager, lambda: {})
    finally:
        release.set()
        manager.shutdown()


def test_finished_jobs_are_pruned_after_the_retention_window() -> None:
    now = [0.0]
    manager = UploadJobManager(retention_seconds=10.0, clock=lambda: now[0])

    job = submit(manager, lambda: {})
    manager.shutdown()
    assert manager.get(job.job_id, principal="anonymous", tenant_id=None)

    now[0] = 100.0
    assert manager.list(principal="anonymous", tenant_id=None) == []
    with pytest.raises(JobNotFoundError):
        manager.get(job.job_id, principal="anonymous", tenant_id=None)


def test_error_codes_map_the_known_ingestion_failures() -> None:
    from app.knowledge.ingestion.metadata import UnsupportedDocumentTypeError
    from app.knowledge.ingestion.pdf import PdfExtractionError
    from app.knowledge.ingestion.service import EmptyDocumentError

    assert error_code_for(UnsupportedDocumentTypeError("x")) == (
        "unsupported_document_type"
    )
    assert error_code_for(PdfExtractionError("x")) == "pdf_extraction_failed"
    assert error_code_for(EmptyDocumentError("x")) == "empty_document"
    assert error_code_for(RuntimeError("x")) == "ingestion_failed"

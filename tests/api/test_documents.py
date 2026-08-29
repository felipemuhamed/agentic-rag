import asyncio
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx

from app.knowledge.models import IndexedDocument
from app.main import create_app
from app.settings import Settings
from app.uploads.service import DocumentNotFoundError


API_KEY = "0123456789abcdef0123456789abcdef"


class RecordingLibrary:
    def __init__(self, *, documents=None, images=None, removal=None):
        self.saved: dict[Path, dict[str, object]] = {}
        self.documents_ingested: list[dict[str, object]] = []
        self.images_ingested: list[dict[str, object]] = []
        self.listed: list[str | None] = []
        self.deleted: list[tuple[str, str | None]] = []
        self._documents = documents or []
        self._images = images or []
        self._removal = removal

    def save_document(self, *, filename, data, tenant_id):
        path = Path("/uploads") / (tenant_id or "_shared") / filename
        self.saved[path] = {
            "filename": filename,
            "data": data,
            "tenant_id": tenant_id,
        }
        return path

    def ingest_saved_document(self, path, *, tenant_id, equipment_id=""):
        self.documents_ingested.append(
            {**self.saved[path], "equipment_id": equipment_id}
        )
        return {"documentId": "doc-1", "chunksWritten": 4, "filename": path.name}

    def save_reference_image(self, *, filename, data, reference):
        path = Path("/uploads") / reference.equipment_id / filename
        self.saved[path] = {
            "filename": filename,
            "data": data,
            "reference": reference,
        }
        return path

    def ingest_saved_reference_image(self, path, *, reference):
        self.images_ingested.append(self.saved[path])
        return {"documentId": "img-1", "equipmentId": reference.equipment_id}

    def list_documents(self, *, tenant_id):
        self.listed.append(tenant_id)
        return self._documents

    def list_reference_images(self, *, tenant_id):
        self.listed.append(tenant_id)
        return self._images

    def delete_document(self, document_id, *, tenant_id):
        self.deleted.append((document_id, tenant_id))
        if self._removal is None:
            raise DocumentNotFoundError(document_id)
        return self._removal

    def delete_reference_image(self, document_id, *, tenant_id):
        return self.delete_document(document_id, tenant_id=tenant_id)


def build_app(library: RecordingLibrary, config: Settings | None = None):
    application = create_app(
        config or Settings(app_environment="test", _env_file=None),
        runtime=SimpleNamespace(
            document_library=lambda: library,
            adapter=lambda: None,
            ready=True,
            close=lambda: None,
        ),
    )
    return application


async def call(application, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, url, **kwargs)


def drain(application) -> None:
    application.state.upload_jobs.shutdown()


def test_uploaded_document_is_accepted_and_reported_by_its_job() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    accepted = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"# Guia\n\nConteudo.", "text/markdown")},
            data={"tenantId": "public", "equipmentId": "OLT-9000"},
        )
    )
    drain(application)
    job_id = accepted.json()["jobId"]
    status = asyncio.run(
        call(application, "GET", f"/documents/jobs/{job_id}?tenantId=public")
    )

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert accepted.json()["statusUrl"] == f"/documents/jobs/{job_id}"
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"
    assert status.json()["result"] == {
        "documentId": "doc-1",
        "chunksWritten": 4,
        "filename": "guia.md",
    }
    assert library.documents_ingested[0]["tenant_id"] == "public"
    assert library.documents_ingested[0]["equipment_id"] == "OLT-9000"
    assert library.documents_ingested[0]["data"] == b"# Guia\n\nConteudo."


def test_failed_ingestion_is_reported_with_the_error_code() -> None:
    class FailingLibrary(RecordingLibrary):
        def ingest_saved_document(self, path, **kwargs):
            from app.knowledge.ingestion.service import EmptyDocumentError

            raise EmptyDocumentError("Document 'guia.md' is empty")

    application = build_app(FailingLibrary())

    accepted = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"conteudo", "text/markdown")},
        )
    )
    drain(application)
    job_id = accepted.json()["jobId"]
    status = asyncio.run(call(application, "GET", f"/documents/jobs/{job_id}"))

    assert status.json()["status"] == "failed"
    assert status.json()["errorCode"] == "empty_document"
    assert "empty" in status.json()["error"]


def test_pdf_uploads_are_accepted(build_pdf) -> None:
    library = RecordingLibrary()
    application = build_app(library)

    response = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={
                "file": (
                    "manual.pdf",
                    build_pdf(["Procedimento de troca"]),
                    "application/pdf",
                )
            },
        )
    )
    drain(application)

    assert response.status_code == 202
    assert library.documents_ingested[0]["filename"] == "manual.pdf"


def test_unsupported_extension_and_empty_file_are_rejected() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    unsupported = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("planilha.xlsx", b"conteudo", "application/vnd.ms-excel")},
        )
    )
    empty = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"", "text/markdown")},
        )
    )
    drain(application)

    assert unsupported.status_code == 415
    assert empty.status_code == 422
    assert library.documents_ingested == []


def test_files_above_the_limit_are_rejected_before_ingestion() -> None:
    library = RecordingLibrary()
    application = build_app(
        library,
        Settings(app_environment="test", upload_max_bytes=1_024, _env_file=None),
    )

    response = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"x" * 2_048, "text/markdown")},
        )
    )
    drain(application)

    assert response.status_code == 413
    assert library.documents_ingested == []


def test_reference_image_upload_carries_the_equipment_metadata() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    response = asyncio.run(
        call(
            application,
            "POST",
            "/documents/images",
            files={
                "file": ("frontal.png", b"\x89PNG\r\n\x1a\ndados", "image/png"),
            },
            data={
                "equipmentId": "OLT-9000",
                "equipmentName": "OLT proprietaria",
                "manufacturer": "CPQD",
                "model": "X9000",
                "description": "Chassi frontal",
                "tenantId": "public",
            },
        )
    )
    drain(application)
    reference = library.images_ingested[0]["reference"]

    assert response.status_code == 202
    assert reference.equipment_id == "OLT-9000"
    assert reference.equipment_name == "OLT proprietaria"
    assert reference.manufacturer == "CPQD"
    assert reference.tenant_id == "public"


def test_image_bytes_must_match_the_declared_extension() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    response = asyncio.run(
        call(
            application,
            "POST",
            "/documents/images",
            files={"file": ("frontal.png", b"nao e um png", "image/png")},
            data={"equipmentId": "OLT-9000", "equipmentName": "OLT"},
        )
    )
    drain(application)

    assert response.status_code == 422
    assert library.images_ingested == []


def test_documents_are_listed_with_pagination_and_tenant_filter() -> None:
    documents = [
        IndexedDocument(
            document_id=f"doc-{index}",
            title=f"Guia {index}",
            source=f"/data/uploads/public/documents/guia-{index}.md",
            media_type="text/markdown",
            chunk_count=index + 1,
            tenant_id="public",
        )
        for index in range(3)
    ]
    library = RecordingLibrary(documents=documents)
    application = build_app(library)

    response = asyncio.run(
        call(application, "GET", "/documents?tenantId=public&limit=2&offset=1")
    )
    drain(application)
    body = response.json()

    assert response.status_code == 200
    assert body["total"] == 3
    assert [item["documentId"] for item in body["documents"]] == ["doc-1", "doc-2"]
    assert body["documents"][0]["chunkCount"] == 2
    assert library.listed == ["public"]


def test_deleting_a_document_reports_what_was_removed() -> None:
    from app.uploads.service import DocumentRemoval

    library = RecordingLibrary(
        removal=DocumentRemoval(
            document_id="doc-1",
            chunks_deleted=7,
            file_removed=True,
        )
    )
    application = build_app(library)

    removed = asyncio.run(
        call(application, "DELETE", "/documents/doc-1?tenantId=public")
    )
    drain(application)

    assert removed.status_code == 200
    assert removed.json() == {
        "documentId": "doc-1",
        "chunksDeleted": 7,
        "fileRemoved": True,
    }
    assert library.deleted == [("doc-1", "public")]


def test_deleting_an_unknown_document_answers_404() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    response = asyncio.run(call(application, "DELETE", "/documents/inexistente"))
    drain(application)

    assert response.status_code == 404


def test_jobs_are_not_visible_to_another_tenant() -> None:
    library = RecordingLibrary()
    application = build_app(library)

    accepted = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"conteudo", "text/markdown")},
            data={"tenantId": "public"},
        )
    )
    drain(application)
    job_id = accepted.json()["jobId"]
    other = asyncio.run(
        call(application, "GET", f"/documents/jobs/{job_id}?tenantId=privado")
    )

    assert other.status_code == 404


def test_uploads_require_a_valid_api_key_when_authentication_is_enabled() -> None:
    config = Settings(
        app_environment="test",
        auth_enabled=True,
        api_key_policies=json.dumps(
            [{"key": API_KEY, "principal": "frontend", "tenants": ["public"]}]
        ),
        _env_file=None,
    )
    library = RecordingLibrary()
    application = build_app(library, config)
    upload = {"files": {"file": ("guia.md", b"conteudo", "text/markdown")}}

    anonymous = asyncio.run(call(application, "POST", "/documents", **upload))
    wrong_tenant = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            headers={"X-API-Key": API_KEY},
            data={"tenantId": "privado"},
            **upload,
        )
    )
    authorized = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            headers={"X-API-Key": API_KEY},
            data={"tenantId": "public"},
            **upload,
        )
    )
    drain(application)

    assert anonymous.status_code == 401
    assert wrong_tenant.status_code == 403
    assert authorized.status_code == 202
    assert library.documents_ingested[0]["tenant_id"] == "public"


def test_endpoints_are_absent_when_uploads_are_disabled() -> None:
    library = RecordingLibrary()
    application = build_app(
        library,
        Settings(app_environment="test", upload_enabled=False, _env_file=None),
    )

    upload = asyncio.run(
        call(
            application,
            "POST",
            "/documents",
            files={"file": ("guia.md", b"conteudo", "text/markdown")},
        )
    )
    listing = asyncio.run(call(application, "GET", "/documents"))
    drain(application)

    assert upload.status_code == 404
    assert listing.status_code == 404
    assert library.documents_ingested == []


def test_uploads_are_rejected_while_the_queue_is_full() -> None:
    release = Event()

    class BlockingLibrary(RecordingLibrary):
        def ingest_saved_document(self, path, **kwargs):
            release.wait(timeout=5)
            return super().ingest_saved_document(path, **kwargs)

    library = BlockingLibrary()
    application = build_app(
        library,
        Settings(
            app_environment="test",
            upload_max_pending_jobs=1,
            _env_file=None,
        ),
    )
    upload = {"files": {"file": ("guia.md", b"conteudo", "text/markdown")}}

    try:
        first = asyncio.run(call(application, "POST", "/documents", **upload))
        second = asyncio.run(call(application, "POST", "/documents", **upload))
    finally:
        release.set()
        drain(application)

    assert first.status_code == 202
    assert second.status_code == 503
    assert second.headers["retry-after"] == "30"
    assert len(library.documents_ingested) == 1

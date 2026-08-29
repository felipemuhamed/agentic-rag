import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from app.knowledge.vectorstore.chroma import ChromaVectorStore
from app.main import create_app
from app.settings import Settings
from app.uploads.service import DocumentLibraryService
from app.uploads.storage import UploadStorage


class DeterministicEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class FakeVisualEmbeddingProvider:
    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return [[float(len(image)), 0.5, 0.25] for image in images]


def build_app(tmp_path: Path):
    store = ChromaVectorStore(tmp_path / "chroma", "knowledge_e2e")
    library = DocumentLibraryService(
        storage=UploadStorage(tmp_path / "uploads"),
        knowledge_store_factory=lambda: store,
        embedding_provider_factory=DeterministicEmbeddingProvider,
        visual_store_factory=lambda: store,
        visual_embedding_provider_factory=FakeVisualEmbeddingProvider,
        chunk_size=60,
        chunk_overlap=10,
    )
    application = create_app(
        Settings(
            app_environment="test",
            upload_dir=str(tmp_path / "uploads"),
            _env_file=None,
        ),
        runtime=SimpleNamespace(
            document_library=lambda: library,
            adapter=lambda: None,
            ready=True,
            close=lambda: None,
        ),
    )
    return application, store


async def request(application, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, url, **kwargs)


def test_pdf_travels_from_the_endpoint_to_chroma_and_back(
    tmp_path: Path,
    build_pdf,
) -> None:
    application, store = build_app(tmp_path)
    pdf = build_pdf(
        [
            "Procedimento de troca do modulo optico da porta 3",
            "Alarme critico exige inspecao do chassi frontal",
        ]
    )

    accepted = asyncio.run(
        request(
            application,
            "POST",
            "/documents",
            files={"file": ("manual-olt.pdf", pdf, "application/pdf")},
            data={"tenantId": "public", "equipmentId": "OLT-9000"},
        )
    )
    application.state.upload_jobs.shutdown()
    job = asyncio.run(
        request(
            application,
            "GET",
            f"/documents/jobs/{accepted.json()['jobId']}?tenantId=public",
        )
    ).json()
    listing = asyncio.run(
        request(application, "GET", "/documents?tenantId=public")
    ).json()
    document_id = job["result"]["documentId"]
    chunks = store.get_document_chunks(document_id)

    assert accepted.status_code == 202
    assert job["status"] == "succeeded"
    assert job["result"]["chunksWritten"] == len(chunks)
    assert Path(job["result"]["source"]).exists()
    assert {chunk["metadata"]["section"] for chunk in chunks} == {
        "Pagina 1",
        "Pagina 2",
    }
    assert chunks[0]["metadata"]["equipment_id"] == "OLT-9000"
    assert listing["total"] == 1
    assert listing["documents"][0]["documentId"] == document_id
    assert listing["documents"][0]["mediaType"] == "application/pdf"
    assert listing["documents"][0]["chunkCount"] == len(chunks)

    removed = asyncio.run(
        request(
            application,
            "DELETE",
            f"/documents/{document_id}?tenantId=public",
        )
    )
    remaining = asyncio.run(
        request(application, "GET", "/documents?tenantId=public")
    ).json()

    assert removed.status_code == 200
    assert removed.json()["chunksDeleted"] == len(chunks)
    assert removed.json()["fileRemoved"] is True
    assert not Path(job["result"]["source"]).exists()
    assert remaining["total"] == 0
    assert store.count() == 0


def test_documents_from_another_tenant_are_invisible_and_undeletable(
    tmp_path: Path,
) -> None:
    application, _ = build_app(tmp_path)

    accepted = asyncio.run(
        request(
            application,
            "POST",
            "/documents",
            files={
                "file": (
                    "privado.md",
                    b"# Privado\n\nConteudo restrito ao tenant.",
                    "text/markdown",
                )
            },
            data={"tenantId": "privado"},
        )
    )
    application.state.upload_jobs.shutdown()
    job = asyncio.run(
        request(
            application,
            "GET",
            f"/documents/jobs/{accepted.json()['jobId']}?tenantId=privado",
        )
    ).json()
    document_id = job["result"]["documentId"]

    other_listing = asyncio.run(
        request(application, "GET", "/documents?tenantId=public")
    ).json()
    other_delete = asyncio.run(
        request(application, "DELETE", f"/documents/{document_id}?tenantId=public")
    )
    own_listing = asyncio.run(
        request(application, "GET", "/documents?tenantId=privado")
    ).json()

    assert job["status"] == "succeeded"
    assert other_listing["total"] == 0
    assert other_delete.status_code == 404
    assert own_listing["total"] == 1

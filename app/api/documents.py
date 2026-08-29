import logging
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field

from app.api.security import acquire_rate_limit, authorize_request
from app.knowledge.ingestion.metadata import SUPPORTED_MEDIA_TYPES
from app.knowledge.models import IndexedDocument
from app.multimodal.models import matches_image_signature
from app.security.access import AccessContext
from app.uploads.jobs import UploadJobManager
from app.uploads.models import (
    JobNotFoundError,
    JobQueueFullError,
    UploadJob,
)
from app.uploads.service import DocumentLibraryService, DocumentNotFoundError
from app.uploads.storage import UnsafeFilenameError, safe_filename
from app.visual_catalog.ingestion import (
    SUPPORTED_REFERENCE_IMAGES,
    VisualReference,
)


router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

_READ_CHUNK_BYTES = 1_048_576
_MAX_IDENTIFIER_LENGTH = 128


class UploadAcceptedResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: str
    status_url: str = Field(serialization_alias="statusUrl")

    model_config = ConfigDict(populate_by_name=True)


class JobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    kind: str
    filename: str
    status: str
    created_at: str = Field(serialization_alias="createdAt")
    updated_at: str = Field(serialization_alias="updatedAt")
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = Field(default=None, serialization_alias="errorCode")

    model_config = ConfigDict(populate_by_name=True)


class DocumentResponse(BaseModel):
    document_id: str = Field(serialization_alias="documentId")
    title: str
    source: str
    media_type: str = Field(serialization_alias="mediaType")
    chunk_count: int = Field(serialization_alias="chunkCount")
    created_at: str = Field(serialization_alias="createdAt")
    updated_at: str = Field(serialization_alias="updatedAt")
    tenant_id: str = Field(serialization_alias="tenantId")
    equipment_id: str = Field(serialization_alias="equipmentId")

    model_config = ConfigDict(populate_by_name=True)


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class DeletionResponse(BaseModel):
    document_id: str = Field(serialization_alias="documentId")
    chunks_deleted: int = Field(serialization_alias="chunksDeleted")
    file_removed: bool = Field(serialization_alias="fileRemoved")

    model_config = ConfigDict(populate_by_name=True)


@router.post(
    "",
    status_code=202,
    response_model=UploadAcceptedResponse,
    response_model_by_alias=True,
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    tenant_id: Annotated[str | None, Form(alias="tenantId")] = None,
    equipment_id: Annotated[str | None, Form(alias="equipmentId")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> UploadAcceptedResponse:
    """Accept a TXT, Markdown or PDF file and index it in the background."""
    config = _enabled_config(request)
    access = authorize_request(
        request,
        api_key=api_key,
        tenant_id=_optional_identifier(tenant_id, field="tenantId"),
    )
    equipment = _optional_identifier(equipment_id, field="equipmentId") or ""
    filename = _validated_filename(file.filename, SUPPORTED_MEDIA_TYPES)
    lease = acquire_rate_limit(request, access)
    try:
        data = await _read_upload(file, max_bytes=config.upload_max_bytes)
        library = _library(request)
        # Saving here, before queueing, keeps the pending queue from holding
        # every uploaded file in memory while the single worker catches up.
        path = await anyio.to_thread.run_sync(
            lambda: library.save_document(
                filename=filename,
                data=data,
                tenant_id=access.tenant_id,
            )
        )
        job = _submit(
            request,
            kind="document",
            filename=filename,
            access=access,
            work=lambda: library.ingest_saved_document(
                path,
                tenant_id=access.tenant_id,
                equipment_id=equipment,
            ),
        )
    finally:
        lease.release()
    return _accepted(job)


@router.post(
    "/images",
    status_code=202,
    response_model=UploadAcceptedResponse,
    response_model_by_alias=True,
)
async def upload_reference_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    equipment_id: Annotated[str, Form(alias="equipmentId")],
    equipment_name: Annotated[str, Form(alias="equipmentName")],
    manufacturer: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    tenant_id: Annotated[str | None, Form(alias="tenantId")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> UploadAcceptedResponse:
    """Accept an equipment reference photo for the proprietary visual catalog."""
    config = _enabled_config(request)
    if not config.visual_catalog_enabled:
        raise HTTPException(status_code=422, detail="visual catalog is disabled")
    access = authorize_request(
        request,
        api_key=api_key,
        tenant_id=_optional_identifier(tenant_id, field="tenantId"),
    )
    filename = _validated_filename(file.filename, SUPPORTED_REFERENCE_IMAGES)
    try:
        reference = VisualReference(
            equipment_id=_required_identifier(equipment_id, field="equipmentId"),
            equipment_name=equipment_name,
            manufacturer=manufacturer,
            model=model,
            description=description,
            tenant_id=access.tenant_id or "",
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    lease = acquire_rate_limit(request, access)
    try:
        data = await _read_upload(
            file,
            max_bytes=config.multimodal_max_image_bytes,
        )
        mime_type = SUPPORTED_REFERENCE_IMAGES[Path(filename).suffix.lower()]
        if not matches_image_signature(data, mime_type):
            raise HTTPException(
                status_code=422,
                detail="image bytes do not match the file extension",
            )
        library = _library(request)
        path = await anyio.to_thread.run_sync(
            lambda: library.save_reference_image(
                filename=filename,
                data=data,
                reference=reference,
            )
        )
        job = _submit(
            request,
            kind="image",
            filename=filename,
            access=access,
            work=lambda: library.ingest_saved_reference_image(
                path,
                reference=reference,
            ),
        )
    finally:
        lease.release()
    return _accepted(job)


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    response_model_by_alias=True,
)
async def read_job(
    request: Request,
    job_id: str,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> JobResponse:
    _enabled_config(request)
    access = authorize_request(
        request,
        api_key=api_key,
        tenant_id=_optional_identifier(tenant_id, field="tenantId"),
    )
    try:
        job = _jobs(request).get(
            job_id,
            principal=access.principal,
            tenant_id=access.tenant_id,
        )
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail="job not found") from error
    return _job_response(job)


@router.get(
    "/jobs",
    response_model=list[JobResponse],
    response_model_by_alias=True,
)
async def list_jobs(
    request: Request,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> list[JobResponse]:
    _enabled_config(request)
    access = authorize_request(
        request,
        api_key=api_key,
        tenant_id=_optional_identifier(tenant_id, field="tenantId"),
    )
    jobs = _jobs(request).list(
        principal=access.principal,
        tenant_id=access.tenant_id,
        limit=limit,
    )
    return [_job_response(job) for job in jobs]


@router.get(
    "",
    response_model=DocumentListResponse,
    response_model_by_alias=True,
)
async def list_documents(
    request: Request,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> DocumentListResponse:
    access = _authorized_access(request, api_key=api_key, tenant_id=tenant_id)
    library = _library(request)
    documents = await anyio.to_thread.run_sync(
        lambda: library.list_documents(tenant_id=access.tenant_id)
    )
    return _document_page(documents, limit=limit, offset=offset)


@router.get(
    "/images",
    response_model=DocumentListResponse,
    response_model_by_alias=True,
)
async def list_reference_images(
    request: Request,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> DocumentListResponse:
    access = _authorized_access(request, api_key=api_key, tenant_id=tenant_id)
    library = _library(request)
    documents = await anyio.to_thread.run_sync(
        lambda: library.list_reference_images(tenant_id=access.tenant_id)
    )
    return _document_page(documents, limit=limit, offset=offset)


@router.delete(
    "/images/{document_id}",
    response_model=DeletionResponse,
    response_model_by_alias=True,
)
async def delete_reference_image(
    request: Request,
    document_id: str,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> DeletionResponse:
    access = _authorized_access(request, api_key=api_key, tenant_id=tenant_id)
    library = _library(request)
    return await _delete(
        lambda: library.delete_reference_image(
            document_id,
            tenant_id=access.tenant_id,
        )
    )


@router.delete(
    "/{document_id}",
    response_model=DeletionResponse,
    response_model_by_alias=True,
)
async def delete_document(
    request: Request,
    document_id: str,
    tenant_id: Annotated[str | None, Query(alias="tenantId")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> DeletionResponse:
    access = _authorized_access(request, api_key=api_key, tenant_id=tenant_id)
    library = _library(request)
    return await _delete(
        lambda: library.delete_document(document_id, tenant_id=access.tenant_id)
    )


def _authorized_access(
    request: Request,
    *,
    api_key: str | None,
    tenant_id: str | None,
) -> AccessContext:
    _enabled_config(request)
    return authorize_request(
        request,
        api_key=api_key,
        tenant_id=_optional_identifier(tenant_id, field="tenantId"),
    )


async def _delete(operation) -> DeletionResponse:
    try:
        removal = await anyio.to_thread.run_sync(operation)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail="document not found") from error
    return DeletionResponse(
        document_id=removal.document_id,
        chunks_deleted=removal.chunks_deleted,
        file_removed=removal.file_removed,
    )


def _enabled_config(request: Request):
    config = request.app.state.config
    if not config.upload_enabled:
        raise HTTPException(status_code=404, detail="document uploads are disabled")
    return config


def _library(request: Request) -> DocumentLibraryService:
    return request.app.state.rag_runtime.document_library()


def _jobs(request: Request) -> UploadJobManager:
    return request.app.state.upload_jobs


def _submit(
    request: Request,
    *,
    kind: str,
    filename: str,
    access: AccessContext,
    work,
) -> UploadJob:
    try:
        return _jobs(request).submit(
            kind=kind,
            filename=filename,
            principal=access.principal,
            tenant_id=access.tenant_id,
            work=work,
        )
    except JobQueueFullError as error:
        raise HTTPException(
            status_code=503,
            detail="ingestion queue is full",
            headers={"Retry-After": "30"},
        ) from error


async def _read_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    """Stream the upload into memory, stopping as soon as it exceeds the limit."""
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"the file must have at most {max_bytes} bytes",
            )
    if not buffer:
        raise HTTPException(status_code=422, detail="the file must not be empty")
    return bytes(buffer)


def _validated_filename(filename: str | None, supported: dict[str, str]) -> str:
    if not filename or not filename.strip():
        raise HTTPException(status_code=422, detail="filename is required")
    try:
        normalized = safe_filename(filename)
    except UnsafeFilenameError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    suffix = Path(normalized).suffix.lower()
    if suffix not in supported:
        accepted = ", ".join(sorted(supported))
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{suffix or normalized}'; use: {accepted}",
        )
    return normalized


def _required_identifier(value: str, *, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{field} must not be blank")
    if len(normalized) > _MAX_IDENTIFIER_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"{field} must have at most {_MAX_IDENTIFIER_LENGTH} characters",
        )
    return normalized


def _optional_identifier(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return _required_identifier(normalized, field=field)


def _accepted(job: UploadJob) -> UploadAcceptedResponse:
    return UploadAcceptedResponse(
        job_id=job.job_id,
        status=job.status,
        status_url=f"/documents/jobs/{job.job_id}",
    )


def _job_response(job: UploadJob) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        kind=job.kind,
        filename=job.filename,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
        error_code=job.error_code,
    )


def _document_page(
    documents: list[IndexedDocument],
    *,
    limit: int,
    offset: int,
) -> DocumentListResponse:
    page = documents[offset : offset + limit]
    return DocumentListResponse(
        documents=[
            DocumentResponse(
                document_id=document.document_id,
                title=document.title,
                source=document.source,
                media_type=document.media_type,
                chunk_count=document.chunk_count,
                created_at=document.created_at,
                updated_at=document.updated_at,
                tenant_id=document.tenant_id,
                equipment_id=document.equipment_id,
            )
            for document in page
        ],
        total=len(documents),
        limit=limit,
        offset=offset,
    )

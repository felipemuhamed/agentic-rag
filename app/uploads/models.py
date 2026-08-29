from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


JobKind = Literal["document", "image"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class JobNotFoundError(LookupError):
    pass


class JobQueueFullError(RuntimeError):
    pass


@dataclass(slots=True)
class UploadJob:
    job_id: str
    kind: JobKind
    filename: str
    principal: str
    tenant_id: str | None
    status: JobStatus = "queued"
    created_at: str = field(default_factory=lambda: utc_now_iso())
    updated_at: str = field(default_factory=lambda: utc_now_iso())
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None

    def owned_by(self, *, principal: str, tenant_id: str | None) -> bool:
        return self.principal == principal and self.tenant_id == tenant_id

    def snapshot(self) -> "UploadJob":
        return UploadJob(
            job_id=self.job_id,
            kind=self.kind,
            filename=self.filename,
            principal=self.principal,
            tenant_id=self.tenant_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            result=dict(self.result) if self.result is not None else None,
            error=self.error,
            error_code=self.error_code,
        )


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()

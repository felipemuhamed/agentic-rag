import re
from hashlib import sha256
from pathlib import Path, PurePosixPath
from uuid import uuid4


MAX_FILENAME_LENGTH = 120
SHARED_TENANT_DIRECTORY = "_shared"

_SAFE_DIRECTORY = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


class UnsafeFilenameError(ValueError):
    pass


class UploadStorage:
    """Persist uploaded files under a deterministic path inside the root.

    The path is deterministic on purpose: `FileMetadataExtractor` derives the
    document ID from the tenant plus the resolved path, so re-uploading a file
    with the same name replaces the previous version instead of duplicating it.
    """

    def __init__(self, root: str | Path):
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def document_path(self, *, tenant_id: str | None, filename: str) -> Path:
        return (
            self._root
            / _directory_name(tenant_id)
            / "documents"
            / safe_filename(filename)
        )

    def reference_image_path(
        self,
        *,
        tenant_id: str | None,
        equipment_id: str,
        filename: str,
    ) -> Path:
        return (
            self._root
            / _directory_name(tenant_id)
            / "images"
            / _directory_name(equipment_id)
            / safe_filename(filename)
        )

    def save(self, path: Path, data: bytes) -> Path:
        """Write the file atomically so a reader never sees a partial upload."""
        if not self.contains(path):
            raise UnsafeFilenameError("upload path escapes the storage root")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def remove(self, path: str | Path) -> bool:
        """Delete a stored file, ignoring anything outside the storage root."""
        candidate = Path(path)
        if not candidate.is_absolute() or not self.contains(candidate):
            return False
        try:
            candidate.unlink()
        except FileNotFoundError:
            return False
        return True

    def contains(self, path: str | Path) -> bool:
        candidate = Path(path)
        try:
            resolved = candidate.resolve()
        except OSError:
            return False
        return resolved == self._root or self._root in resolved.parents


def safe_filename(filename: str) -> str:
    """Reduce a client-supplied name to a single safe path component."""
    normalized = filename.strip().replace("\\", "/")
    name = PurePosixPath(normalized).name.strip()
    if not name or name in {".", ".."}:
        raise UnsafeFilenameError("filename must not be blank")

    suffix = Path(name).suffix.lower()
    stem = _UNSAFE_CHARACTERS.sub("_", name[: len(name) - len(suffix)]).strip("._-")
    if not stem:
        stem = "documento"
    stem = stem[: MAX_FILENAME_LENGTH - len(suffix)]
    return f"{stem}{suffix}"


def _directory_name(value: str | None) -> str:
    """Map an arbitrary identifier to a collision-free directory name."""
    normalized = (value or "").strip()
    if not normalized:
        return SHARED_TENANT_DIRECTORY
    if _SAFE_DIRECTORY.fullmatch(normalized):
        return normalized
    digest = sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"id-{digest}"

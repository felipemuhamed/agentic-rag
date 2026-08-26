import base64
import binascii
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


ImageMimeType = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
]

SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
ABSOLUTE_MAX_IMAGES = 5
ABSOLUTE_MAX_BASE64_CHARACTERS = 28_000_000


class ImageInput(TypedDict):
    name: str
    mime_type: ImageMimeType
    data: str
    sha256: str


class ChatImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(default="imagem", min_length=1, max_length=255)
    mime_type: ImageMimeType = Field(alias="mimeType")
    data: str = Field(min_length=1, max_length=ABSOLUTE_MAX_BASE64_CHARACTERS)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        name = PurePosixPath(normalized).name
        if not name or name in {".", ".."}:
            raise ValueError("image name must not be blank")
        return name

    @field_validator("data")
    @classmethod
    def normalize_data(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("data:"):
            raise ValueError(
                "image data must contain raw base64, not a data URL"
            )
        return normalized


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(InvalidImageError):
    pass


def validate_image_inputs(
    images: list[ChatImage],
    *,
    max_images: int,
    max_image_bytes: int,
) -> list[ImageInput]:
    if len(images) > max_images:
        raise InvalidImageError(
            f"at most {max_images} image(s) are allowed per request"
        )

    validated: list[ImageInput] = []
    for image in images:
        try:
            decoded = base64.b64decode(image.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise InvalidImageError("image data is not valid base64") from error
        if not decoded:
            raise InvalidImageError("decoded image must not be empty")
        if len(decoded) > max_image_bytes:
            raise ImageTooLargeError(
                f"each image must have at most {max_image_bytes} bytes"
            )
        if not _matches_signature(decoded, image.mime_type):
            raise InvalidImageError(
                "image bytes do not match the declared MIME type"
            )
        validated.append(
            {
                "name": image.name,
                "mime_type": image.mime_type,
                "data": image.data,
                "sha256": sha256(decoded).hexdigest(),
            }
        )
    return validated


def _matches_signature(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if mime_type == "image/webp":
        return (
            len(data) >= 12
            and data.startswith(b"RIFF")
            and data[8:12] == b"WEBP"
        )
    return False

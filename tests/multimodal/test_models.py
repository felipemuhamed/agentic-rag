import base64

import pytest

from app.multimodal.models import (
    ChatImage,
    ImageTooLargeError,
    InvalidImageError,
    validate_image_inputs,
)


def make_png_data(extra: bytes = b"") -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + extra).decode()


def test_image_payload_is_validated_and_normalized() -> None:
    payload = ChatImage(
        name="../camera/painel.png",
        mimeType="image/png",
        data=make_png_data(),
    )

    images = validate_image_inputs(
        [payload],
        max_images=1,
        max_image_bytes=1024,
    )

    assert images[0]["name"] == "painel.png"
    assert images[0]["mime_type"] == "image/png"
    assert len(images[0]["sha256"]) == 64


def test_image_payload_rejects_invalid_base64_and_signature() -> None:
    with pytest.raises(InvalidImageError, match="valid base64"):
        validate_image_inputs(
            [ChatImage(name="x.png", mimeType="image/png", data="@@@@")],
            max_images=1,
            max_image_bytes=1024,
        )

    jpeg_declared_as_png = base64.b64encode(b"\xff\xd8\xffdata").decode()
    with pytest.raises(InvalidImageError, match="MIME"):
        validate_image_inputs(
            [
                ChatImage(
                    name="x.png",
                    mimeType="image/png",
                    data=jpeg_declared_as_png,
                )
            ],
            max_images=1,
            max_image_bytes=1024,
        )


def test_image_payload_enforces_count_and_decoded_size() -> None:
    image = ChatImage(name="x.png", mimeType="image/png", data=make_png_data())
    with pytest.raises(InvalidImageError, match="at most 1"):
        validate_image_inputs(
            [image, image],
            max_images=1,
            max_image_bytes=1024,
        )

    large = ChatImage(
        name="large.png",
        mimeType="image/png",
        data=make_png_data(b"x" * 1024),
    )
    with pytest.raises(ImageTooLargeError, match="at most 1024"):
        validate_image_inputs(
            [large],
            max_images=1,
            max_image_bytes=1024,
        )

from io import BytesIO
from typing import Any, Protocol, Sequence


class VisualEmbeddingProvider(Protocol):
    def embed_images(self, images: Sequence[bytes]) -> list[list[float]]: ...


class SentenceTransformerVisualEmbeddingProvider:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the visual dependencies with "
                "`python -m pip install -e '.[knowledge]'`"
            ) from error
        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name, device=device)

    def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        if not images:
            return []
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as error:
            raise RuntimeError(
                "Install Pillow to process reference images"
            ) from error

        decoded = []
        try:
            for image_bytes in images:
                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    decoded.append(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError("image could not be decoded") from error

        embeddings = self._model.encode(
            decoded,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

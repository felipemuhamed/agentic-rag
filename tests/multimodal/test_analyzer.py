from app.multimodal.analyzer import (
    ImageAnalysis,
    LangChainImageAnalyzer,
    image_analysis_document,
)
from app.multimodal.models import ImageInput


class FakeStructuredModel:
    def __init__(self):
        self.calls: list[object] = []
        self.configs: list[object] = []
        self.retry_attempts: list[int] = []

    def with_config(self, config: object):
        self.configs.append(config)
        return self

    def with_retry(self, *, stop_after_attempt: int):
        self.retry_attempts.append(stop_after_attempt)
        return self

    def invoke(self, messages: object) -> ImageAnalysis:
        self.calls.append(messages)
        return ImageAnalysis(
            summary="Equipamento com LED LOS vermelho.",
            observations=["LED vermelho aceso"],
            visible_text=["LOS"],
            identifiers=["ONT-123"],
            uncertainties=["Número de série ilegível"],
            retrieval_query="ONT-123 LOS vermelho procedimento",
        )


class FakeModel:
    def __init__(self):
        self.structured = FakeStructuredModel()
        self.schemas: list[type[ImageAnalysis]] = []

    def with_structured_output(self, schema: type[ImageAnalysis]):
        self.schemas.append(schema)
        return self.structured


def make_image() -> ImageInput:
    return {
        "name": "ont.png",
        "mime_type": "image/png",
        "data": "cG5n",
        "sha256": "1" * 64,
    }


def test_analyzer_sends_text_before_base64_image_and_returns_structure() -> None:
    model = FakeModel()
    analyzer = LangChainImageAnalyzer(model, max_attempts=2)

    analysis = analyzer.analyze("O que significa?", [make_image()])

    assert model.schemas == [ImageAnalysis]
    assert model.structured.retry_attempts == [2]
    human_message = model.structured.calls[0][1]
    assert human_message.content[0]["type"] == "text"
    assert human_message.content[1] == {
        "type": "image",
        "base64": "cG5n",
        "mime_type": "image/png",
    }
    assert analysis.retrieval_query == "ONT-123 LOS vermelho procedimento"


def test_visual_document_is_citable_and_does_not_contain_image_bytes() -> None:
    analysis = ImageAnalysis(
        summary="Resumo visual.",
        observations=[],
        visible_text=["LOS"],
        identifiers=[],
        uncertainties=[],
        retrieval_query="alarme LOS",
    )

    document = image_analysis_document(analysis, [make_image()])

    assert document.chunk_id == "user-image:1111111111111111"
    assert document.metadata["source"] == "user-image:1111111111111111"
    assert document.metadata["title"] == "Imagem enviada: ont.png"
    assert "Resumo visual" in document.content
    assert "cG5n" not in document.content

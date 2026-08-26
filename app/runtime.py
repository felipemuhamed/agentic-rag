from collections.abc import Callable
from threading import Lock
from typing import Any

from app.graph.factory import create_rag_graph
from app.multimodal.analyzer import ImageAnalyzer
from app.multimodal.factory import create_image_analyzer
from app.persistence.checkpointer import CheckpointerHandle
from app.settings import Settings
from app.streaming.adapter import LangGraphStreamAdapter, StreamableRagGraph
from app.visual_catalog.factory import create_visual_catalog_retriever
from app.visual_catalog.retrieval import VisualCatalogRetriever


class RagRuntime:
    def __init__(
        self,
        config: Settings,
        *,
        graph_builder: Callable[[Settings, Any], StreamableRagGraph] | None = None,
    ):
        self._config = config
        self._checkpointer = CheckpointerHandle(config)
        self._graph_builder = graph_builder or _build_graph
        self._graph: StreamableRagGraph | None = None
        self._adapter: LangGraphStreamAdapter | None = None
        self._image_analyzer: ImageAnalyzer | None = None
        self._visual_catalog_retriever: VisualCatalogRetriever | None = None
        self._lock = Lock()
        self._last_error: str | None = None

    def graph(self) -> StreamableRagGraph:
        with self._lock:
            if self._graph is not None:
                return self._graph
            try:
                checkpointer = self._checkpointer.start()
                self._graph = self._graph_builder(self._config, checkpointer)
            except Exception as error:
                self._last_error = type(error).__name__
                raise
            self._last_error = None
            return self._graph

    def adapter(self) -> LangGraphStreamAdapter:
        with self._lock:
            if self._adapter is None:
                self._adapter = LangGraphStreamAdapter(
                    graph_factory=self.graph,
                    image_analyzer_factory=self.image_analyzer,
                    visual_catalog_retriever_factory=(
                        self.visual_catalog_retriever
                        if self._config.visual_catalog_enabled
                        else None
                    ),
                    max_rag_attempts=self._config.max_rag_attempts,
                    web_search_enabled=self._config.web_search_enabled,
                    max_duration_seconds=(
                        self._config.stream_max_duration_seconds
                    ),
                )
            return self._adapter

    def image_analyzer(self) -> ImageAnalyzer:
        with self._lock:
            if self._image_analyzer is None:
                try:
                    self._image_analyzer = create_image_analyzer(self._config)
                except Exception as error:
                    self._last_error = type(error).__name__
                    raise
                self._last_error = None
            return self._image_analyzer

    def visual_catalog_retriever(self) -> VisualCatalogRetriever:
        with self._lock:
            if self._visual_catalog_retriever is None:
                try:
                    self._visual_catalog_retriever = (
                        create_visual_catalog_retriever(self._config)
                    )
                except Exception as error:
                    self._last_error = type(error).__name__
                    raise
                self._last_error = None
            return self._visual_catalog_retriever

    def close(self) -> None:
        with self._lock:
            self._adapter = None
            self._graph = None
            self._image_analyzer = None
            self._visual_catalog_retriever = None
        self._checkpointer.close()

    @property
    def ready(self) -> bool:
        return self._last_error is None

    @property
    def last_error(self) -> str | None:
        return self._last_error


def _build_graph(config: Settings, checkpointer: Any) -> StreamableRagGraph:
    return create_rag_graph(config, checkpointer=checkpointer)

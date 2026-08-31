from collections.abc import Callable
from threading import RLock
from typing import Any

from app.graph.factory import create_rag_graph
from app.knowledge.embeddings import SentenceTransformerEmbeddingProvider
from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.vectorstore.base import ManagedKnowledgeVectorStore
from app.knowledge.vectorstore.factory import create_knowledge_vector_store
from app.multimodal.analyzer import ImageAnalyzer
from app.multimodal.factory import create_image_analyzer
from app.observability.tracing import Tracing, create_tracing
from app.persistence.checkpointer import CheckpointerHandle
from app.settings import Settings
from app.streaming.adapter import LangGraphStreamAdapter, StreamableRagGraph
from app.uploads.service import DocumentLibraryService
from app.uploads.storage import UploadStorage
from app.visual_catalog.embeddings import VisualEmbeddingProvider
from app.visual_catalog.factory import (
    create_equipment_resolver,
    create_visual_catalog_retriever,
    create_visual_embedding_provider,
    create_visual_vector_store,
)
from app.visual_catalog.retrieval import VisualCatalogRetriever


class RagRuntime:
    """Own the process-wide singletons shared by chat and document uploads.

    Every component is created on first use and reused afterwards: the embedding
    models weigh gigabytes, so retrieval and ingestion must share one instance.
    The lock is reentrant because the accessors call each other while building.
    """

    def __init__(
        self,
        config: Settings,
        *,
        graph_builder: Callable[[Settings, Any], StreamableRagGraph] | None = None,
    ):
        self._config = config
        self._checkpointer = CheckpointerHandle(config)
        self._graph_builder = graph_builder or self._build_graph
        self._graph: StreamableRagGraph | None = None
        self._adapter: LangGraphStreamAdapter | None = None
        self._image_analyzer: ImageAnalyzer | None = None
        self._visual_catalog_retriever: VisualCatalogRetriever | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._visual_embedding_provider: VisualEmbeddingProvider | None = None
        self._knowledge_vector_store: ManagedKnowledgeVectorStore | None = None
        self._visual_vector_store: ManagedKnowledgeVectorStore | None = None
        self._document_library: DocumentLibraryService | None = None
        self._tracing: Tracing | None = None
        self._lock = RLock()
        self._last_error: str | None = None
        # Lazy mode has nothing to wait for, so it starts warm. With warmup
        # enabled, readiness stays closed until every model is in memory.
        self._warm = not config.warmup_on_startup

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
                    equipment_resolver=(
                        create_equipment_resolver(self._config)
                        if self._config.visual_catalog_enabled
                        else None
                    ),
                    tracing=self.tracing(),
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
                self._image_analyzer = self._create(
                    lambda: create_image_analyzer(
                        self._config,
                        callbacks=self.tracing().callbacks(),
                    )
                )
            return self._image_analyzer

    def tracing(self) -> Tracing:
        with self._lock:
            if self._tracing is None:
                # Never routed through _create: a tracing backend that is down
                # must not close readiness, so create_tracing degrades to a
                # no-op instead of raising.
                self._tracing = create_tracing(self._config)
            return self._tracing

    def visual_catalog_retriever(self) -> VisualCatalogRetriever:
        with self._lock:
            if self._visual_catalog_retriever is None:
                self._visual_catalog_retriever = self._create(
                    lambda: create_visual_catalog_retriever(
                        self._config,
                        vector_store=self.visual_vector_store(),
                        embedding_provider_factory=self.visual_embedding_provider,
                    )
                )
            return self._visual_catalog_retriever

    def embedding_provider(self) -> EmbeddingProvider:
        with self._lock:
            if self._embedding_provider is None:
                self._embedding_provider = self._create(
                    lambda: SentenceTransformerEmbeddingProvider(
                        self._config.embedding_model,
                        device=self._config.embedding_device,
                    )
                )
            return self._embedding_provider

    def visual_embedding_provider(self) -> VisualEmbeddingProvider:
        with self._lock:
            if self._visual_embedding_provider is None:
                self._visual_embedding_provider = self._create(
                    lambda: create_visual_embedding_provider(self._config)
                )
            return self._visual_embedding_provider

    def knowledge_vector_store(self) -> ManagedKnowledgeVectorStore:
        with self._lock:
            if self._knowledge_vector_store is None:
                self._knowledge_vector_store = self._create(
                    lambda: create_knowledge_vector_store(self._config)
                )
            return self._knowledge_vector_store

    def visual_vector_store(self) -> ManagedKnowledgeVectorStore:
        with self._lock:
            if self._visual_vector_store is None:
                self._visual_vector_store = self._create(
                    lambda: create_visual_vector_store(self._config)
                )
            return self._visual_vector_store

    def document_library(self) -> DocumentLibraryService:
        with self._lock:
            if self._document_library is None:
                self._document_library = DocumentLibraryService(
                    storage=UploadStorage(self._config.upload_dir),
                    knowledge_store_factory=self.knowledge_vector_store,
                    embedding_provider_factory=self.embedding_provider,
                    visual_store_factory=self.visual_vector_store,
                    visual_embedding_provider_factory=(
                        self.visual_embedding_provider
                    ),
                    chunk_size=self._config.ingestion_chunk_size,
                    chunk_overlap=self._config.ingestion_chunk_overlap,
                )
            return self._document_library

    def warmup(self) -> None:
        """Load every heavy model before the first request arrives.

        The graph pulls in the embedding model and the cross-encoder reranker,
        and the visual provider pulls in CLIP. Together they weigh gigabytes and
        take far longer to load than a stream is allowed to last, so a replica
        that loads them inside the first request spends that request's whole
        deadline on startup work and answers nothing.
        """
        self.tracing()
        self.graph()
        self.image_analyzer()
        if self._config.visual_catalog_enabled:
            self.visual_embedding_provider()
            self.visual_catalog_retriever()
        with self._lock:
            self._warm = True

    def close(self) -> None:
        with self._lock:
            if self._tracing is not None:
                # Buffered spans are lost unless the exporter is drained.
                self._tracing.shutdown()
                self._tracing = None
            self._warm = not self._config.warmup_on_startup
            self._adapter = None
            self._graph = None
            self._image_analyzer = None
            self._visual_catalog_retriever = None
            self._document_library = None
            self._embedding_provider = None
            self._visual_embedding_provider = None
            self._knowledge_vector_store = None
            self._visual_vector_store = None
        self._checkpointer.close()

    @property
    def ready(self) -> bool:
        return self._last_error is None and self._warm

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _build_graph(
        self,
        config: Settings,
        checkpointer: Any,
    ) -> StreamableRagGraph:
        return create_rag_graph(
            config,
            checkpointer=checkpointer,
            vector_store=self.knowledge_vector_store(),
            embedding_provider=self.embedding_provider(),
        )

    def _create(self, factory: Callable[[], Any]) -> Any:
        try:
            created = factory()
        except Exception as error:
            self._last_error = type(error).__name__
            raise
        self._last_error = None
        return created

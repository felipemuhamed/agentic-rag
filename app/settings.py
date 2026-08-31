from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "Agentic RAG"
    app_version: str = "0.1.0"
    app_environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False

    # Configuration used by later roadmap stages. Defaults keep the API
    # bootstrap independent from model and vector-store services.
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1, le=5)
    vision_model: str = ""
    multimodal_enabled: bool = True
    multimodal_max_images: int = Field(default=1, ge=1, le=5)
    multimodal_max_image_bytes: int = Field(
        default=5_000_000,
        ge=1_024,
        le=20_000_000,
    )
    visual_catalog_enabled: bool = True
    visual_collection: str = "equipment_images"
    visual_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    visual_embedding_device: str | None = None
    visual_max_results: int = Field(default=3, ge=1, le=20)
    visual_max_distance: float = Field(default=0.8, ge=0, le=4)

    # Equipment resolution fuses OCR identifiers with visual similarity. The
    # weight decides how much a legible model plate outranks a lookalike photo.
    equipment_identifier_weight: float = Field(default=0.65, ge=0, le=1)
    equipment_min_confidence: float = Field(default=0.55, ge=0, le=1)
    equipment_ambiguity_margin: float = Field(default=0.08, ge=0, le=1)

    warmup_on_startup: bool = True

    # Tracing is opt-in: the stack is six extra containers, and the service has
    # to keep answering when it is not running.
    langfuse_enabled: bool = False
    langfuse_base_url: str = "http://langfuse-web:3000"
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None

    chroma_mode: Literal["http", "persistent"] = "http"
    chroma_collection: str = "knowledge"
    chroma_path: str = "./chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_ssl: bool = False
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    ingestion_chunk_size: int = 700
    ingestion_chunk_overlap: int = 100

    upload_enabled: bool = True
    upload_dir: str = "./data/uploads"
    upload_max_bytes: int = Field(
        default=20_000_000,
        ge=1_024,
        le=200_000_000,
    )
    upload_workers: int = Field(default=1, ge=1, le=8)
    upload_max_pending_jobs: int = Field(default=32, ge=1, le=1_000)
    upload_max_jobs: int = Field(default=200, ge=1, le=10_000)
    upload_job_retention_seconds: float = Field(default=3_600.0, gt=0)

    rag_fetch_k: int = 60
    rag_candidate_k: int = 20
    rag_final_k: int = 5

    rag_lambda_mult: float = 0.65

    max_rag_attempts: int = Field(default=2, ge=1)

    checkpointer_mode: Literal["memory", "sqlite"] = "memory"
    checkpointer_path: str = "./data/checkpoints.sqlite"

    auth_enabled: bool = False
    api_key_policies: SecretStr = SecretStr("")
    rate_limit_enabled: bool = False
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)
    max_concurrent_streams: int = Field(default=4, ge=1)

    metrics_enabled: bool = True
    metrics_api_key: SecretStr = SecretStr("")
    stream_max_duration_seconds: float = Field(default=120.0, gt=0)

    web_search_enabled: bool = False
    web_search_provider: Literal["tavily"] = "tavily"
    web_search_max_results: int = Field(default=5, ge=1, le=20)
    web_search_depth: Literal[
        "basic",
        "advanced",
        "fast",
        "ultra-fast",
    ] = "basic"
    web_search_timeout_seconds: float = Field(default=10.0, gt=0)
    tavily_api_key: SecretStr = SecretStr("")
    tavily_base_url: str = "https://api.tavily.com"
    external_max_attempts: int = Field(default=3, ge=1, le=5)
    external_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)

    knowledge_tenant_id: str = ""

    @model_validator(mode="after")
    def validate_production_controls(self) -> "Settings":
        if self.checkpointer_mode == "sqlite" and not self.checkpointer_path.strip():
            raise ValueError("CHECKPOINTER_PATH must not be blank in sqlite mode")
        if self.upload_enabled and not self.upload_dir.strip():
            raise ValueError("UPLOAD_DIR must not be blank when uploads are enabled")
        if (
            self.web_search_enabled
            and not self.tavily_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "TAVILY_API_KEY must be configured when WEB_SEARCH_ENABLED=true"
            )
        if self.app_environment != "production":
            return self
        if self.checkpointer_mode == "memory":
            raise ValueError("production requires a durable checkpointer")
        if not self.auth_enabled or not self.api_key_policies.get_secret_value().strip():
            raise ValueError("production requires API key authentication policies")
        if not self.rate_limit_enabled:
            raise ValueError("production requires rate limiting")
        if not self.json_logs:
            raise ValueError("production requires JSON logs")
        if self.metrics_enabled and not self.metrics_api_key.get_secret_value().strip():
            raise ValueError("production metrics require METRICS_API_KEY")
        return self


settings = Settings()

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "Agentic RAG"
    app_version: str = "0.1.0"
    app_environment: str = "development"

    # Configuration used by later roadmap stages. Defaults keep the API
    # bootstrap independent from model and vector-store services.
    llm_model: str = ""

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

    rag_fetch_k: int = 60
    rag_candidate_k: int = 20
    rag_final_k: int = 5

    rag_lambda_mult: float = 0.65

    max_rag_attempts: int = Field(default=2, ge=1)


settings = Settings()

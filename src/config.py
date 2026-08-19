"""
Configuration management for the Research Intelligence System
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load the project file explicitly; runtime behavior must not depend on editor injection.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # LLM Configuration
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gemini-3.5-flash", alias="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        alias="GEMINI_BASE_URL",
    )
    ollama_api_url: str = Field(default="http://localhost:11434", alias="OLLAMA_API_URL")
    ollama_model: str = Field(default="llama2", alias="OLLAMA_MODEL")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    
    # Retrieval Configuration
    bm25_k1: float = Field(default=1.5, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, alias="BM25_B")
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    hybrid_alpha: float = Field(default=0.5, alias="HYBRID_ALPHA")
    fusion_strategy: str = Field(default="weighted", alias="FUSION_STRATEGY")
    rrf_k: int = Field(default=60, alias="RRF_K")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, alias="CHUNK_OVERLAP")
    
    # Reranking Configuration
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        alias="RERANKER_MODEL"
    )
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    
    # Storage Configuration
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    documents_dir: Path = Field(default=Path("./data/documents"), alias="DOCUMENTS_DIR")
    indexes_dir: Path = Field(default=Path("./data/indexes"), alias="INDEXES_DIR")
    experiments_dir: Path = Field(default=Path("./data/experiments"), alias="EXPERIMENTS_DIR")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, alias="STREAMLIT_SERVER_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def model_post_init(self, __context) -> None:
        """Create data directories if they don't exist"""
        for directory in [
            self.data_dir,
            self.documents_dir,
            self.indexes_dir,
            self.experiments_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton"""
    return Settings()

"""Serializable configuration for repeatable retrieval experiments."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RetrievalExperimentConfig(BaseModel):
    """All retrieval choices that affect an experiment's result."""

    name: str
    embedding_model: str = "all-MiniLM-L6-v2"
    chunking_method: Literal["fixed", "sentence", "recursive"] = "recursive"
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=128, ge=0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    dense_top_k: int = Field(default=20, gt=0)
    bm25_top_k: int = Field(default=20, gt=0)
    fusion_strategy: Literal["weighted", "rrf"] = "weighted"
    hybrid_alpha: float = Field(default=0.5, ge=0, le=1)
    rrf_k: int = Field(default=60, gt=0)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = Field(default=5, gt=0)
    graph_expansion_enabled: bool = False
    graph_direction: Literal["upstream", "downstream", "both"] = "both"
    graph_depth: int = Field(default=1, ge=1)

    def save(self, path: Path) -> None:
        """Persist the exact configuration used for an experiment."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RetrievalExperimentConfig":
        """Load an experiment configuration without applying ambient defaults."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
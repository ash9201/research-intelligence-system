"""
Core data models for the Research Intelligence System
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an ingested document"""
    
    doc_id: str = Field(description="Unique document identifier")
    title: Optional[str] = Field(default=None, description="Document title")
    content: str = Field(description="Full document content")
    file_path: str = Field(description="Original file path")
    file_type: str = Field(description="File type (e.g., 'pdf', 'txt')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """Represents a chunk of a document"""
    
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk content")
    start_char: int = Field(description="Starting character position in document")
    end_char: int = Field(description="Ending character position in document")
    chunk_index: int = Field(description="Index of chunk in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class EmbeddingResult(BaseModel):
    """Result of embedding a text"""
    
    text: str = Field(description="Original text")
    embedding: List[float] = Field(description="Embedding vector")
    model: str = Field(description="Embedding model used")


class RetrievalResult(BaseModel):
    """Single result from retrieval"""
    
    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="Chunk content")
    score: float = Field(description="Retrieval score")
    retrieval_method: str = Field(description="Method used (bm25/dense/hybrid)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Complete retrieval response"""
    
    query: str = Field(description="Original query")
    results: List[RetrievalResult] = Field(description="Retrieved results")
    total_count: int = Field(description="Total number of results")
    retrieval_time_ms: float = Field(description="Retrieval time in milliseconds")


class RerankingResult(BaseModel):
    """Result of reranking"""
    
    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="Chunk content")
    relevance_score: float = Field(description="Reranked relevance score (0-1)")
    original_rank: int = Field(description="Original rank before reranking")
    new_rank: int = Field(description="New rank after reranking")


class Citation(BaseModel):
    """Represents a citation/grounding for a generated answer"""
    
    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="Cited content")
    relevance_score: float = Field(description="Relevance score")
    position_in_answer: List[tuple] = Field(
        default_factory=list,
        description="Positions in answer (start, end) where this citation is used"
    )


class EvidenceSource(BaseModel):
    """Normalized selected evidence passed to generation regardless of retrieval path."""

    source_index: int
    chunk_id: str
    doc_id: str
    content: str
    score: float
    title: Optional[str] = None
    page: Optional[int] = None
    pages: List[int] = Field(default_factory=list)
    section: Optional[str] = None


class GenerationStatus(BaseModel):
    """Safe provenance for an answer-generation attempt; no credentials are included."""

    generation_mode: str
    provider: str
    configured_model: str
    used_model: Optional[str] = None
    fallback_reason: Optional[str] = None
    provider_status: str = "not_attempted"
    grounding_status: str = "not_evaluated"


class AnswerReliability(BaseModel):
    """Uncalibrated evidence indicators, not answer-probability claims."""

    evidence_quality: float
    citation_coverage: float
    grounding_score: float
    reliability_indicator: Optional[float] = None
    score_type: str


class GeneratedAnswer(BaseModel):
    """LLM-generated answer with citations"""
    
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer text")
    citations: List[Citation] = Field(description="Supporting citations")
    model: str = Field(description="LLM model used")
    generation_time_ms: float = Field(description="Generation time in milliseconds")
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score (0-1)"
    )
    generation_status: Optional[GenerationStatus] = None
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)
    reliability: Optional[AnswerReliability] = None


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for retrieval or generation"""
    
    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

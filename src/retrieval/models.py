"""
Retrieval models and data structures
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RetrievalIndex(BaseModel):
    """Metadata about a retrieval index"""
    
    index_name: str = Field(description="Name of the index")
    index_type: str = Field(description="Type (bm25, dense, hybrid)")
    chunk_count: int = Field(description="Number of chunks indexed")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model information")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")

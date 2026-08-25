"""Models for query analysis and decomposition."""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Classification of a user's query for retrieval routing."""

    original_query: str = Field(description="Original user query")
    complexity: str = Field(
        description="Query complexity category, e.g. simple or multi_part"
    )
    requires_decomposition: bool = Field(
        description="Whether the query should be decomposed"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation for the classification decision",
    )


class SubQuery(BaseModel):
    """A single retrieval-oriented subquery derived from a parent query."""

    query_id: str = Field(description="Identifier within the parent query")
    text: str = Field(description="Text used for retrieval")
    parent_query: str = Field(description="Original user query")


class DecomposedQuery(BaseModel):
    """Result of decomposing a user query into retrieval subqueries."""

    original_query: str = Field(description="Original user query")
    subqueries: List[SubQuery] = Field(
        description="Retrieval-oriented subqueries"
    )
"""Lightweight query complexity analysis."""

import re

from src.query.models import QueryAnalysis


class QueryAnalyzer:
    """
    Detect whether a query is likely to contain multiple information needs.

    This is deliberately heuristic for the first experimental baseline.
    """

    _MULTI_PART_PATTERNS = (
        r"\bwhat\b.+\band\b.+\bhow\b",
        r"\bwhat\b.+,\s*.+,\s*and\b",
        r"\bhow\b.+\band why\b",
        r"\bwhy\b.+\band\b.+\bhow\b",
        r"\bwhat\b.+\band what\b",
        r"\bcompare\b.+\band\b",
        r"\bdifference between\b.+\band\b",
    )

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        """Analyze a query and decide whether decomposition is warranted."""
        normalized = " ".join(query.strip().split()).lower()

        if not normalized:
            return QueryAnalysis(
                original_query=query,
                complexity="invalid",
                requires_decomposition=False,
                reason="Query is empty or contains only whitespace",
            )

        for pattern in cls._MULTI_PART_PATTERNS:
            if re.search(pattern, normalized):
                return QueryAnalysis(
                    original_query=query,
                    complexity="multi_part",
                    requires_decomposition=True,
                    reason=f"Matched multi-part pattern: {pattern}",
                )

        return QueryAnalysis(
            original_query=query,
            complexity="simple",
            requires_decomposition=False,
            reason="No multi-part trigger detected",
        )
"""Deterministic baseline query decomposition."""

from __future__ import annotations

import re

from src.query.models import DecomposedQuery, SubQuery


class QueryDecomposer:
    """
    Decompose multi-part questions into retrieval-oriented subqueries.

    This is deliberately a deterministic baseline. It should be reasonably
    general without relying on an LLM or hard-coding individual benchmark
    questions.
    """

    @staticmethod
    def _clean_fragment(fragment: str) -> str:
        """Normalize punctuation and whitespace in a subquery fragment."""
        fragment = " ".join(fragment.strip().split())
        fragment = fragment.strip(" ,;")

        if fragment and not fragment.endswith("?"):
            fragment += "?"

        return fragment

    @classmethod
    def _make_subqueries(
        cls,
        query: str,
        fragments: list[str],
    ) -> list[SubQuery]:
        """Convert fragments into normalized SubQuery objects."""
        subqueries: list[SubQuery] = []

        for fragment in fragments:
            text = cls._clean_fragment(fragment)

            if not text or text == "?":
                continue

            subqueries.append(
                SubQuery(
                    query_id=f"q{len(subqueries) + 1}",
                    text=text,
                    parent_query=query,
                )
            )

        return subqueries

    @staticmethod
    def _split_question_clauses(query: str) -> list[str]:
        """
        Split independent interrogative clauses in a multi-part question.

        Examples:
            "What X, how Y, and why Z?"
            "How X, and what Y?"
            "What X, which Y, and why Z?"
        """
        # Protect the dedicated "How does X differ from Y, and why?" pattern.
        if re.match(
            r"(?i)^how does .+? differ from .+?,\s*and why\??$",
            query,
        ):
            return [query]

        # Normalize ", and <question word>" to ", <question word>".
        normalized = re.sub(
            r",?\s+and\s+(?=(?:how|why|what|which|where|when|whether)\b)",
            ", ",
            query,
            flags=re.IGNORECASE,
        )

        # Split before a new interrogative clause.
        parts = re.split(
            r",\s*(?=(?:how|why|what|which|where|when|whether)\b)",
            normalized,
            flags=re.IGNORECASE,
        )

        return [
            part.strip(" ,")
            for part in parts
            if part.strip(" ,")
        ]

    @staticmethod
    def _split_comma_list_question(query: str) -> list[str] | None:
        """
        Handle questions of the form:

        What X, Y, and Z ...?

        The items are interpreted as separate information needs while
        retaining the common question context.
        """
        match = re.match(
            r"""
            ^what\s+
            (.+?),\s*
            (.+?),\s*
            and\s+
            (.+?)
            (
                \s+were\s+used
                |
                \s+was\s+used
                |
                \s+is\s+used
                |
                \s+are\s+used
            )
            (?:\s+(.+?))?
            \??$
            """,
            query,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if not match:
            return None

        first, second, third, verb, continuation = match.groups()

        continuation = (
            f" {continuation.strip()}"
            if continuation
            else ""
        )

        # Preserve the original grammatical form as much as possible.
        # "What optimizer was used..." etc.
        return [
            f"What {first.strip()}{verb}{continuation}?",
            f"What {second.strip()}{verb}{continuation}?",
            f"What {third.strip()}{verb}{continuation}?",
        ]

    @classmethod
    def decompose(cls, query: str) -> DecomposedQuery:
        """
        Decompose a query using deterministic linguistic patterns.

        If no reliable decomposition is identified, return the original
        query as one subquery.
        """
        normalized = " ".join(query.strip().split())

        if not normalized:
            return DecomposedQuery(
                original_query=query,
                subqueries=[],
            )

        # 1. Explicit comma-list question:
        #    "What X, Y, and Z were used ...?"
        fragments = cls._split_comma_list_question(normalized)

        if fragments:
            return DecomposedQuery(
                original_query=normalized,
                subqueries=cls._make_subqueries(
                    normalized,
                    fragments,
                ),
            )

        # 2. Existing comparison structure:
        #    "How does X differ from Y, and why?"
        match = re.match(
            r"(?i)^how does (.+?) differ from (.+?),\s*and why\??$",
            normalized,
        )

        if match:
            first, second = match.groups()

            fragments = [
                f"How does {first} differ from {second}?",
                f"Why does {first} differ from {second}?",
            ]

            return DecomposedQuery(
                original_query=normalized,
                subqueries=cls._make_subqueries(
                    normalized,
                    fragments,
                ),
            )

        # 3. "What is X and what is Y?"
        match = re.match(
            r"(?i)^what is (.+?)\s+and\s+what is (.+?)\??$",
            normalized,
        )

        if match:
            first, second = match.groups()

            fragments = [
                f"What is {first}?",
                f"What is {second}?",
            ]

            return DecomposedQuery(
                original_query=normalized,
                subqueries=cls._make_subqueries(
                    normalized,
                    fragments,
                ),
            )

        # 4. Split independent interrogative clauses:
        #    "... how X ..., and why Y ...?"
        fragments = cls._split_question_clauses(normalized)

        if len(fragments) > 1:
            return DecomposedQuery(
                original_query=normalized,
                subqueries=cls._make_subqueries(
                    normalized,
                    fragments,
                ),
            )

        # 5. Safe fallback: retain the original query unchanged.
        return DecomposedQuery(
            original_query=normalized,
            subqueries=cls._make_subqueries(
                normalized,
                [normalized],
            ),
        )
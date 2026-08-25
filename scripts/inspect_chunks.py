"""Fast inspection/search utility for indexed document chunks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.retrieval import IndexManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
INDEX_NAME = "attention_is_all_you_need"


def normalize(text: str) -> str:
    """Normalize whitespace for matching/display."""
    return " ".join(text.split())


def find_matches(
    chunks: dict,
    phrase: str,
) -> list[tuple[str, object]]:
    """Find chunks whose text contains all words/phrase terms."""
    phrase_normalized = normalize(phrase).lower()

    # First try exact normalized phrase matching.
    exact_matches = []
    for chunk_id, chunk in chunks.items():
        content = normalize(chunk.content)
        if phrase_normalized in content.lower():
            exact_matches.append((chunk_id, chunk))

    if exact_matches:
        return exact_matches

    terms = [
        term
        for term in re.findall(r"[A-Za-z0-9_]+", phrase_normalized)
        if len(term) > 2
    ]   

    if not terms:
        return []

    matches = []

    for chunk_id, chunk in chunks.items():
        content = normalize(chunk.content).lower()

        if all(
            re.search(rf"\b{re.escape(term)}\b", content)
            for term in terms
        ):
            matches.append((chunk_id, chunk))

    return matches


def print_chunk(chunk_id: str, chunk: object) -> None:
    """Print chunk metadata and complete content."""
    metadata = getattr(chunk, "metadata", {}) or {}

    title = metadata.get("title")
    section = metadata.get("section")
    page = metadata.get("page")
    pages = metadata.get("pages")
    actual_chunk_id = getattr(chunk, "chunk_id", chunk_id)

    print("=" * 100)
    print(f"Chunk:   {actual_chunk_id}")
    print(f"Title:   {title}")
    print(f"Section: {section}")
    print(f"Page:    {page}")
    print(f"Pages:   {pages}")
    print("-" * 100)
    print(chunk.content)
    print()
    

def main() -> None:
    """Search the persisted index for one or more phrases."""
    phrases = sys.argv[1:]

    if not phrases:
        print(
            "Usage:\n"
            "  python -m scripts.inspect_chunks \"search phrase\"\n\n"
            "Examples:\n"
            "  python -m scripts.inspect_chunks \"label smoothing\"\n"
            "  python -m scripts.inspect_chunks "
            "\"beam search length penalty alpha 0.6\""
        )
        return

    index_manager = IndexManager(INDEX_DIR)
    retriever = index_manager.load_index(INDEX_NAME)

    # The BM25 chunk map gives us direct access to the actual indexed chunks.
    chunks = retriever.bm25.chunk_map

    print(f"Loaded {len(chunks)} indexed chunks.\n")

    for phrase in phrases:
        print("\n" + "#" * 100)
        print(f"SEARCH: {phrase}")
        print("#" * 100)

        matches = find_matches(chunks, phrase)

        if not matches:
            print("No direct/all-term matches found.")
            continue

        print(f"Found {len(matches)} matching chunk(s).\n")

        for chunk_id, chunk in matches:
            print_chunk(chunk_id, chunk)


if __name__ == "__main__":
    main()
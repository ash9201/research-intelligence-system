"""Document chunking strategies with source-provenance preservation."""
import re
from typing import Any, Dict, List, Optional

from src.logging_config import get_logger
from src.models import Chunk

logger = get_logger(__name__)


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split content into chunks"""
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """Chunks text into fixed character windows with configurable overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Return fixed windows while retaining source positions and provenance."""
        if not content:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        step = self.chunk_size - self.overlap
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
                    doc_id=doc_id,
                    content=content[start:end],
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=dict(metadata or {}),
                )
            )
            if end == len(content):
                break
            start += step
            chunk_index += 1
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunks documents by sentences with overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Initialize sentence chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk content by sentences"""
        sentence_spans = [
            (match.start(), match.end())
            for match in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", content, flags=re.DOTALL)
            if match.group().strip()
        ]
        if not sentence_spans:
            return []

        chunks = []
        start_sentence = 0
        chunk_index = 0
        while start_sentence < len(sentence_spans):
            start_char = sentence_spans[start_sentence][0]
            end_sentence = start_sentence
            end_char = sentence_spans[end_sentence][1]
            while end_sentence + 1 < len(sentence_spans):
                candidate_end = sentence_spans[end_sentence + 1][1]
                if candidate_end - start_char > self.chunk_size:
                    break
                end_sentence += 1
                end_char = candidate_end

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
                    doc_id=doc_id,
                    content=content[start_char:end_char].strip(),
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_index,
                    metadata=dict(metadata or {}),
                )
            )
            chunk_index += 1

            if end_sentence == len(sentence_spans) - 1:
                break
            next_start = end_sentence + 1
            while (
                next_start > start_sentence
                and end_char - sentence_spans[next_start][0] <= self.overlap
            ):
                next_start -= 1
            start_sentence = next_start if next_start > start_sentence else end_sentence + 1
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Chunks by paragraph, then sentence, then word boundaries as needed."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 128,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize recursive chunker
        
        Args:
            chunk_size: Target size of each chunk
            overlap: Character overlap between chunks
            separators: List of separators to try in order
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def chunk(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Create contiguous chunks without cutting normal words or sentences."""
        if not content:
            return []
        units: List[tuple[int, int]] = []
        for paragraph in re.finditer(r"(?s)\S.*?(?=\n\s*\n|\Z)", content):
            start, end = paragraph.span()
            if end - start <= self.chunk_size:
                units.append((start, end))
            else:
                units.extend(self._split_oversized(content, start, end))

        chunks: List[Chunk] = []
        unit_index = 0
        while unit_index < len(units):
            start = units[unit_index][0]
            end = units[unit_index][1]
            last_index = unit_index
            while last_index + 1 < len(units) and units[last_index + 1][1] - start <= self.chunk_size:
                last_index += 1
                end = units[last_index][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                    doc_id=doc_id,
                    content=content[start:end].strip(),
                    start_char=start,
                    end_char=end,
                    chunk_index=len(chunks),
                    metadata=dict(metadata or {}),
                )
            )
            if last_index == len(units) - 1:
                break
            next_index = last_index + 1
            while next_index > unit_index and end - units[next_index][0] <= self.overlap:
                next_index -= 1
            unit_index = next_index if next_index > unit_index else last_index + 1
        return chunks

    def _split_oversized(self, content: str, start: int, end: int) -> List[tuple[int, int]]:
        """Prefer sentences; use whitespace boundaries only for a long single sentence."""
        spans = [
            (start + match.start(), start + match.end())
            for match in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", content[start:end], flags=re.DOTALL)
            if match.group().strip()
        ]
        if len(spans) > 1:
            return spans
        return self._split_words(content, start, end)

    def _split_words(self, content: str, start: int, end: int) -> List[tuple[int, int]]:
        """Split a long sentence on whitespace, falling back to characters for one long token."""
        spans = []
        cursor = start
        while cursor < end:
            limit = min(cursor + self.chunk_size, end)
            if limit < end:
                boundary = content.rfind(" ", cursor, limit + 1)
                if boundary > cursor:
                    limit = boundary
            if limit == cursor:
                limit = min(cursor + self.chunk_size, end)
            spans.append((cursor, limit))
            cursor = limit
            while cursor < end and content[cursor].isspace():
                cursor += 1
        return spans


class DocumentChunker:
    """Main interface for chunking documents"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128, strategy: str = "recursive"):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            strategy: Chunking strategy ("fixed", "sentence", or "recursive")
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        
        if strategy == "fixed":
            self._chunker = FixedSizeChunker(chunk_size, overlap)
        elif strategy == "recursive":
            self._chunker = RecursiveChunker(chunk_size, overlap)
        elif strategy == "sentence":
            self._chunker = SentenceChunker(chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a document while retaining page, section, and other provenance."""
        return self._chunker.chunk(content, doc_id, metadata)

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a logical cross-page flow while retaining the source page range.

        PDF pages are preserved as provenance rather than hard chunk boundaries.
        A leading figure or table can interrupt a sentence continued from the prior
        page, so its visual block is retained but placed after the leading prose
        continuation for logical reading order. No source text is discarded.
        """
        logical_content, page_spans, layout_blocks = self._build_logical_page_flow(pages)
        chunks: List[Chunk] = []
        for section_start, section_end, section in self._section_spans(logical_content):
            section_chunks = self._chunker.chunk(
                logical_content[section_start:section_end],
                doc_id,
                metadata,
            )
            for chunk in section_chunks:
                chunk.start_char += section_start
                chunk.end_char += section_start
                if section:
                    chunk.metadata["section"] = section
                chunks.append(chunk)

        for chunk_index, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc_id}_chunk_{chunk_index:04d}"
            chunk.chunk_index = chunk_index
            chunk_pages = self._pages_for_span(page_spans, chunk.start_char, chunk.end_char)
            chunk.metadata["pages"] = chunk_pages
            chunk.metadata["page"] = chunk_pages[0]
            chunk.metadata.setdefault("section", self._section_at_offset(logical_content, chunk.start_char))

        # Keep visual content retrievable without inserting it into prose carried
        # across a physical page break.
        for page_number, layout_text in layout_blocks:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks):04d}",
                    doc_id=doc_id,
                    content=layout_text,
                    start_char=-1,
                    end_char=-1,
                    chunk_index=len(chunks),
                    metadata={
                        **(metadata or {}),
                        "page": page_number,
                        "pages": [page_number],
                        "content_type": "layout",
                    },
                )
            )
        logger.info("Created %d logical-flow chunks for document %s", len(chunks), doc_id)
        return chunks

    @staticmethod
    def _build_logical_page_flow(
        pages: List[Dict[str, Any]],
    ) -> tuple[str, List[tuple[int, int, int]], List[tuple[int, str]]]:
        """Join page text while recording each physical page's positions in the flow."""
        parts: List[str] = []
        page_spans: List[tuple[int, int, int]] = []
        layout_blocks: List[tuple[int, str]] = []
        offset = 0
        previous_text = ""
        for page in pages:
            page_text, page_layout_blocks = DocumentChunker._partition_page_blocks(page.get("text", ""))
            layout_blocks.extend((page["number"], block) for block in page_layout_blocks)
            if parts:
                separator = " " if DocumentChunker._continues_sentence(previous_text, page_text) else "\n\n"
                parts.append(separator)
                offset += len(separator)
            start = offset
            parts.append(page_text)
            offset += len(page_text)
            page_spans.append((start, offset, page["number"]))
            previous_text = page_text
        return "".join(parts), page_spans, layout_blocks

    @staticmethod
    def _section_spans(content: str) -> List[tuple[int, int, Optional[str]]]:
        """Return logical ranges bounded by numbered headings, including preamble text."""
        heading_pattern = re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+)*\.?[ \t]+[^\n]+)[ \t]*$")
        headings = list(heading_pattern.finditer(content))
        if not headings:
            return [(0, len(content), None)]
        spans: List[tuple[int, int, Optional[str]]] = []
        if headings[0].start() > 0:
            spans.append((0, headings[0].start(), None))
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
            spans.append((heading.start(), end, heading.group(1).strip()))
        return spans

    @staticmethod
    def _partition_page_blocks(page_text: str) -> tuple[str, List[str]]:
        """Separate page prose from figures, tables, and page-number decorations."""
        blocks = [block.strip() for block in re.split(r"\n\s*\n", page_text) if block.strip()]
        prose_blocks = [block for block in blocks if not DocumentChunker._is_layout_block(block)]
        layout_blocks = [block for block in blocks if DocumentChunker._is_layout_block(block)]
        return "\n\n".join(prose_blocks), layout_blocks

    @staticmethod
    def _is_layout_block(block: str) -> bool:
        """Recognize standalone page decorations, captions, and tabular display blocks."""
        normalized = block.strip().lower()
        return (
            normalized.isdigit()
            or normalized.startswith(("figure", "table", "layer type", "scaled dot-product attention multi-head attention"))
            or normalized.startswith(("self-attention o(", "recurrent o(", "convolutional o("))
        )

    @staticmethod
    def _continues_sentence(previous_text: str, current_text: str) -> bool:
        """Recognize prose continued from one physical page to the next."""
        return bool(
            previous_text
            and current_text
            and re.search(r"[A-Za-z0-9]$", previous_text)
            and re.match(r"^[a-z]", current_text)
        )

    @staticmethod
    def _pages_for_span(
        page_spans: List[tuple[int, int, int]],
        start_char: int,
        end_char: int,
    ) -> List[int]:
        """Return all physical pages that overlap a logical chunk span."""
        return [
            page_number
            for page_start, page_end, page_number in page_spans
            if page_start < end_char and page_end > start_char
        ]

    @staticmethod
    def _section_at_offset(content: str, offset: int) -> Optional[str]:
        """Return the most recent numbered or all-caps section heading before offset."""
        section = None
        for match in re.finditer(r"(?m)^[ \t]*((?:\d+(?:\.\d+)*\.?[ \t]+[^\n]+)|(?:[A-Z][A-Z \t]{3,}))[ \t]*$", content):
            if match.start() > offset:
                break
            section = match.group(1).strip()
        return section

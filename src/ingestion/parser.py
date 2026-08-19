"""Document parsing with page-aware, reading-order-oriented PDF extraction."""
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import List, Optional

from pypdf import PdfReader

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractedPage:
    """One extracted PDF page before document chunking."""

    number: int
    text: str


class DocumentParser:
    """Parses documents from various formats"""
    
    @staticmethod
    def parse_pdf(file_path: Path) -> str:
        """Parse a PDF to the backward-compatible page-marked text representation."""
        return "\n".join(
            f"--- Page {page.number} ---\n{page.text}"
            for page in DocumentParser.parse_pdf_pages(file_path)
        )

    @staticmethod
    def parse_pdf_pages(file_path: Path) -> List[ExtractedPage]:
        """Extract pages with PyMuPDF sorted reading order, falling back to pypdf."""
        file_path = Path(file_path)
        try:
            import pymupdf

            document = pymupdf.open(str(file_path))
            pages = [
                ExtractedPage(number=index + 1, text=DocumentParser.normalize_pdf_text(page.get_text("text", sort=True)))
                for index, page in enumerate(document)
            ]
            document.close()
            logger.info("Parsed PDF with PyMuPDF: %s (%d pages)", file_path, len(pages))
            return pages
        except ImportError:
            logger.warning("PyMuPDF is unavailable; falling back to pypdf for %s", file_path)
        except Exception as error:
            logger.warning("PyMuPDF extraction failed for %s: %s; falling back to pypdf", file_path, error)

        try:
            reader = PdfReader(str(file_path))
            pages = [
                ExtractedPage(number=index + 1, text=DocumentParser.normalize_pdf_text(page.extract_text() or ""))
                for index, page in enumerate(reader.pages)
            ]
            logger.info("Parsed PDF with pypdf fallback: %s (%d pages)", file_path, len(pages))
            return pages
        except Exception as error:
            logger.error(f"Error parsing PDF {file_path}: {error}")
            raise

    @staticmethod
    def normalize_pdf_text(text: str) -> str:
        """Conservatively repair layout whitespace without inventing mathematical notation.

        This intentionally does not translate formulas to LaTeX or replace extracted
        symbols. It only joins ordinary line-wrapped words and collapses layout-only
        whitespace while retaining paragraph boundaries.
        """
        normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
        # Preserve a source hyphen rather than guessing whether it was semantic.
        normalized = re.sub(r"([-‐])\s*\n\s*", r"\1", normalized)
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
    
    @staticmethod
    def parse_text(file_path: Path, encoding: str = "utf-8") -> str:
        """Parse text from a plain text file"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            logger.info(f"Successfully parsed text file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            raise
    
    @staticmethod
    def parse_markdown(file_path: Path, encoding: str = "utf-8") -> str:
        """Parse markdown file"""
        # Markdown is plain text, so same as text parser
        return DocumentParser.parse_text(file_path, encoding)
    
    @staticmethod
    def parse_document(file_path: Path) -> str:
        """Parse a document based on its file extension"""
        file_path = Path(file_path)
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return DocumentParser.parse_pdf(file_path)
        elif suffix in [".txt", ".md"]:
            return DocumentParser.parse_text(file_path)
        else:
            # Try as text by default
            logger.warning(f"Unknown file type {suffix}, attempting to parse as text")
            return DocumentParser.parse_text(file_path)

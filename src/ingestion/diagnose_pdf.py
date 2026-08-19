"""Write page-level PDF extraction output before metadata extraction or chunking."""
import argparse
from pathlib import Path

from pypdf import PdfReader

from src.ingestion.parser import DocumentParser


def legacy_pypdf_text(pdf_path: Path) -> str:
    """Return the previous pypdf-only extraction for a transparent comparison."""
    reader = PdfReader(str(pdf_path))
    return "\n".join(
        f"--- Page {index + 1} ---\n{page.extract_text() or ''}"
        for index, page in enumerate(reader.pages)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDF text before document chunking.")
    parser.add_argument("pdf", type=Path, help="PDF to inspect")
    parser.add_argument("--output", type=Path, help="Write new extraction to this text file")
    parser.add_argument("--compare-pypdf", action="store_true", help="Also write legacy pypdf output")
    args = parser.parse_args()

    pages = DocumentParser.parse_pdf_pages(args.pdf)
    extracted = "\n".join(f"--- Page {page.number} ---\n{page.text}" for page in pages)
    output = args.output or args.pdf.with_suffix(".extracted.txt")
    output.write_text(extracted, encoding="utf-8")
    print(f"extractor=PyMuPDF pages={len(pages)} characters={len(extracted)} output={output}")

    if args.compare_pypdf:
        legacy_output = output.with_name(f"{output.stem}.pypdf{output.suffix}")
        legacy_output.write_text(legacy_pypdf_text(args.pdf), encoding="utf-8")
        print(f"extractor=pypdf output={legacy_output}")


if __name__ == "__main__":
    main()
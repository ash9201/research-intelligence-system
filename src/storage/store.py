"""
Local persistence storage
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from src.logging_config import get_logger
from src.models import Document, Chunk

logger = get_logger(__name__)


class LocalStore:
    """Local file-based storage"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize local store
        
        Args:
            data_dir: Base directory for storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_document(self, document: Document) -> None:
        """Save document to disk"""
        doc_dir = self.data_dir / "documents" / document.doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(doc_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "file_path": document.file_path,
                    "file_type": document.file_type,
                    "metadata": document.metadata,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                },
                f,
                indent=2,
            )
        
        # Save content
        with open(doc_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(document.content)
        
        logger.info(f"Document saved: {document.doc_id}")
    
    def load_document(self, doc_id: str) -> Document:
        """Load document from disk"""
        doc_dir = self.data_dir / "documents" / doc_id
        
        if not doc_dir.exists():
            raise FileNotFoundError(f"Document not found: {doc_id}")
        
        # Load metadata
        with open(doc_dir / "metadata.json", "r") as f:
            meta = json.load(f)
        
        # Load content
        with open(doc_dir / "content.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        document = Document(
            doc_id=meta["doc_id"],
            title=meta["title"],
            content=content,
            file_path=meta["file_path"],
            file_type=meta["file_type"],
            metadata=meta["metadata"],
        )
        
        logger.info(f"Document loaded: {doc_id}")
        return document
    
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks to disk"""
        chunks_dir = self.data_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON lines
        with open(chunks_dir / "chunks.jsonl", "w") as f:
            for chunk in chunks:
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(chunk_data) + "\n")
        
        logger.info(f"Saved {len(chunks)} chunks")
    
    def load_chunks(self, doc_id: str = None) -> List[Chunk]:
        """Load chunks from disk"""
        chunks_file = self.data_dir / "chunks" / "chunks.jsonl"
        
        if not chunks_file.exists():
            return []
        
        chunks = []
        with open(chunks_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if doc_id and data["doc_id"] != doc_id:
                    continue
                chunk = Chunk(**data)
                chunks.append(chunk)
        
        return chunks
    
    def save_metadata(self, key: str, value: Any) -> None:
        """Save metadata"""
        meta_dir = self.data_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        with open(meta_dir / f"{key}.json", "w") as f:
            json.dump(value, f, indent=2, default=str)
    
    def load_metadata(self, key: str) -> Any:
        """Load metadata"""
        meta_file = self.data_dir / "metadata" / f"{key}.json"
        
        if not meta_file.exists():
            return None
        
        with open(meta_file, "r") as f:
            return json.load(f)

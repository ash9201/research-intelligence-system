"""
BM25 sparse retrieval
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from src.logging_config import get_logger
from src.models import Chunk, RetrievalResult

logger = get_logger(__name__)


class BM25Retriever:
    """BM25-based sparse retrieval"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: BM25 parameter (default 1.5)
            b: BM25 parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.chunk_map: Dict[int, Chunk] = {}
        self.corpus: List[List[str]] = []
    
    def build_index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from chunks"""
        logger.info(f"Building BM25 index from {len(chunks)} chunks")
        
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        self.corpus = [chunk.content.split() for chunk in chunks]
        
        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        
        logger.info("BM25 index built successfully")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k results for query"""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = query.split()
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunk_map[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=float(scores[idx]),
                    retrieval_method="bm25",
                    metadata={"rank": rank},
                )
            )
        
        logger.debug(f"BM25 retrieval: retrieved {len(results)} results for query")
        return results
    
    def save(self, path: Path) -> None:
        """Save BM25 index to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "bm25": self.bm25,
            "chunk_map": self.chunk_map,
            "corpus": self.corpus,
            "k1": self.k1,
            "b": self.b,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load BM25 index from disk"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.chunk_map = data["chunk_map"]
        self.corpus = data["corpus"]
        self.k1 = data["k1"]
        self.b = data["b"]
        
        logger.info(f"BM25 index loaded from {path}")

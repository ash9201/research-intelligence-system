"""
Dense vector retrieval with FAISS
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.logging_config import get_logger
from src.models import Chunk, RetrievalResult

logger = get_logger(__name__)


class DenseRetriever:
    """Dense vector retrieval using FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize dense retriever
        
        Args:
            model_name: Sentence transformer model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.chunk_map: Dict[int, Chunk] = {}
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def build_index(self, chunks: List[Chunk], batch_size: int = 32) -> None:
        """Build FAISS index from chunks"""
        logger.info(f"Building dense index from {len(chunks)} chunks")
        
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        
        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            contents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        self.embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(self.embeddings)
        
        logger.info(f"Dense index built with {len(chunks)} embeddings")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k results for query"""
        if self.faiss_index is None:
            raise RuntimeError("Dense index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Build results (convert L2 distance to similarity)
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:  # Invalid index
                continue
            
            chunk = self.chunk_map[idx]
            # Convert L2 distance to similarity (higher is better)
            similarity = 1 / (1 + float(distance))
            
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=similarity,
                    retrieval_method="dense",
                    metadata={"rank": rank, "distance": float(distance)},
                )
            )
        
        logger.debug(f"Dense retrieval: retrieved {len(results)} results for query")
        return results
    
    def save(self, path: Path) -> None:
        """Save dense index to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "faiss_index": self.faiss_index,
            "embeddings": self.embeddings,
            "chunk_map": self.chunk_map,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Dense index saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load dense index from disk"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dense index not found: {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.faiss_index = data["faiss_index"]
        self.embeddings = data["embeddings"]
        self.chunk_map = data["chunk_map"]
        self.model_name = data["model_name"]
        self.embedding_dim = data["embedding_dim"]
        
        logger.info(f"Dense index loaded from {path}")

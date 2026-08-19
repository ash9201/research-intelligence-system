"""
Retrieval index management
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.logging_config import get_logger
from src.models import Chunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


class IndexManager:
    """Manages creation and loading of retrieval indexes"""
    
    def __init__(self, index_dir: Path):
        """
        Initialize index manager
        
        Args:
            index_dir: Directory to store indexes
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def create_index(
        self,
        index_name: str,
        chunks: List[Chunk],
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        fusion_strategy: str = "weighted",
        hybrid_alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> HybridRetriever:
        """
        Create a new hybrid retrieval index
        
        Args:
            index_name: Name for the index
            chunks: Chunks to index
            embedding_model: Sentence transformer model
        """
        logger.info(f"Creating index: {index_name}")
        
        # Create BM25 retriever
        bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        bm25.build_index(chunks)
        
        # Create dense retriever
        dense = DenseRetriever(model_name=embedding_model)
        dense.build_index(chunks)
        
        # Create hybrid retriever
        hybrid = HybridRetriever(
            bm25,
            dense,
            alpha=hybrid_alpha,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
        )
        
        # Save indexes
        index_path = self.index_dir / index_name
        index_path.mkdir(parents=True, exist_ok=True)
        
        bm25.save(index_path / "bm25.pkl")
        dense.save(index_path / "dense.pkl")
        
        # Save metadata
        metadata = {
            "index_name": index_name,
            "chunk_count": len(chunks),
            "embedding_model": embedding_model,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "fusion_strategy": fusion_strategy,
            "hybrid_alpha": hybrid_alpha,
            "rrf_k": rrf_k,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        with open(index_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Index created and saved: {index_name}")
        return hybrid
    
    def load_index(self, index_name: str) -> HybridRetriever:
        """
        Load a retrieval index
        
        Args:
            index_name: Name of the index to load
        """
        logger.info(f"Loading index: {index_name}")
        
        index_path = self.index_dir / index_name
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_name}")
        
        # Load BM25
        bm25 = BM25Retriever()
        bm25.load(index_path / "bm25.pkl")
        
        # Load dense
        dense = DenseRetriever()
        dense.load(index_path / "dense.pkl")
        
        # Create hybrid retriever
        metadata_path = index_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        hybrid = HybridRetriever(
            bm25,
            dense,
            alpha=metadata.get("hybrid_alpha", 0.5),
            fusion_strategy=metadata.get("fusion_strategy", "weighted"),
            rrf_k=metadata.get("rrf_k", 60),
        )
        
        logger.info(f"Index loaded: {index_name}")
        return hybrid
    
    def list_indexes(self) -> List[str]:
        """List all available indexes"""
        indexes = []
        for path in self.index_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                indexes.append(path.name)
        return sorted(indexes)

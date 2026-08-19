#!/usr/bin/env python3
"""
Debug reranker scores
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
import tempfile

# Create sample document
sample_text = """
Machine learning is a subset of artificial intelligence (AI) that gives systems the ability to automatically 
learn and improve from experience without being explicitly programmed. It focuses on developing algorithms 
and statistical models that computers can use to accomplish specific tasks.

Key concepts in machine learning include supervised learning, unsupervised learning, and reinforcement learning. 
In supervised learning, models are trained with labeled data. Unsupervised learning deals with unlabeled data, 
while reinforcement learning involves training agents through rewards and penalties.

Deep learning is a subset of machine learning based on artificial neural networks. Neural networks consist of 
interconnected layers of nodes (neurons) that process information similarly to the human brain.

Applications of machine learning are vast and include computer vision, natural language processing, and 
recommendation systems. Companies use machine learning for predictive analytics, fraud detection, and 
personalized user experiences.
"""

# Write sample to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_text)
    temp_file = f.name

try:
    # Load and chunk
    doc = DocumentLoader.load_document(temp_file)
    print(f"Loaded document: {doc.title}")
    
    chunker = DocumentChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk_document(doc.content, doc.doc_id)
    print(f"Created {len(chunks)} chunks")
    
    # Build index and retrieve
    with tempfile.TemporaryDirectory() as tmpdir:
        index_manager = IndexManager(Path(tmpdir))
        retriever = index_manager.create_index("test_index", chunks)
        
        # Retrieve
        results = retriever.retrieve("machine learning", top_k=5)
        print(f"\nRetrieved {len(results)} results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. score={r.score:.4f}, retrieval_method={r.retrieval_method}")
            print(f"     content: {r.content[:80]}...")
        
        # Rerank
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("machine learning", results, top_k=3)
        
        print(f"\nReranked {len(reranked)} results:")
        for i, r in enumerate(reranked):
            print(f"  {i+1}. relevance_score={r.relevance_score}, type={type(r.relevance_score)}")
            print(f"     original_rank={r.original_rank}, new_rank={r.new_rank}")
            print(f"     Check: 0 <= {r.relevance_score} <= 1? {0 <= r.relevance_score <= 1}")
            
finally:
    import os
    os.unlink(temp_file)

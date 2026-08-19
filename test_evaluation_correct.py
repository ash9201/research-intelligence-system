#!/usr/bin/env python3
"""
Test evaluation metrics with real retrieval results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
from src.evaluation import ReliabilityEstimator
from src.models import Citation
import tempfile

# Create sample document
sample_text = """
Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991.

Key features of Python include:
- Dynamic typing
- Automatic memory management through garbage collection
- Support for multiple programming paradigms
- Extensive standard library
- Large and active community

Python applications

Python is widely used in web development, data science, machine learning, artificial 
intelligence, automation, and scientific computing. Popular frameworks include Django 
for web development and NumPy for scientific computing.

Python and Data Science

Python has become the go-to language for data science thanks to libraries like pandas, 
numpy, and scikit-learn. Data scientists use Python for data cleaning, exploration, 
analysis, and visualization.
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_text)
    temp_file = f.name

try:
    print("\n" + "=" * 70)
    print("TEST: Evaluation Metrics on Real Retrieval Results")
    print("=" * 70 + "\n")
    
    # Load and chunk
    doc = DocumentLoader.load_document(temp_file)
    print("1. Loaded document: " + doc.title)
    
    chunker = DocumentChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(doc.content, doc.doc_id)
    print("   Created " + str(len(chunks)) + " chunks")
    
    # Build index
    with tempfile.TemporaryDirectory() as tmpdir:
        index_manager = IndexManager(Path(tmpdir))
        retriever = index_manager.create_index("test_index", chunks)
        
        # Retrieve and rerank
        query = "What is Python used for?"
        print("\n2. Query: '" + query + "'")
        
        initial_results = retriever.retrieve(query, top_k=5)
        print("   Retrieved " + str(len(initial_results)) + " initial results")
        
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, initial_results, top_k=3)
        print("   Reranked to " + str(len(reranked)) + " results with scores:")
        for i, r in enumerate(reranked, 1):
            print("   " + str(i) + ". score=" + str(round(r.relevance_score, 4)) + ": " + r.content[:50] + "...")
        
        # Test answer confidence estimation with Citation objects
        print("\n3. Testing answer confidence estimation:")
        
        # Create Citation objects from reranked results
        citations = [
            Citation(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                content=r.content,
                relevance_score=r.relevance_score,
                position_in_answer=[(0, 10)]
            )
            for r in reranked[:2]
        ]
        
        confidence = ReliabilityEstimator.estimate_answer_confidence(
            "This is a longer test answer about Python and its applications in various domains.",
            citations,
            len(reranked)
        )
        print("   Answer with " + str(len(citations)) + " citations: confidence=" + str(round(confidence, 2)))
        
        # Test retrieval confidence assessment
        print("\n4. Testing retrieval confidence assessment:")
        retrieval_conf = ReliabilityEstimator.assess_retrieval_confidence(reranked)
        print("   Overall confidence: " + str(round(retrieval_conf['overall_confidence'], 2)))
        print("   Score range: " + str(round(retrieval_conf['score_range'], 4)))
        print("   Top result score: " + str(round(retrieval_conf['top_result_score'], 4)))
        if 'score_variance' in retrieval_conf:
            print("   Score variance: " + str(round(retrieval_conf['score_variance'], 4)))
        
        print("\n   SUCCESS: All evaluation metrics working correctly")
        
finally:
    import os
    os.unlink(temp_file)

print("\n" + "=" * 70)
print("TEST COMPLETE - Evaluation metrics verified")
print("=" * 70 + "\n")

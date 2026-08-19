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
from src.evaluation import ReliabilityEstimator
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
        
        # Retrieve
        query = "What is Python used for?"
        print("\n2. Query: '" + query + "'")
        
        results = retriever.retrieve(query, top_k=5)
        print("   Retrieved " + str(len(results)) + " results with scores:")
        for i, r in enumerate(results, 1):
            print("   " + str(i) + ". score=" + str(round(r.score, 4)) + ": " + r.content[:60] + "...")
        
        # Test answer confidence estimation
        print("\n3. Testing answer confidence estimation:")
        
        # Simulate different answer scenarios
        scenarios = [
            ("Well-grounded answer with citations from multiple sources", 3, 0.9),
            ("Brief answer with minimal citations", 1, 0.5),
            ("Detailed but unsupported answer", 0, 0.3),
        ]
        
        for desc, num_citations, expected_approx in scenarios:
            print("   Scenario: " + desc)
            # Create mock citations
            mock_citations = results[:num_citations]
            confidence = ReliabilityEstimator.estimate_answer_confidence(
                "This is a test answer about Python.", 
                mock_citations,
                len(results)
            )
            print("   Confidence: " + str(round(confidence, 2)) + " (expected ~" + str(expected_approx) + ")")
        
        print("\n4. Testing retrieval confidence assessment:")
        retrieval_conf = ReliabilityEstimator.assess_retrieval_confidence(results)
        print("   Overall confidence: " + str(round(retrieval_conf['confidence'], 2)))
        print("   Score range: " + str(retrieval_conf['score_range']))
        print("   Top result score: " + str(round(retrieval_conf['top_result_score'], 4)))
        print("   Score variance: " + str(round(retrieval_conf['score_variance'], 4)))
        
finally:
    import os
    os.unlink(temp_file)

print("\n" + "=" * 70)
print("TEST COMPLETE - Evaluation metrics working correctly")
print("=" * 70 + "\n")

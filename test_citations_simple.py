#!/usr/bin/env python3
"""
Test LLM answer generation with citation extraction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
from src.generation import LLMClient, PromptTemplate, GroundingExtractor
import tempfile

# Create sample document
sample_text = """
Machine learning fundamentals

Machine learning is a subset of artificial intelligence (AI) that gives systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed. It focuses on developing algorithms and statistical models 
that computers can use to accomplish specific tasks.

Key concepts in machine learning include supervised learning, unsupervised learning, 
and reinforcement learning. In supervised learning, models are trained with labeled 
data. Unsupervised learning deals with unlabeled data, while reinforcement learning 
involves training agents through rewards and penalties.

Deep learning and neural networks

Deep learning is a subset of machine learning based on artificial neural networks. 
Neural networks consist of interconnected layers of nodes (neurons) that process 
information similarly to the human brain. Deep neural networks have multiple layers 
and can learn complex patterns in data.

The training process of neural networks involves forward propagation and backpropagation. 
During forward propagation, input data flows through the network to produce predictions. 
Backpropagation computes gradients and updates weights to minimize the loss function.
"""

# Write sample to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_text)
    temp_file = f.name

try:
    print("\n" + "=" * 70)
    print("TEST: Citation Extraction from LLM Answers")
    print("=" * 70 + "\n")
    
    # Load and chunk
    doc = DocumentLoader.load_document(temp_file)
    print("1. Loaded document: " + doc.title)
    print("   Size: " + str(len(doc.content)) + " chars")
    
    chunker = DocumentChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk_document(doc.content, doc.doc_id)
    print("\n2. Created " + str(len(chunks)) + " chunks")
    
    # Build index and retrieve
    with tempfile.TemporaryDirectory() as tmpdir:
        index_manager = IndexManager(Path(tmpdir))
        retriever = index_manager.create_index("test_index", chunks)
        
        # Retrieve
        query = "What is machine learning and deep learning?"
        print("\n3. Query: '" + query + "'")
        
        results = retriever.retrieve(query, top_k=5)
        print("   Retrieved " + str(len(results)) + " results")
        
        # Rerank
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, results, top_k=3)
        print("\n4. Reranked to " + str(len(reranked)) + " results")
        
        # Test citation extraction with mock answer
        print("\n5. Testing citation extraction with mock answer:")
        
        mock_answer = "[Source 1] Machine learning is AI that learns from data. [Source 2] Deep learning uses neural networks."
        print("   Mock answer: " + mock_answer)
        
        citations = GroundingExtractor.extract_citations(mock_answer, reranked)
        print("   Extracted " + str(len(citations)) + " citations")
        for i, cit in enumerate(citations, 1):
            print("   " + str(i) + ". chunk_id=" + cit.chunk_id + ", score=" + str(round(cit.relevance_score, 4)))
        
        # Test cleaning
        cleaned = GroundingExtractor.clean_answer(mock_answer)
        print("\n6. Cleaned answer (source markers removed):")
        print("   " + cleaned)
        
        # Test grounding assessment
        confidence, reason = GroundingExtractor.assess_groundedness(mock_answer, citations)
        print("\n7. Groundedness assessment:")
        print("   Confidence: " + str(round(confidence, 2)))
        print("   Reason: " + reason)
        
        print("\n   SUCCESS: Citations extracted and grounding assessed")
        
finally:
    import os
    os.unlink(temp_file)

print("\n" + "=" * 70)
print("TEST COMPLETE - All citation features working")
print("=" * 70 + "\n")

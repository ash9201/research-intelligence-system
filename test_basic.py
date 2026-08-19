#!/usr/bin/env python3
"""
Minimal verification script - tests basic functionality
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pydantic_models():
    """Test that Pydantic models work"""
    print("Testing Pydantic models...")
    from src.models import Document, Chunk, RetrievalResult
    from datetime import datetime
    
    # Create a document
    doc = Document(
        doc_id="test_doc_1",
        title="Test Document",
        content="This is a test document for verification.",
        file_path="/tmp/test.txt",
        file_type="txt"
    )
    
    # Create a chunk
    chunk = Chunk(
        chunk_id="chunk_1",
        doc_id="test_doc_1",
        content="This is a test",
        start_char=0,
        end_char=14,
        chunk_index=0
    )
    
    # Create retrieval result
    result = RetrievalResult(
        chunk_id="chunk_1",
        doc_id="test_doc_1",
        content="This is a test",
        score=0.95,
        retrieval_method="bm25"
    )
    
    print(f"  ✅ Document model: {doc.doc_id}")
    print(f"  ✅ Chunk model: {chunk.chunk_id}")
    print(f"  ✅ RetrievalResult model: {result.chunk_id}")
    return True

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    from src.config import get_settings
    
    settings = get_settings()
    print(f"  ✅ LLM Provider: {settings.llm_provider}")
    print(f"  ✅ Embedding Model: {settings.embedding_model}")
    print(f"  ✅ Data Dir: {settings.data_dir}")
    return True

def test_logging():
    """Test logging configuration"""
    print("Testing logging...")
    from src.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Test log message")
    print("  ✅ Logging configured")
    return True

def test_parser():
    """Test document parser"""
    print("Testing document parser...")
    from src.ingestion.parser import DocumentParser
    import tempfile
    
    # Create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.\nFor testing parsing.")
        temp_path = Path(f.name)
    
    try:
        content = DocumentParser.parse_text(temp_path)
        assert len(content) > 0
        print(f"  ✅ Parser: read {len(content)} characters")
        return True
    finally:
        temp_path.unlink()

def test_chunker():
    """Test document chunker"""
    print("Testing document chunker...")
    from src.chunking.chunker import DocumentChunker
    
    content = "First sentence. Second sentence. Third sentence. " * 5
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(content, "test_doc")
    
    assert len(chunks) > 0
    print(f"  ✅ Chunker: created {len(chunks)} chunks")
    return True

def test_bm25():
    """Test BM25 retriever"""
    print("Testing BM25 retriever...")
    from src.retrieval.bm25 import BM25Retriever
    from src.models import Chunk
    
    # Create test chunks
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="machine learning algorithms", 
              start_char=0, end_char=27, chunk_index=0),
        Chunk(chunk_id="c2", doc_id="d2", content="neural networks deep learning", 
              start_char=0, end_char=29, chunk_index=0),
        Chunk(chunk_id="c3", doc_id="d3", content="natural language processing", 
              start_char=0, end_char=27, chunk_index=0),
    ]
    
    # Build index
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    
    # Test retrieval
    results = retriever.retrieve("machine learning", top_k=2)
    assert len(results) > 0
    print(f"  ✅ BM25: retrieved {len(results)} results")
    return True

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("Testing evaluation metrics...")
    from src.evaluation.metrics import RetrievalMetrics
    
    relevant = ["doc1", "doc2", "doc3"]
    retrieved = ["doc1", "doc4", "doc2"]
    
    precision = RetrievalMetrics.precision_at_k(relevant, retrieved, 3)
    recall = RetrievalMetrics.recall_at_k(relevant, retrieved, 3)
    mrr = RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= mrr <= 1
    
    print(f"  ✅ Metrics: P@3={precision:.2f}, R@3={recall:.2f}, MRR={mrr:.2f}")
    return True

def test_knowledge_graph():
    """Test knowledge graph"""
    print("Testing knowledge graph...")
    from src.graph.knowledge_graph import KnowledgeGraph
    
    graph = KnowledgeGraph()
    
    # Add some relations
    graph.add_citation_relation("doc1", "doc2", "cites findings from")
    graph.add_citation_relation("doc2", "doc3", "builds upon")
    
    # Get cited documents
    cited = graph.get_cited_documents("doc1")
    assert "doc2" in cited
    
    print(f"  ✅ Knowledge Graph: added relations, retrieved correctly")
    return True

def main():
    print("\n" + "="*60)
    print("🧪 Research Intelligence System - Basic Verification")
    print("="*60 + "\n")
    
    tests = [
        test_pydantic_models,
        test_config,
        test_logging,
        test_parser,
        test_chunker,
        test_bm25,
        test_evaluation_metrics,
        test_knowledge_graph,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {test_func.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("✅ All basic verification tests passed!\n")
        print("Next steps:")
        print("  1. Start backend: python -m src.backend.app")
        print("  2. Start frontend: streamlit run src/frontend/main.py")
        print("  3. Open browser: http://localhost:8501\n")
        return 0
    else:
        print(f"❌ {failed} test(s) failed\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

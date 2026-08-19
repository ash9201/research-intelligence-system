"""
Quick verification script for Research Intelligence System
"""
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    try:
        from src.config import get_settings
        from src.ingestion import DocumentLoader
        from src.chunking import DocumentChunker
        from src.metadata import MetadataExtractor
        from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
        from src.reranking import CrossEncoderReranker
        from src.generation import LLMClient, PromptTemplate
        from src.evaluation import RetrievalMetrics
        from src.graph import KnowledgeGraph
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    try:
        from src.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded:")
        print(f"   - LLM Provider: {settings.llm_provider}")
        print(f"   - Embedding Model: {settings.embedding_model}")
        print(f"   - Reranker Model: {settings.reranker_model}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_document_parsing():
    """Test document parsing"""
    print("\n🔍 Testing document parsing...")
    try:
        from src.ingestion import DocumentParser
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. It contains sample text.")
            temp_file = Path(f.name)
        
        try:
            content = DocumentParser.parse_text(temp_file)
            assert len(content) > 0
            print(f"✅ Document parsing works! (parsed {len(content)} chars)")
            return True
        finally:
            temp_file.unlink()
    except Exception as e:
        print(f"❌ Document parsing failed: {e}")
        return False

def test_chunking():
    """Test document chunking"""
    print("\n🔍 Testing chunking...")
    try:
        from src.chunking import DocumentChunker
        
        content = "First sentence. Second sentence. Third sentence. " * 10
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(content, "test_doc")
        
        assert len(chunks) > 0
        assert all(chunk.doc_id == "test_doc" for chunk in chunks)
        print(f"✅ Chunking works! (created {len(chunks)} chunks)")
        return True
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return False

def test_bm25_retrieval():
    """Test BM25 retrieval"""
    print("\n🔍 Testing BM25 retrieval...")
    try:
        from src.retrieval import BM25Retriever
        from src.models import Chunk
        
        # Create test chunks
        chunks = [
            Chunk(
                chunk_id="c1", doc_id="d1", content="Machine learning is about AI",
                start_char=0, end_char=30, chunk_index=0
            ),
            Chunk(
                chunk_id="c2", doc_id="d2", content="Neural networks process data",
                start_char=0, end_char=28, chunk_index=0
            ),
        ]
        
        bm25 = BM25Retriever()
        bm25.build_index(chunks)
        results = bm25.retrieve("machine learning", top_k=2)
        
        assert len(results) > 0
        print(f"✅ BM25 retrieval works! (retrieved {len(results)} results)")
        return True
    except Exception as e:
        print(f"❌ BM25 retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\n🔍 Testing evaluation metrics...")
    try:
        from src.evaluation import RetrievalMetrics
        
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2"]
        
        p_at_3 = RetrievalMetrics.precision_at_k(relevant, retrieved, 3)
        r_at_3 = RetrievalMetrics.recall_at_k(relevant, retrieved, 3)
        mrr = RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved)
        
        assert 0 <= p_at_3 <= 1
        assert 0 <= r_at_3 <= 1
        assert 0 <= mrr <= 1
        
        print(f"✅ Evaluation metrics work!")
        print(f"   - Precision@3: {p_at_3:.3f}")
        print(f"   - Recall@3: {r_at_3:.3f}")
        print(f"   - MRR: {mrr:.3f}")
        return True
    except Exception as e:
        print(f"❌ Evaluation metrics failed: {e}")
        return False

def test_models():
    """Test Pydantic models"""
    print("\n🔍 Testing data models...")
    try:
        from src.models import Document, Chunk, RetrievalResult
        
        doc = Document(
            doc_id="test",
            title="Test",
            content="Test content",
            file_path="/tmp/test.txt",
            file_type="txt"
        )
        
        chunk = Chunk(
            chunk_id="c1", doc_id="test", content="Test",
            start_char=0, end_char=4, chunk_index=0
        )
        
        result = RetrievalResult(
            chunk_id="c1", doc_id="test", content="Test",
            score=0.9, retrieval_method="bm25"
        )
        
        print(f"✅ Data models work!")
        print(f"   - Document: {doc.doc_id}")
        print(f"   - Chunk: {chunk.chunk_id}")
        print(f"   - RetrievalResult: {result.chunk_id}")
        return True
    except Exception as e:
        print(f"❌ Data models failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("🧪 Research Intelligence System - Verification Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_configuration,
        test_models,
        test_document_parsing,
        test_chunking,
        test_bm25_retrieval,
        test_evaluation_metrics,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print(f"📊 Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("✅ All verification tests passed!")
        print("\n🚀 Next steps:")
        print("   1. Run: python -m src.backend.app")
        print("   2. In another terminal: streamlit run src/frontend/main.py")
        print("   3. Open: http://localhost:8501")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

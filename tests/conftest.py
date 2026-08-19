"""
Pytest configuration and fixtures
"""
import pytest
from pathlib import Path
from src.config import Settings
from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.models import Chunk, Document


@pytest.fixture
def test_data_dir():
    """Create test data directory"""
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def sample_text_document(test_data_dir):
    """Create a sample text document"""
    text_content = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed.
    
    Key Concepts
    
    Supervised learning involves training a model with labeled data. 
    The model learns to map inputs to outputs based on examples.
    
    Unsupervised learning works with unlabeled data, finding patterns and structures.
    
    Applications
    
    Machine learning has applications in computer vision, natural language processing,
    recommendation systems, and autonomous vehicles.
    """
    
    file_path = test_data_dir / "sample.txt"
    with open(file_path, "w") as f:
        f.write(text_content)
    
    return file_path


@pytest.fixture
def sample_chunks():
    """Create sample chunks"""
    return [
        Chunk(
            chunk_id="doc1_chunk_0001",
            doc_id="doc1",
            content="This is the first chunk about machine learning.",
            start_char=0,
            end_char=48,
            chunk_index=0,
        ),
        Chunk(
            chunk_id="doc1_chunk_0002",
            doc_id="doc1",
            content="This is the second chunk about deep learning.",
            start_char=48,
            end_char=93,
            chunk_index=1,
        ),
        Chunk(
            chunk_id="doc2_chunk_0001",
            doc_id="doc2",
            content="Neural networks are inspired by biological neurons.",
            start_char=0,
            end_char=51,
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_document():
    """Create a sample document"""
    return Document(
        doc_id="test_doc",
        title="Test Document",
        content="This is a test document with some content.",
        file_path="/tmp/test.txt",
        file_type="txt",
        metadata={"source": "test"},
    )

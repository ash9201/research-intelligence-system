# Research Intelligence System

A production-oriented research intelligence and document reasoning platform built with Python 3.12.

## 🎯 Features

### Core Capabilities
- **📄 Document Ingestion**: Support for PDF and text files with automatic parsing
- **✂️ Selectable Chunking**: Fixed-size, sentence-aware, and recursive structure-aware chunking with overlap and source metadata
- **🏷️ Metadata Extraction**: Automatic extraction of keywords, entities, and document properties
- **🔍 Hybrid Retrieval**: BM25 plus dense retrieval with score-normalized weighted fusion or Reciprocal Rank Fusion (RRF)
- **📊 Vector Search**: FAISS-powered approximate nearest neighbor search
- **🔗 Cross-Encoder Reranking**: Intelligent result reranking using transformer models
- **💬 Grounded Generation**: LLM-based answer generation with automatic citations
- **⭐ Evidence Evaluation**: Separate retrieval-score description, citation coverage, claim support, source agreement, and answer-confidence signals
- **📈 Evaluation Framework**: Retrieval metrics (Precision@K, Recall@K, NDCG, MRR)
- **🔗 Citation Graph**: Knowledge graph with configurable upstream/downstream candidate expansion
- **🎨 Web Interface**: Streamlit-based UI for interactive querying
- **⚙️ REST API**: FastAPI backend for programmatic access

## 🏗️ Architecture

### System Components

```
Document Ingestion
↓
Structure-Aware Chunking
↓
Metadata Extraction
↓
Index Building
├─ BM25 Sparse Index
├─ Dense Vector Index (FAISS)
└─ Hybrid Fusion (weighted or RRF)
↓
Query Processing
├─ BM25 Retrieval
├─ Dense Retrieval
├─ Hybrid Fusion
├─ Optional Citation-Graph Expansion
└─ Cross-Encoder Reranking
↓
Grounded Generation
├─ Prompt Construction
├─ LLM Generation
├─ Citation Extraction
└─ Confidence Estimation
↓
Output (Answer + Citations)
```

### Module Structure

```
src/
├── ingestion/          # Document loading and parsing
├── chunking/           # Document segmentation
├── metadata/           # Metadata extraction
├── retrieval/          # BM25, Dense, Hybrid retrieval
├── reranking/          # Cross-encoder reranking
├── generation/         # LLM integration and prompting
├── evaluation/         # Metrics and reliability assessment
├── graph/              # Knowledge graph structures
├── storage/            # Local persistence
├── backend/            # FastAPI application
├── frontend/           # Streamlit UI
├── config.py           # Configuration management
├── logging_config.py   # Logging setup
└── models.py           # Core data models
```

## 📋 Technology Stack

### Core Libraries
- **Python 3.12** - Programming language
- **Pydantic** - Data validation and serialization
- **rank-bm25** - BM25 sparse retrieval
- **sentence-transformers** - Embeddings and reranking
- **FAISS** - Vector indexing and search
- **pypdf** - PDF parsing

### Backend & Frontend
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Streamlit** - Web UI framework

### Testing
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting

### Optional
- **openai** - OpenAI API support
- **anthropic** - Anthropic API support
- **networkx** - Graph algorithms

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**
   ```bash
   cd research-intelligence-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the System

#### Option 1: Using Streamlit UI (Recommended for Beginners)

**Terminal 1 - Start API Server:**
```bash
python -m src.backend.app
```

**Terminal 2 - Start UI:**
```bash
streamlit run src/frontend/main.py
```

Access the UI at: http://localhost:8501

#### Option 2: Using REST API Directly

**Start the API server:**
```bash
python -m src.backend.app
```

API will be available at: http://localhost:8000

**Example API calls:**
```bash
# Check health
curl http://localhost:8000/health

# Get system info
curl http://localhost:8000/info

# Ingest document (assuming file.pdf exists)
curl -F "file=@file.pdf" http://localhost:8000/ingest

# Create index
curl -X POST "http://localhost:8000/index?index_name=my_index"

# Retrieve documents
curl -X POST "http://localhost:8000/retrieve?query=machine+learning&top_k=5"

# Generate answer
curl -X POST "http://localhost:8000/answer?query=What+is+machine+learning?"
```

### Usage Workflow

1. **Place Documents**: Put PDF or text files in `data/documents/` or use the UI to upload

2. **Create Index**: Index creates hybrid retrieval indexes from documents
   - Chunks documents with a configurable fixed, sentence-aware, or recursive strategy
   - Preserves supplied document provenance, including title, section, page, and chunk index metadata
   - Extracts metadata
   - Builds BM25 index
   - Generates embeddings with sentence-transformers
   - Creates FAISS vector index

3. **Retrieve**: Query the system
   - BM25 retrieves relevant chunks
   - Dense retrieval finds similar embeddings
   - Uses explicit max-normalized weighted fusion (default 50/50) or rank-based RRF
   - Can expand candidates through upstream/downstream citation links before reranking
   - Cross-encoder reranks top results

4. **Generate**: Get grounded answers
   - Passes top sources to LLM
   - Generates answer with source citations
   - Estimates confidence based on sources
   - Extracts and displays citations

5. **Evaluate**: Assess retrieval quality
   - Precision@K, Recall@K metrics
   - NDCG (normalized discounted cumulative gain)
   - Citation correctness evaluation
   - Hallucination detection

## ⚙️ Configuration

### Environment Variables

Key configuration variables (see `.env.example`):

```bash
# LLM Configuration
LLM_PROVIDER=openai              # openai, anthropic, ollama
LLM_MODEL=gpt-4

# OpenAI (if using)
OPENAI_API_KEY=your_key_here

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Retrieval Configuration
BM25_K1=1.5
BM25_B=0.75
RETRIEVAL_TOP_K=10
HYBRID_ALPHA=0.5                # 0=BM25 only, 1=Dense only, 0.5=equal
FUSION_STRATEGY=weighted         # weighted or rrf
RRF_K=60

# Chunking
CHUNKING_STRATEGY=recursive      # fixed, sentence, recursive
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_TOP_K=5

# Directories
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
INDEXES_DIR=./data/indexes

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### LLM Integration

The system supports multiple LLM providers. Configure via environment:

**Gemini (development default):**
```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

The Ask response reports whether a provider generated the answer or the system returned a retrieval-only evidence summary. Provider-generated answers include the configured and used model; retrieval-only summaries are explicitly labeled and include `[Source N]` evidence markers.

Gemini uses the current OpenAI-compatible Gemini endpoint without sending legacy sampling parameters such as `temperature`, `top_p`, or `top_k`.

**OpenRouter:**
```bash
LLM_PROVIDER=openrouter
LLM_MODEL=openrouter/free
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

**OpenAI:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

**Anthropic Claude:**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

**Local Ollama:**
```bash
LLM_PROVIDER=ollama
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

**Fallback Mode:**
If no LLM is configured or API fails, the system operates in retrieval-only mode, returning concatenated source summaries.

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_retrieval.py

# Specific test
pytest tests/test_retrieval.py::TestBM25Retriever::test_build_and_retrieve

# Integration tests only
pytest -m integration
```

### Test Coverage

The test suite includes:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline testing
- **Fixtures**: Sample documents and chunks for testing

Current coverage includes:
- Document ingestion and parsing
- Document chunking
- Metadata extraction
- Retrieval (BM25, Dense, Hybrid)
- Reranking
- Evaluation metrics

## 📊 Evaluation Framework

### Retrieval Metrics

The system evaluates retrieval quality using:

- **Precision@K** - Fraction of top-k results that are relevant
- **Recall@K** - Fraction of relevant documents found in top-k
- **NDCG@K** - Discounted cumulative gain accounting for ranking
- **MRR** - Mean reciprocal rank (position of first relevant result)
- **Hit Rate@K** - Whether any relevant document is in top-k

### Answer Quality Evaluation

- **Citation Coverage** - How well citations cover the answer
- **Claim Support** - Sentence-level lexical support from retrieved evidence
- **Retrieval-Source Agreement** - Whether displayed citations refer to retrieved chunks
- **Confidence Scoring** - Modular estimate for answers; retrieval scores are not represented as probabilities

### Using Evaluation

```python
from src.evaluation import RetrievalMetrics

relevant_ids = ["doc1", "doc2", "doc3"]
retrieved_ids = ["doc1", "doc3", "doc4"]

metrics = RetrievalMetrics.evaluate_retrieval(
    relevant_ids, 
    retrieved_ids,
    k_values=[5, 10]
)
# Returns: precision@5, recall@5, ndcg@5, hit_rate@5, mrr, ...
```

## 🔄 Workflow Examples

### Example 1: Simple Question Answering

```python
from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
from src.generation import LLMClient, PromptTemplate, GroundingExtractor
from pathlib import Path

# Load documents
docs = DocumentLoader.load_documents(Path("data/documents"))

# Chunk
chunker = DocumentChunker()
chunks = []
for doc in docs:
    chunks.extend(chunker.chunk_document(doc.content, doc.doc_id))

# Build index
index_mgr = IndexManager(Path("data/indexes"))
retriever = index_mgr.create_index("main", chunks)

# Retrieve
query = "What is machine learning?"
results = retriever.retrieve(query, top_k=5)

# Rerank
reranker = CrossEncoderReranker()
top_results = reranker.rerank(query, results, top_k=3)

# Generate
llm = LLMClient()
prompt = PromptTemplate.grounded_qa_prompt(query, top_results)
answer = llm.generate(prompt)

# Extract citations
citations = GroundingExtractor.extract_citations(answer, top_results)
print(f"Answer: {answer}\nCitations: {citations}")
```

### Example 2: Batch Evaluation

```python
from src.evaluation import EvaluationFramework

# Evaluate multiple queries
queries_with_relevant = [
    {
        "query": "machine learning",
        "relevant_docs": ["doc1", "doc3"]
    },
    {
        "query": "neural networks", 
        "relevant_docs": ["doc2", "doc4"]
    }
]

results_per_query = []
for item in queries_with_relevant:
    retrieved = retriever.retrieve(item["query"], top_k=10)
    retrieved_ids = [r.doc_id for r in retrieved]
    
    metrics = EvaluationFramework.evaluate_retrieval(
        item["relevant_docs"],
        retrieved_ids,
        k_values=[1, 5, 10]
    )
    results_per_query.append(metrics)

# Aggregate results
avg_precision_5 = sum(r["precision@5"] for r in results_per_query) / len(results_per_query)
print(f"Average Precision@5: {avg_precision_5:.3f}")
```

### Reproducible Demonstration Benchmark

The versioned [project benchmark](data/benchmarks/project_benchmark.json) contains a small labeled corpus and query relevance labels for local regression checks. It is deliberately a demonstration benchmark, not a production-scale quality claim.

```bash
python -m src.evaluation.run_benchmark --output data/experiments
```

The command runs BM25, dense, weighted hybrid, RRF hybrid, and weighted hybrid plus reranker against the same labels. Each configuration writes `experiment.json`, `results.json`, `per_query.csv`, and `report.md` under `data/experiments/`.

Experiments can be represented directly with `RetrievalExperimentConfig`, which captures the embedding model, chunking method, BM25 settings, candidate depths, fusion settings, reranker settings, and graph-expansion settings.

## 📈 Performance Characteristics

### Typical Latencies (on Windows with CPU)
- Document ingestion (single PDF): 100-500ms
- Indexing (100 chunks): 5-15 seconds
- Hybrid retrieval query: 50-200ms
- Cross-encoder reranking (10 results): 100-300ms
- LLM answer generation: 1-30 seconds (depends on provider)

### Scalability Notes
- BM25 index: Linear memory with chunk count
- Dense index (FAISS): ~4 bytes × 384 dimensions × chunk count
- Typical 100 PDFs → ~10-50K chunks → ~150-200MB indexes

### Optimization Tips
1. Adjust chunk size based on content (smaller = more chunks)
2. Use `hybrid_alpha=1.0` for dense-only (faster semantic search)
3. Disable reranking for speed, enable for quality
4. Use smaller embedding models for speed
5. Cache embeddings and indexes

## 🔐 Security Notes

- ⚠️ **Never commit .env files** with API keys
- Configure credentials via environment variables only
- Use `.gitignore` to exclude sensitive data
- For production, use secrets management
- Sanitize user inputs before LLM processing

## 📚 API Documentation

Full API documentation available at: `http://localhost:8000/docs` (interactive Swagger UI)

### Main Endpoints

- `GET /health` - Health check
- `GET /info` - System information
- `POST /ingest` - Upload and ingest document
- `POST /index` - Create retrieval index
- `GET /indexes` - List available indexes
- `POST /retrieve` - Retrieve documents
- `POST /answer` - Generate grounded answer

## 🚧 Roadmap & Limitations

### Currently Implemented ✅
- Document ingestion (PDF, TXT, MD)
- Recursive document chunking
- BM25 sparse retrieval
- Dense retrieval with FAISS
- Hybrid retrieval with configurable fusion
- Reciprocal Rank Fusion (RRF) and explicit normalized weighted fusion
- Cross-encoder reranking
- Provider-agnostic LLM integration
- Citation extraction and display
- Confidence scoring
- Retrieval evaluation metrics
- Knowledge graph structures
- Configurable citation-graph candidate expansion
- Reproducible labeled demonstration benchmark with JSON, CSV, and Markdown artifacts
- FastAPI backend
- Streamlit UI
- Comprehensive test suite

### Planned/Future Enhancements ⏳
- Query decomposition and rewriting
- Multi-turn conversational support
- Advanced query understanding
- Semantic query expansion
- Graph-based retrieval using citation relationships
- Batch evaluation and experiment tracking
- Fine-tuned local embeddings
- Advanced hallucination detection
- User authentication and multi-user support
- Production vector database integration
- Advanced visualization dashboards
- Evaluation result persistence and comparison

### Known Limitations ⚠️
- LLM generation requires external API or local model
- Dense retrieval requires sentence-transformers download
- FAISS CPU may be slow for very large corpora (100K+ chunks)
- Single-machine only (no distributed indexing)
- No authentication/authorization in current version
- Streamlit UI suitable for development/demo, not production

## 🛠️ Development

### Code Style
- Type hints throughout
- Docstrings for all modules
- Pydantic models for data validation
- Comprehensive error handling
- Structured logging

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Adding Features
1. Create new module in appropriate package
2. Add unit tests in `tests/test_*.py`
3. Document in docstrings and README
4. Run test suite to ensure no regressions

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Follow code style (type hints, docstrings)
2. Add tests for new features
3. Update README if adding major features
4. Ensure all tests pass

## 📞 Support

For issues or questions:
1. Check existing documentation
2. Search test suite for usage examples
3. Open an issue with detailed description

---

**Research Intelligence System v0.1.0** - A production-oriented research intelligence platform
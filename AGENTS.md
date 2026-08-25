# Research Intelligence System — Agent Instructions

## Project Goal

Build a production-oriented research intelligence and document reasoning platform.

The system should allow users to ingest scientific/technical documents, search them using hybrid lexical + semantic retrieval, rerank retrieved evidence, generate grounded answers with source citations, and provide evaluation and reliability tooling.

The project should evolve beyond a basic "RAG chatbot" into a modular information-retrieval and AI reasoning system.

## Primary Technical Goals

The long-term architecture should support:

1. Document ingestion and parsing
2. Structure-aware document chunking
3. Metadata extraction and storage
4. BM25 sparse retrieval
5. Dense embedding retrieval
6. Hybrid retrieval
7. Approximate nearest-neighbor vector search
8. Transformer/cross-encoder reranking
9. LLM-based grounded answer generation
10. Source/citation attribution
11. Retrieval evaluation
12. Answer/grounding evaluation
13. Confidence and reliability estimation
14. Query rewriting/decomposition
15. Citation/research knowledge graph
16. Graph-based retrieval
17. Optional graph ML/GNN components
18. FastAPI backend
19. A usable web interface
20. Automated tests and reproducible experiments

## Architecture Principles

Keep major components modular and independently replaceable.

Use clear interfaces between:

- ingestion
- parsing
- chunking
- embeddings
- sparse retrieval
- dense retrieval
- hybrid retrieval
- reranking
- generation
- evaluation
- graph functionality

Do not tightly couple business logic to a single framework.

Avoid unnecessary use of LangChain/LlamaIndex abstractions when straightforward Python implementations are clearer.

The system should be understandable by a developer reading the repository.

## Initial Technology Preferences

Primary language:
- Python 3.12

Backend:
- FastAPI

Frontend:
- Streamlit initially, unless a better lightweight architecture is clearly justified

Embeddings:
- sentence-transformers or another well-supported embedding library

Sparse retrieval:
- BM25 implementation

Dense retrieval:
- FAISS initially

Reranking:
- transformer cross-encoder/reranker

LLM:
- provider-agnostic interface
- API credentials must come from environment variables
- do not hard-code API keys

Storage:
- simple local persistence initially
- design interfaces so a production vector database can be substituted later

Graph:
- NetworkX initially
- design so graph functionality can later be upgraded to a graph database or GNN system

## Engineering Requirements

Use:
- type hints
- clear module boundaries
- meaningful docstrings
- sensible error handling
- configuration through environment variables
- structured logging where appropriate

Create:
- unit tests for important components
- integration tests for the retrieval pipeline
- reproducible scripts for indexing and evaluation

Do not fabricate benchmark numbers, accuracy, latency, or evaluation results.

If a feature is not actually implemented, do not claim it is implemented in documentation.

## Retrieval Design

The intended retrieval flow is:

Document ingestion
→ structure-aware chunking
→ metadata extraction
→ embedding generation
→ vector index
→ BM25 index

Query
→ query processing
→ BM25 retrieval
→ dense retrieval
→ candidate fusion
→ reranking
→ final evidence selection
→ grounded generation

The exact algorithms and weights should remain configurable.

## Evaluation Philosophy

Evaluation is a first-class component of the project.

The system should eventually support:

- Precision@k
- Recall@k
- Hit Rate@k
- MRR
- NDCG
- retrieval latency
- grounding/citation correctness
- answer quality
- hallucination/error analysis

Do not evaluate the entire system only by whether the generated answer "looks good."

## Reliability Philosophy

The system should distinguish:

- retrieval similarity
- relevance score
- confidence
- factual support

These are not automatically probabilities.

Confidence estimates should only be described as probabilistic after calibration/validation.

## Security

Never commit:
- API keys
- passwords
- tokens
- private credentials

Create `.env.example`.

Use `.gitignore` appropriately.

## Documentation

Maintain a README containing:

- project overview
- architecture
- setup instructions
- local development instructions
- usage
- API information
- evaluation methodology
- limitations
- roadmap

The README must clearly distinguish currently implemented functionality from planned functionality.

## Agent Behavior

Before making major architectural changes, inspect the repository and existing implementation.

Prefer incremental, testable implementation over generating a large amount of untested code.

When a command fails:
- diagnose the error
- fix the underlying issue
- rerun the relevant test

Do not silently work around errors.

Do not delete functional code merely to simplify implementation.

Do not invent dependencies if an existing dependency already provides the required capability.

Keep the project runnable at intermediate stages.

## Long-Term Project Direction

The finished project should demonstrate expertise across:

- information retrieval
- modern NLP
- LLM systems
- vector search
- ranking
- evaluation
- reliability
- statistical experimentation
- graph-based information retrieval

The project should feel like an AI information system rather than a tutorial RAG chatbot.
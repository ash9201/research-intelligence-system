"""
FastAPI backend application
"""
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import get_settings
from src.logging_config import setup_logging, get_logger
from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.metadata import MetadataExtractor
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
from src.generation import LLMClient, PromptTemplate, GroundingExtractor, ProviderRequestError, ProviderUnavailableError, normalize_evidence, retrieval_only_summary
from src.evaluation import GroundingEvaluator, ReliabilityEstimator
from src.models import GenerationStatus, RetrievalResponse, GeneratedAnswer, RetrievalResult

logger = get_logger(__name__)
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Research Intelligence System",
    description="Production-oriented research intelligence and document reasoning platform",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
current_retriever = None
current_reranker = None
index_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    setup_logging()
    logger.info("Research Intelligence System backend starting up")
    
    global index_manager, current_retriever, current_reranker
    
    # Initialize managers
    index_manager = IndexManager(settings.indexes_dir)
    current_reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    
    # Try to load default index if it exists
    indexes = index_manager.list_indexes()
    if indexes:
        try:
            current_retriever = index_manager.load_index(indexes[0])
            logger.info(f"Loaded index: {indexes[0]}")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Research Intelligence System backend shutting down")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Research Intelligence System"}


@app.get("/info")
async def get_info():
    """Get system information"""
    indexes = index_manager.list_indexes() if index_manager else []
    
    return {
        "version": "0.1.0",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "available_indexes": indexes,
        "current_index": None,  # Track separately
    }


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document"""
    try:
        # Save uploaded file temporarily
        temp_path = settings.documents_dir / file.filename
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load and parse document
        doc = DocumentLoader.load_document(temp_path)
        
        logger.info(f"Ingested document: {doc.doc_id}")
        
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "size_chars": len(doc.content),
            "file_type": doc.file_type,
        }
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/index")
async def create_index(index_name: str):
    """
    Create a retrieval index from ingested documents
    
    This will:
    1. Load all documents from data/documents
    2. Chunk them
    3. Extract metadata
    4. Build BM25 and dense indexes
    """
    global current_retriever
    
    try:
        logger.info(f"Creating index: {index_name}")
        
        # Load all documents
        documents = DocumentLoader.load_documents(settings.documents_dir)
        if not documents:
            raise ValueError("No documents found in documents directory")
        
        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            strategy=settings.chunking_strategy,
        )
        all_chunks = []
        for doc in documents:
            chunk_metadata = {
                "title": doc.title,
                **{key: value for key, value in doc.metadata.items() if key != "pages"},
            }
            pages = doc.metadata.get("pages") if doc.file_type == "pdf" else None
            chunks = (
                chunker.chunk_pages(pages, doc.doc_id, metadata=chunk_metadata)
                if pages
                else chunker.chunk_document(doc.content, doc.doc_id, metadata=chunk_metadata)
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Enrich metadata
        for chunk in all_chunks:
            MetadataExtractor.enrich_chunk_metadata(chunk)
        
        # Build indexes
        current_retriever = index_manager.create_index(
            index_name,
            all_chunks,
            embedding_model=settings.embedding_model,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
            fusion_strategy=settings.fusion_strategy,
            hybrid_alpha=settings.hybrid_alpha,
            rrf_k=settings.rrf_k,
        )
        
        logger.info(f"Index created successfully: {index_name}")
        
        return {
            "index_name": index_name,
            "document_count": len(documents),
            "chunk_count": len(all_chunks),
            "status": "created",
        }
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/indexes")
async def list_indexes():
    """List available indexes"""
    if not index_manager:
        return {"indexes": []}
    
    indexes = index_manager.list_indexes()
    return {"indexes": indexes}


@app.post("/retrieve")
async def retrieve(
    query: str,
    top_k: int = 10,
    use_reranking: bool = True,
):
    """
    Retrieve relevant documents for a query
    """
    global current_retriever, current_reranker
    
    if current_retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No index loaded. Create an index first using /index endpoint.",
        )
    
    try:
        start_time = time.time()
        
        # Retrieve
        logger.info(f"Retrieving documents for: {query}")
        initial_results = current_retriever.retrieve(query, top_k=top_k * 2)
        
        # Rerank if requested
        if use_reranking and current_reranker:
            logger.info("Reranking results...")
            reranked_results = current_reranker.rerank(
                query,
                [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        content=r.content,
                        score=r.score,
                        retrieval_method=r.retrieval_method,
                    )
                    for r in initial_results
                ],
                top_k=top_k,
            )
            # Convert back to RetrievalResult
            results = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.relevance_score,
                    retrieval_method="hybrid_reranked",
                )
                for r in reranked_results
            ]
        else:
            results = initial_results[:top_k]
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        response = RetrievalResponse(
            query=query,
            results=results,
            total_count=len(results),
            retrieval_time_ms=retrieval_time_ms,
        )
        
        logger.info(f"Retrieved {len(results)} documents in {retrieval_time_ms:.2f}ms")
        
        return response.dict()
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/answer")
async def generate_answer(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True,
):
    """
    Generate a grounded answer for a query
    """
    global current_retriever, current_reranker
    
    if current_retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No index loaded. Create an index first using /index endpoint.",
        )
    
    try:
        start_time = time.time()
        
        # Retrieve and rerank
        logger.info(f"Generating answer for: {query}")
        initial_results = current_retriever.retrieve(query, top_k=top_k * 2)
        
        if use_reranking and current_reranker:
            reranked_results = current_reranker.rerank(
                query,
                [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        doc_id=r.doc_id,
                        content=r.content,
                        score=r.score,
                        retrieval_method=r.retrieval_method,
                    )
                    for r in initial_results
                ],
                top_k=top_k,
            )
        else:
            reranked_results = initial_results[:top_k]
        
        # Normalize the exact selected evidence, regardless of whether it was reranked.
        chunk_metadata = {
            chunk.chunk_id: chunk.metadata
            for chunk in current_retriever.bm25.chunk_map.values()
        }
        evidence_sources = normalize_evidence(reranked_results, chunk_metadata)
        prompt = PromptTemplate.grounded_qa_prompt(
            query,
            evidence_sources,
        )

        llm = LLMClient()
        try:
            provider_result = llm.generate(prompt, temperature=0.7, max_tokens=500)
            answer_text = provider_result.text
            generation_status = GenerationStatus(
                generation_mode="provider",
                provider=provider_result.provider,
                configured_model=settings.llm_model,
                used_model=provider_result.used_model,
                provider_status="generated",
            )
        except ProviderUnavailableError as error:
            logger.info("Provider unavailable (%s); returning retrieval-only evidence summary", error)
            answer_text = retrieval_only_summary(evidence_sources)
            generation_status = GenerationStatus(
                generation_mode="fallback",
                provider=llm.provider,
                configured_model=llm.model,
                fallback_reason=str(error),
                provider_status="unavailable",
                grounding_status="fallback_evidence_summary",
            )
        except ProviderRequestError as error:
            logger.info("Provider response was unusable (%s); returning retrieval-only evidence summary", error)
            answer_text = retrieval_only_summary(evidence_sources)
            generation_status = GenerationStatus(
                generation_mode="fallback",
                provider=llm.provider,
                configured_model=llm.model,
                fallback_reason=str(error),
                provider_status="empty_response" if str(error) == "provider_returned_empty_content" else "request_failed",
                grounding_status="fallback_evidence_summary",
            )
        except Exception as error:
            logger.warning("Provider request failed (%s); returning retrieval-only evidence summary", type(error).__name__)
            answer_text = retrieval_only_summary(evidence_sources)
            generation_status = GenerationStatus(
                generation_mode="fallback",
                provider=llm.provider,
                configured_model=llm.model,
                fallback_reason="provider_request_failed",
                provider_status="request_failed",
                grounding_status="fallback_evidence_summary",
            )
        
        # Extract citations
        citations = GroundingExtractor.extract_citations(
            answer_text,
            evidence_sources,
        )
        if generation_status.generation_mode == "provider" and _states_insufficient_evidence(answer_text):
            citations = []
            generation_status.grounding_status = "insufficient_evidence"
        elif generation_status.generation_mode == "provider" and (
            not citations or _has_invalid_source_markers(answer_text, len(evidence_sources))
        ):
            # Missing citation syntax is a grounding-quality failure, not a generation failure.
            # Preserve coherent provider text so the user can inspect the response honestly.
            generation_status.grounding_status = "citation_missing"
        elif generation_status.generation_mode == "provider":
            supported_citations, unsupported_citations = GroundingEvaluator.validate_citation_support(
                answer_text,
                citations,
            )
            if unsupported_citations:
                logger.warning("Provider answer cited unsupported claims; returning retrieval-only evidence summary")
                answer_text = retrieval_only_summary(evidence_sources)
                generation_status = GenerationStatus(
                    generation_mode="fallback",
                    provider=llm.provider,
                    configured_model=llm.model,
                    fallback_reason="provider_response_citations_not_supported",
                    provider_status="generated",
                    grounding_status="unsupported_claim",
                )
                citations = GroundingExtractor.extract_citations(answer_text, evidence_sources)
            else:
                citations = supported_citations
                generation_status.grounding_status = "grounded"
        elif generation_status.grounding_status == "not_evaluated":
            generation_status.grounding_status = "fallback_evidence_summary"
        grounding_score = GroundingEvaluator.evaluate_retrieval_source_agreement(
            [source.chunk_id for source in evidence_sources], citations,
        )
        reliability = ReliabilityEstimator.answer_reliability(
            [source.score for source in evidence_sources],
            citations,
            len(evidence_sources),
            grounding_score,
            provider_generated=generation_status.generation_mode == "provider",
        )
        
        generation_time_ms = (time.time() - start_time) * 1000
        
        response = GeneratedAnswer(
            query=query,
            answer=GroundingExtractor.clean_answer(answer_text),
            citations=citations,
            model=generation_status.used_model or generation_status.configured_model,
            generation_time_ms=generation_time_ms,
            confidence_score=reliability.reliability_indicator,
            generation_status=generation_status,
            evidence_sources=evidence_sources,
            reliability=reliability,
        )
        
        logger.info(f"Generated answer in {generation_time_ms:.2f}ms")
        
        return response.dict()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=400, detail=str(e))


def _states_insufficient_evidence(answer: str) -> bool:
    """Allow uncited provider answers only when they explicitly decline unsupported claims."""
    normalized = answer.lower()
    return any(phrase in normalized for phrase in (
        "evidence is insufficient",
        "evidence does not specify",
        "sources do not mention",
        "do not mention the specific",
    ))


def _has_invalid_source_markers(answer: str, source_count: int | None = None) -> bool:
    """Detect grouped, malformed, or out-of-range source markers."""
    import re
    for match in re.finditer(r"\[([^\]]*Source[^\]]*)\]", answer):
        marker = match.group(1)
        if not re.fullmatch(r"Source \d+", marker):
            return True
        if source_count is not None and int(marker.split()[1]) > source_count:
            return True
    return False


def main():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

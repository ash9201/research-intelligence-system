"""Focused Ask-pipeline tests. Provider calls use controlled local fakes only."""
from types import SimpleNamespace
import asyncio
import importlib
import pytest

from src.config import Settings
from src.evaluation import ReliabilityEstimator
from src.generation import LLMClient, GroundingExtractor, PromptTemplate, normalize_evidence, retrieval_only_summary
from src.models import RetrievalResult, RerankingResult
from src.generation.llm_client import ProviderRequestError, ProviderUnavailableError, _response_text


def _fake_openai_client(**kwargs):
    _fake_openai_client.kwargs = kwargs
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Alpha supports a claim. [Source 1]"))]
                )
            )
        )
    )


def _fake_gemini_client(**kwargs):
    _fake_gemini_client.kwargs = kwargs
    def create(**request):
        _fake_gemini_client.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Gemini answer. [Source 1]"))]
        )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )


def _selected_results():
    return [
        RetrievalResult(chunk_id="chunk_a", doc_id="doc_a", content="Alpha supports a claim. More evidence follows.", score=0.8, retrieval_method="hybrid"),
        RerankingResult(chunk_id="chunk_b", doc_id="doc_b", content="Beta supports another claim.", relevance_score=0.9, original_rank=1, new_rank=0),
    ]


def test_configuration_loads_explicit_env_file(tmp_path, monkeypatch):
    """Settings can load provider values from an explicit .env file without terminal injection."""
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openrouter\nLLM_MODEL=test/model\nOPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    for key in ("LLM_PROVIDER", "LLM_MODEL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "test/model"
    assert settings.openrouter_api_key == "test-key"


def test_openrouter_success_uses_controlled_client(monkeypatch):
    """OpenRouter uses the OpenAI-compatible client with its configured base URL."""
    client = LLMClient(provider="openrouter", model="openrouter/free", client_factory=_fake_openai_client)
    monkeypatch.setattr(client.settings, "openrouter_api_key", "test-key")
    result = client.generate("Evidence [Source 1]")

    assert result.provider == "openrouter"
    assert result.used_model == "openrouter/free"
    assert result.text.endswith("[Source 1]")
    assert _fake_openai_client.kwargs["base_url"] == "https://openrouter.ai/api/v1"


def test_gemini_success_uses_current_compatible_interface(monkeypatch):
    """Gemini uses the tested OpenAI-compatible endpoint without legacy sampling args."""
    client = LLMClient(provider="gemini", model="gemini-3.5-flash", client_factory=_fake_gemini_client)
    monkeypatch.setattr(client.settings, "gemini_api_key", "test-key")

    result = client.generate("Evidence [Source 1]", temperature=0.7, max_tokens=500)

    assert result.provider == "gemini"
    assert result.used_model == "gemini-3.5-flash"
    assert result.text == "Gemini answer. [Source 1]"
    assert _fake_gemini_client.kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert _fake_gemini_client.request["model"] == "gemini-3.5-flash"
    assert "temperature" not in _fake_gemini_client.request
    assert "top_p" not in _fake_gemini_client.request
    assert "top_k" not in _fake_gemini_client.request


def test_gemini_readiness_requires_gemini_credential(monkeypatch):
    """Gemini readiness is independent of OpenAI and OpenRouter credentials."""
    client = LLMClient(provider="gemini", model="gemini-3.5-flash", client_factory=_fake_gemini_client)
    monkeypatch.setattr(client.settings, "gemini_api_key", None)

    assert client.readiness() == (False, "credential_not_configured")


def test_provider_empty_response_is_a_distinct_failure():
    """An empty compatible-provider response is not treated as a grounded answer."""
    class EmptyResponse:
        choices = [SimpleNamespace(message=SimpleNamespace(content=""))]

    with pytest.raises(ProviderRequestError, match="provider_returned_empty_content"):
        _response_text(EmptyResponse())


def test_openai_readiness_requires_its_own_configuration(monkeypatch):
    """OpenAI remains independently configurable from the OpenRouter provider."""
    client = LLMClient(provider="openai", model="gpt-test", client_factory=_fake_openai_client)
    monkeypatch.setattr(client.settings, "openai_api_key", None)
    assert client.readiness() == (False, "credential_not_configured")


def test_openrouter_readiness_reports_missing_credential(monkeypatch):
    """OpenRouter reports a safe unavailable reason without inspecting a secret value."""
    client = LLMClient(provider="openrouter", model="openrouter/free")
    monkeypatch.setattr(client.settings, "openrouter_api_key", None)
    assert client.readiness() == (False, "credential_not_configured")


def test_normalized_evidence_and_fallback_preserve_boundaries_and_citations():
    """Both score shapes normalize, and fallback emits complete sentence source markers."""
    evidence = normalize_evidence(_selected_results(), {"chunk_a": {"title": "A", "pages": [1]}, "chunk_b": {"section": "B"}})
    summary = retrieval_only_summary(evidence)

    assert [source.score for source in evidence] == [0.8, 0.9]
    assert "Alpha supports a claim. [Source 1]" in summary
    assert "[Source 2]" in summary
    assert "- Alpha supports a claim. [Source 1]" in summary
    citations = GroundingExtractor.extract_citations(summary, evidence)
    assert [citation.chunk_id for citation in citations] == ["chunk_a", "chunk_b"]


def test_strict_citations_reject_invalid_source_numbers_and_prompt_has_contract():
    """Only exact supplied [Source N] markers resolve to citations."""
    evidence = normalize_evidence(_selected_results(), {})
    prompt = PromptTemplate.grounded_qa_prompt("Question?", evidence)
    citations = GroundingExtractor.extract_citations("Claim [Source 1]. Invalid [Source 9].", evidence)

    assert "Place the exact marker [Source N]" in prompt
    assert "Never invent a source number" in prompt
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_a"

    repeated = GroundingExtractor.extract_citations("Twice [Source 1]. Again [Source 1].", evidence)
    assert len(repeated) == 1
    assert len(repeated[0].position_in_answer) == 2
    assert GroundingExtractor.clean_answer("Claim [Source 1]") == "Claim [Source 1]"


def test_reliability_is_not_a_fallback_answer_probability():
    """Fallback returns evidence indicators without a fabricated reliability value."""
    fallback = ReliabilityEstimator.answer_reliability([0.8, 0.9], [], 2, 0.0, provider_generated=False)
    provider = ReliabilityEstimator.answer_reliability([0.8], [], 1, 0.0, provider_generated=True)

    assert fallback.reliability_indicator is None
    assert fallback.score_type == "retrieval_only_evidence_indicators"
    assert provider.reliability_indicator == 0.0


def test_invalid_grouped_source_marker_is_detected():
    """Grouped markers are rejected because the public citation contract is [Source N]."""
    backend = importlib.import_module("src.backend.app")
    assert backend._has_invalid_source_markers("Claim [Source 1, Source 2]")
    assert not backend._has_invalid_source_markers("Claim [Source 1]")
    assert backend._has_invalid_source_markers("Claim [Source 2]", source_count=1)


def test_answer_endpoint_passes_path_selected_evidence_to_provider(monkeypatch):
    """Reranking ON/OFF produces a visibly different provider evidence set."""
    backend = importlib.import_module("src.backend.app")
    first = RetrievalResult(chunk_id="first", doc_id="doc", content="First evidence supports alpha.", score=0.8, retrieval_method="hybrid")
    second = RetrievalResult(chunk_id="second", doc_id="doc", content="Second evidence supports beta.", score=0.7, retrieval_method="hybrid")
    chunk_map = {
        0: SimpleNamespace(chunk_id="first", metadata={"title": "Paper", "pages": [1], "section": "One"}),
        1: SimpleNamespace(chunk_id="second", metadata={"title": "Paper", "pages": [2], "section": "Two"}),
    }
    backend.current_retriever = SimpleNamespace(retrieve=lambda query, top_k: [first, second], bm25=SimpleNamespace(chunk_map=chunk_map))
    backend.current_reranker = SimpleNamespace(
        rerank=lambda query, results, top_k: [
            RerankingResult(chunk_id="second", doc_id="doc", content="Second evidence supports beta.", relevance_score=0.9, original_rank=1, new_rank=0),
            RerankingResult(chunk_id="first", doc_id="doc", content="First evidence supports alpha.", relevance_score=0.8, original_rank=0, new_rank=1),
        ]
    )

    prompts = []
    class FakeProvider:
        provider = "openrouter"
        model = "test/model"
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            text = "Second evidence supports beta. [Source 1]" if "[Source 1]: Second evidence" in prompt else "First evidence supports alpha. [Source 1]"
            return SimpleNamespace(text=text, provider="openrouter", used_model="test/model")
    monkeypatch.setattr(backend, "LLMClient", FakeProvider)

    plain = asyncio.run(backend.generate_answer("alpha", top_k=2, use_reranking=False))
    reranked = asyncio.run(backend.generate_answer("alpha", top_k=2, use_reranking=True))

    assert [source["chunk_id"] for source in plain["evidence_sources"]] == ["first", "second"]
    assert [source["chunk_id"] for source in reranked["evidence_sources"]] == ["second", "first"]
    assert "[Source 1]: First evidence" in prompts[0]
    assert "[Source 1]: Second evidence" in prompts[1]
    assert plain["citations"][0]["chunk_id"] == "first"
    assert reranked["citations"][0]["chunk_id"] == "second"
    assert plain["generation_status"]["grounding_status"] == "grounded"
    assert reranked["generation_status"]["grounding_status"] == "grounded"


def test_answer_endpoint_marks_unavailable_provider_as_fallback(monkeypatch):
    """Unavailable generation returns retrieval-only evidence, citations, and no probability."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="Complete evidence sentence.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None
    class UnavailableProvider:
        provider = "openrouter"
        model = "openrouter/free"
        def generate(self, prompt, **kwargs):
            raise ProviderUnavailableError("credential_not_configured")
    monkeypatch.setattr(backend, "LLMClient", UnavailableProvider)

    response = asyncio.run(backend.generate_answer("Question", top_k=1, use_reranking=False))

    assert response["generation_status"]["generation_mode"] == "fallback"
    assert response["generation_status"]["fallback_reason"] == "credential_not_configured"
    assert "[Source 1]" in response["answer"]
    assert response["confidence_score"] is None
    assert response["reliability"]["score_type"] == "retrieval_only_evidence_indicators"


def test_provider_answer_without_citations_is_preserved(monkeypatch):
    """A coherent provider answer without markers is citation_missing, not fallback."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="Attention uses a scaled dot product.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None

    class UncitedProvider:
        provider = "gemini"
        model = "gemini-3.5-flash"

        def generate(self, prompt, **kwargs):
            return SimpleNamespace(
                text="The scaled dot product is divided by the square root of the key dimension.",
                provider="gemini",
                used_model="gemini-3.5-flash",
            )

    monkeypatch.setattr(backend, "LLMClient", UncitedProvider)
    response = asyncio.run(backend.generate_answer("Why scale?", top_k=1, use_reranking=False))

    assert response["answer"].startswith("The scaled dot product")
    assert response["generation_status"]["generation_mode"] == "provider"
    assert response["generation_status"]["grounding_status"] == "citation_missing"
    assert response["citations"] == []


def test_answer_endpoint_marks_empty_provider_response(monkeypatch):
    """An empty provider response is reported separately from missing configuration."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="Complete evidence sentence.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None
    class EmptyProvider:
        provider = "openrouter"
        model = "openrouter/free"
        def generate(self, prompt, **kwargs):
            raise ProviderRequestError("provider_returned_empty_content")
    monkeypatch.setattr(backend, "LLMClient", EmptyProvider)

    response = asyncio.run(backend.generate_answer("Question", top_k=1, use_reranking=False))

    assert response["generation_status"]["generation_mode"] == "fallback"
    assert response["generation_status"]["provider_status"] == "empty_response"
    assert response["generation_status"]["fallback_reason"] == "provider_returned_empty_content"


def test_provider_unsupported_claim_is_not_grounded(monkeypatch):
    """A syntactically valid citation with no claim overlap is rejected by the endpoint."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="The paper describes attention mechanisms.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None
    class UnsupportedProvider:
        provider = "openrouter"
        model = "openrouter/free"
        def generate(self, prompt, **kwargs):
            return SimpleNamespace(text="The implementation used Python. [Source 1]", provider="openrouter", used_model="openrouter/free")
    monkeypatch.setattr(backend, "LLMClient", UnsupportedProvider)

    response = asyncio.run(backend.generate_answer("Which language?", top_k=1, use_reranking=False))

    assert response["generation_status"]["generation_mode"] == "fallback"
    assert response["generation_status"]["fallback_reason"] == "provider_response_citations_not_supported"
    assert response["reliability"]["reliability_indicator"] is None


def test_explicit_insufficient_evidence_clears_citations(monkeypatch):
    """An insufficiency response remains uncited even if it includes a source marker."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="The paper describes attention mechanisms.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None
    class InsufficientProvider:
        provider = "openrouter"
        model = "openrouter/free"
        def generate(self, prompt, **kwargs):
            return SimpleNamespace(text="The evidence is insufficient to identify the language. [Source 1]", provider="openrouter", used_model="openrouter/free")
    monkeypatch.setattr(backend, "LLMClient", InsufficientProvider)

    response = asyncio.run(backend.generate_answer("Which language?", top_k=1, use_reranking=False))

    assert response["generation_status"]["generation_mode"] == "provider"
    assert response["generation_status"]["grounding_status"] == "insufficient_evidence"
    assert response["citations"] == []
    assert response["reliability"]["reliability_indicator"] == 0.0


def test_invalid_provider_citation_syntax_triggers_fallback(monkeypatch):
    """Grouped source syntax is a grounding failure, not a valid citation set."""
    backend = importlib.import_module("src.backend.app")
    result = RetrievalResult(chunk_id="only", doc_id="doc", content="The paper describes attention mechanisms.", score=0.7, retrieval_method="hybrid")
    backend.current_retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [result],
        bm25=SimpleNamespace(chunk_map={0: SimpleNamespace(chunk_id="only", metadata={})}),
    )
    backend.current_reranker = None
    class InvalidCitationProvider:
        provider = "openrouter"
        model = "openrouter/free"
        def generate(self, prompt, **kwargs):
            return SimpleNamespace(text="Attention is discussed. [Source 1, Source 2]", provider="openrouter", used_model="openrouter/free")
    monkeypatch.setattr(backend, "LLMClient", InvalidCitationProvider)

    response = asyncio.run(backend.generate_answer("What is discussed?", top_k=1, use_reranking=False))

    assert response["generation_status"]["generation_mode"] == "provider"
    assert response["generation_status"]["grounding_status"] == "citation_missing"
    assert response["generation_status"]["fallback_reason"] is None
    assert response["citations"] == []
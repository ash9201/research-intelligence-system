"""Provider-agnostic LLM client abstraction."""
from dataclasses import dataclass
from typing import Callable, Optional

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class ProviderUnavailableError(RuntimeError):
    """Provider cannot be called due to missing package, credential, or configuration."""


class ProviderRequestError(RuntimeError):
    """Provider responded or failed without a usable generated answer."""


@dataclass(frozen=True)
class ProviderResult:
    text: str
    provider: str
    used_model: str


def _response_text(response) -> str:
    """Extract non-empty chat content or report a safe provider-response failure."""
    content = response.choices[0].message.content if response.choices else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderRequestError("provider_returned_empty_content")
    return content.strip()


class LLMClient:
    """Abstract LLM client interface"""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, client_factory: Optional[Callable] = None):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider (openai, anthropic, ollama)
            model: Model name
        """
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.settings = settings
        self.client_factory = client_factory

    def readiness(self) -> tuple[bool, Optional[str]]:
        """Check provider requirements safely without a network request."""
        if self.provider not in {"openai", "openrouter", "gemini", "anthropic", "ollama"}:
            return False, "unknown_provider"
        if self.provider in {"openai", "openrouter", "gemini"}:
            credential = {
                "openai": self.settings.openai_api_key,
                "openrouter": self.settings.openrouter_api_key,
                "gemini": self.settings.gemini_api_key,
            }[self.provider]
            if not credential:
                return False, "credential_not_configured"
            try:
                import openai  # noqa: F401
            except ImportError:
                return False, "provider_package_not_installed"
        if self.provider == "anthropic":
            if not self.settings.anthropic_api_key:
                return False, "credential_not_configured"
            try:
                import anthropic  # noqa: F401
            except ImportError:
                return False, "provider_package_not_installed"
        if self.provider == "ollama":
            try:
                import requests  # noqa: F401
            except ImportError:
                return False, "provider_package_not_installed"
        return True, None
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> ProviderResult:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        ready, reason = self.readiness()
        if not ready:
            raise ProviderUnavailableError(reason or "provider_unavailable")
        if self.provider == "openai":
            return self._generate_openai(prompt, temperature, max_tokens)
        elif self.provider == "openrouter":
            return self._generate_openrouter(prompt, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, temperature, max_tokens)
        else:
            raise ProviderUnavailableError("unknown_provider")
    
    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> ProviderResult:
        """Generate using OpenAI API"""
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise ProviderUnavailableError("provider_package_not_installed")
        client = self.client_factory(api_key=self.settings.openai_api_key) if self.client_factory else openai.OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return ProviderResult(_response_text(response), "openai", self.model)

    def _generate_openrouter(self, prompt: str, temperature: float, max_tokens: int) -> ProviderResult:
        """Generate through OpenRouter's OpenAI-compatible endpoint."""
        try:
            import openai
        except ImportError:
            raise ProviderUnavailableError("provider_package_not_installed")
        client = self.client_factory(base_url=self.settings.openrouter_base_url, api_key=self.settings.openrouter_api_key) if self.client_factory else openai.OpenAI(base_url=self.settings.openrouter_base_url, api_key=self.settings.openrouter_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a grounded research assistant."}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ProviderResult(_response_text(response), "openrouter", self.model)

    def _generate_gemini(self, prompt: str) -> ProviderResult:
        """Generate through Gemini's current OpenAI-compatible endpoint.

        Gemini's current compatibility path is intentionally called without legacy
        sampling parameters such as temperature, top_p, or top_k.
        """
        try:
            import openai
        except ImportError:
            raise ProviderUnavailableError("provider_package_not_installed")

        client = (
            self.client_factory(
                base_url=self.settings.gemini_base_url,
                api_key=self.settings.gemini_api_key,
            )
            if self.client_factory
            else openai.OpenAI(
                base_url=self.settings.gemini_base_url,
                api_key=self.settings.gemini_api_key,
            )
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a grounded research assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return ProviderResult(_response_text(response), "gemini", self.model)
    
    def _generate_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> ProviderResult:
        """Generate using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed. Install with: pip install anthropic")
            raise
        
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        
        return ProviderResult(message.content[0].text.strip(), "anthropic", self.model)
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> ProviderResult:
        """Generate using local Ollama API"""
        try:
            import requests
        except ImportError:
            logger.error("requests package not installed. Install with: pip install requests")
            raise
        
        settings = get_settings()
        url = f"{settings.ollama_api_url}/api/generate"
        
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return ProviderResult(result.get("response", "").strip(), "ollama", self.settings.ollama_model)
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

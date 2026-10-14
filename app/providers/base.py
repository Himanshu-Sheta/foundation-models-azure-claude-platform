from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas.models import ChatMessage, CompletionResponse, ToolDefinition


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for Foundation Model Providers.
    Ensures a standardized interface across Anthropic Claude, Microsoft Azure, and others.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'anthropic', 'azure_openai')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate response given a list of messages."""
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate response with tool/function calling support."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health and connectivity check."""
        pass

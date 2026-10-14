import logging
from typing import Any, Dict, List, Optional
from app.config import Settings
from app.core.exceptions import FoundationModelError, ProviderUnavailableError, SafetyPolicyViolationError
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.azure_openai import AzureOpenAIProvider
from app.providers.azure_safety import AzureContentSafetyProvider
from app.providers.base import BaseLLMProvider
from app.schemas.models import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ProviderHealthStatus,
    SafetyCheckResult,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Intelligent Foundation Model Router & High-Availability Gateway.
    Orchestrates routing between Anthropic Claude and Microsoft Azure AI Services.
    Handles automated failover, load balancing, and safety guardrails.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.anthropic = AnthropicProvider(settings)
        self.azure_openai = AzureOpenAIProvider(settings)
        self.safety_guard = AzureContentSafetyProvider(settings)

        self.providers: Dict[str, BaseLLMProvider] = {
            "anthropic": self.anthropic,
            "azure_openai": self.azure_openai,
        }

    def _determine_provider(self, request_provider: Optional[str], prompt_text: str) -> str:
        """Resolve target provider based on request preference or heuristic routing rules."""
        if request_provider and request_provider in self.providers:
            return request_provider

        # Rule-based auto routing:
        # Code generation, JSON structuring, deep reasoning -> prioritize Claude 3.5 Sonnet
        coding_signals = ["def ", "class ", "architecture", "refactor", "json schema", "sql", "explain step-by-step"]
        if any(sig in prompt_text.lower() for sig in coding_signals):
            return "anthropic"

        return self.settings.DEFAULT_PRIMARY_PROVIDER

    async def route_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Route request with pre-call safety guardrails and automatic failover.
        """
        # 1. Extract combined text for safety check
        combined_text = request.prompt or ""
        messages: List[ChatMessage] = []

        if request.messages:
            messages = request.messages
            combined_text += " " + " ".join(m.content for m in request.messages if m.role == "user")
        elif request.prompt:
            messages = [ChatMessage(role="user", content=request.prompt)]

        # 2. Azure AI Content Safety Guardrail Check
        if request.enable_safety_check and self.settings.ENABLE_SAFETY_GUARDRAILS:
            safety_result: SafetyCheckResult = await self.safety_guard.check_content(combined_text)
            if not safety_result.is_safe:
                logger.warning(f"Safety violation blocked: {safety_result.flagged_categories}")
                raise SafetyPolicyViolationError(
                    f"Content blocked by Azure Content Safety policy: {', '.join(safety_result.flagged_categories)}",
                    flagged_categories=safety_result.flagged_categories,
                )

        # 3. Determine target provider & fallback provider
        primary_name = self._determine_provider(request.provider, combined_text)
        fallback_name = (
            "azure_openai" if primary_name == "anthropic" else "anthropic"
        )

        # 4. Attempt primary provider
        try:
            primary_provider = self.providers[primary_name]
            if request.tools:
                resp = await primary_provider.generate_with_tools(
                    messages=messages,
                    tools=request.tools,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                )
            else:
                resp = await primary_provider.generate(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    system_prompt=request.system_prompt,
                )
            resp.metadata["routing_decision"] = f"primary:{primary_name}"
            return resp

        except Exception as primary_error:
            logger.error(
                f"Primary provider '{primary_name}' failed with: {primary_error}. Triggering automatic failover to '{fallback_name}'."
            )

            # 5. Attempt fallback provider
            try:
                fallback_provider = self.providers[fallback_name]
                if request.tools:
                    resp = await fallback_provider.generate_with_tools(
                        messages=messages,
                        tools=request.tools,
                        model=None,  # Use fallback's default model
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        system_prompt=request.system_prompt,
                    )
                else:
                    resp = await fallback_provider.generate(
                        messages=messages,
                        model=None,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        system_prompt=request.system_prompt,
                    )
                resp.metadata["failover_triggered"] = True
                resp.metadata["primary_failure_reason"] = str(primary_error)
                resp.metadata["routing_decision"] = f"failover:{fallback_name}"
                return resp

            except Exception as fallback_error:
                logger.critical(f"Both primary and fallback providers failed! Primary: {primary_error}, Fallback: {fallback_error}")
                raise ProviderUnavailableError(
                    f"All foundation model providers failed. Primary ({primary_name}): {primary_error} | Fallback ({fallback_name}): {fallback_error}"
                )

    async def get_all_health(self) -> List[ProviderHealthStatus]:
        """Aggregate health check across all registered foundation model providers."""
        statuses = []
        import datetime

        for name, prov in self.providers.items():
            check = await prov.health_check()
            statuses.append(
                ProviderHealthStatus(
                    provider=name,
                    status=check.get("status", "unknown"),
                    latency_ms=round(check.get("latency_ms", 0.0), 2),
                    last_checked=datetime.datetime.utcnow().isoformat(),
                )
            )
        return statuses

import time
import json
import logging
from typing import Any, Dict, List, Optional
from app.config import Settings
from app.core.exceptions import ProviderAuthenticationError, RateLimitExceededError, FoundationModelError
from app.core.telemetry import global_tracker
from app.providers.base import BaseLLMProvider
from app.schemas.models import ChatMessage, CompletionResponse, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Microsoft Azure OpenAI Foundation Model Provider.
    Implements GPT-4o and Text Embedding models hosted within Azure enterprise tenant.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._provider_name = "azure_openai"
        self._default_model = settings.AZURE_OPENAI_CHAT_DEPLOYMENT
        self._embedding_model = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self._init_client()

    def _init_client(self):
        if (
            self.settings.AZURE_OPENAI_API_KEY
            and self.settings.AZURE_OPENAI_ENDPOINT
            and not self.settings.ENABLE_SIMULATION_MODE
        ):
            try:
                from openai import AsyncAzureOpenAI
                self._client = AsyncAzureOpenAI(
                    api_key=self.settings.AZURE_OPENAI_API_KEY,
                    api_version=self.settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
                )
                logger.info("Azure OpenAI client initialized successfully.")
            except ImportError:
                logger.warning("openai package not installed or incompatible, running in simulation mode.")
                self._client = None
        else:
            logger.info("Azure OpenAI credentials not fully configured or simulation enabled. Running in simulation mode.")
            self._client = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    def _convert_messages(self, messages: List[ChatMessage], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        azure_msgs = []
        if system_prompt:
            azure_msgs.append({"role": "system", "content": system_prompt})

        for msg in messages:
            azure_msgs.append({"role": msg.role, "content": msg.content})

        if not azure_msgs:
            azure_msgs.append({"role": "user", "content": "Hello"})
        return azure_msgs

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters if t.parameters else {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        model_name = model or self.default_model
        start_time = time.perf_counter()

        if not self._client:
            return self._simulate_response(messages, model_name, start_time)

        try:
            openai_messages = self._convert_messages(messages, system_prompt)

            response = await self._client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = global_tracker.record_request(
                provider=self.provider_name,
                model=model_name,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

            return CompletionResponse(
                provider=self.provider_name,
                model=model_name,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                estimated_cost_usd=cost,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            global_tracker.record_request(
                provider=self.provider_name,
                model=model_name,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_type=type(e).__name__,
            )
            err_msg = str(e)
            if "auth" in err_msg.lower() or "401" in err_msg:
                raise ProviderAuthenticationError(f"Azure OpenAI auth failed: {err_msg}", provider=self.provider_name)
            elif "429" in err_msg:
                raise RateLimitExceededError(f"Azure OpenAI rate limit: {err_msg}", provider=self.provider_name)
            raise FoundationModelError(f"Azure OpenAI error: {err_msg}", provider=self.provider_name)

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
        model_name = model or self.default_model
        start_time = time.perf_counter()

        if not self._client:
            return self._simulate_tool_response(messages, tools, model_name, start_time)

        try:
            openai_messages = self._convert_messages(messages, system_prompt)
            converted_tools = self._convert_tools(tools)

            response = await self._client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                tools=converted_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = global_tracker.record_request(
                provider=self.provider_name,
                model=model_name,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

            return CompletionResponse(
                provider=self.provider_name,
                model=model_name,
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                estimated_cost_usd=cost,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            global_tracker.record_request(
                provider=self.provider_name,
                model=model_name,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_type=type(e).__name__,
            )
            raise FoundationModelError(f"Azure OpenAI tool error: {str(e)}", provider=self.provider_name)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding using Azure OpenAI text-embedding-3."""
        if not self._client:
            # Deterministic simulated 1536-dim normalized vector
            import hashlib
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (10**8)
            import random
            rng = random.Random(seed)
            raw = [rng.gauss(0, 1) for _ in range(1536)]
            norm = sum(x**2 for x in raw) ** 0.5
            return [x / norm for x in raw]

        res = await self._client.embeddings.create(input=[text], model=self._embedding_model)
        return res.data[0].embedding

    async def health_check(self) -> Dict[str, Any]:
        start = time.perf_counter()
        if not self._client:
            return {
                "provider": self.provider_name,
                "status": "simulation_ready",
                "model": self.default_model,
                "latency_ms": 1.1,
                "message": "Running in high-fidelity simulation mode.",
            }
        try:
            res = await self.generate([ChatMessage(role="user", content="ping")], max_tokens=5)
            return {
                "provider": self.provider_name,
                "status": "healthy",
                "model": self.default_model,
                "latency_ms": res.latency_ms,
            }
        except Exception as e:
            return {
                "provider": self.provider_name,
                "status": "degraded",
                "model": self.default_model,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "error": str(e),
            }

    def _simulate_response(
        self, messages: List[ChatMessage], model_name: str, start_time: float
    ) -> CompletionResponse:
        user_prompt = messages[-1].content if messages else "Hello"
        latency_ms = (time.perf_counter() - start_time) * 1000 + 40.0
        content = (
            f"[Microsoft Azure OpenAI | GPT-4o Simulation]\n"
            f"Received request: '{user_prompt}'\n\n"
            f"Microsoft Azure AI provides enterprise-grade compliance (HIPAA, ISO 27001), private virtual networks (VNet), "
            f"and high-throughput inference for mission-critical enterprise workloads."
        )
        input_tokens = len(user_prompt.split()) * 2 + 10
        output_tokens = len(content.split()) * 2

        cost = global_tracker.record_request(
            provider=self.provider_name,
            model=model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )

        return CompletionResponse(
            provider=self.provider_name,
            model=model_name,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=cost,
            metadata={"simulation": True},
        )

    def _simulate_tool_response(
        self, messages: List[ChatMessage], tools: List[ToolDefinition], model_name: str, start_time: float
    ) -> CompletionResponse:
        user_prompt = messages[-1].content if messages else ""
        tool_to_call = tools[0] if tools else None

        tool_calls = []
        if tool_to_call:
            mock_args = {"query": user_prompt}
            tool_calls.append(
                ToolCall(
                    id="call_azure_98231fa89e",
                    name=tool_to_call.name,
                    arguments=mock_args,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000 + 48.0
        input_tokens = len(user_prompt.split()) * 2 + 25
        output_tokens = 45

        cost = global_tracker.record_request(
            provider=self.provider_name,
            model=model_name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )

        return CompletionResponse(
            provider=self.provider_name,
            model=model_name,
            content="",
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=cost,
            metadata={"simulation": True},
        )

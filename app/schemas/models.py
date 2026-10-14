from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    class ChatMessage(BaseModel):
        role: str = Field(..., description="Message author role: system, user, assistant, or tool")
        content: str = Field(..., description="Text content of the message")
        name: Optional[str] = Field(None, description="Optional name for tool result messages")

    class ToolParameterProperty(BaseModel):
        type: str
        description: Optional[str] = None
        enum: Optional[List[str]] = None

    class ToolDefinition(BaseModel):
        name: str = Field(..., description="Unique tool name")
        description: str = Field(..., description="Detailed description of tool purpose")
        parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for parameters")

    class ToolCall(BaseModel):
        id: str
        name: str
        arguments: Dict[str, Any]

    class CompletionRequest(BaseModel):
        prompt: Optional[str] = None
        messages: Optional[List[ChatMessage]] = None
        provider: Optional[str] = Field(None, description="'anthropic', 'azure_openai', or 'auto'")
        model: Optional[str] = None
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        max_tokens: int = Field(1024, ge=1, le=8192)
        system_prompt: Optional[str] = None
        tools: Optional[List[ToolDefinition]] = None
        enable_safety_check: bool = True
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class CompletionResponse(BaseModel):
        provider: str
        model: str
        content: str
        tool_calls: Optional[List[ToolCall]] = None
        input_tokens: int = 0
        output_tokens: int = 0
        latency_ms: float = 0.0
        estimated_cost_usd: float = 0.0
        cached_tokens: int = 0
        safety_passed: bool = True
        safety_details: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class RAGQueryRequest(BaseModel):
        query: str
        top_k: int = Field(3, ge=1, le=20)
        provider: Optional[str] = "anthropic"
        filter_expression: Optional[str] = None
        enable_hybrid: bool = True

    class RAGDocument(BaseModel):
        id: str
        title: str
        content: str
        score: float
        source: Optional[str] = None

    class RAGQueryResponse(BaseModel):
        query: str
        answer: str
        retrieved_documents: List[RAGDocument]
        provider_used: str
        latency_ms: float
        tokens_used: int

    class SafetyCheckResult(BaseModel):
        is_safe: bool
        flagged_categories: List[str] = Field(default_factory=list)
        risk_scores: Dict[str, float] = Field(default_factory=dict)
        action_taken: str = "allow"

    class EvaluationRequest(BaseModel):
        prompt: str
        response: str
        reference_answer: Optional[str] = None
        criteria: List[str] = Field(default=["faithfulness", "relevance", "clarity", "safety"])

    class EvaluationMetricResult(BaseModel):
        criterion: str
        score: float = Field(..., ge=0.0, le=10.0)
        reasoning: str

    class EvaluationResponse(BaseModel):
        overall_score: float
        passed: bool
        metrics: List[EvaluationMetricResult]
        evaluator_model: str
        latency_ms: float

    class ProviderHealthStatus(BaseModel):
        provider: str
        status: str
        latency_ms: float
        last_checked: str

except ImportError:
    # Minimal dataclass / dict-like shim for testing before pip install
    class ChatMessage:
        def __init__(self, role: str, content: str, name: Optional[str] = None):
            self.role = role
            self.content = content
            self.name = name

    class ToolDefinition:
        def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
            self.name = name
            self.description = description
            self.parameters = parameters or {}

    class ToolCall:
        def __init__(self, id: str, name: str, arguments: Dict[str, Any]):
            self.id = id
            self.name = name
            self.arguments = arguments

    class CompletionRequest:
        def __init__(
            self,
            prompt: Optional[str] = None,
            messages: Optional[List[ChatMessage]] = None,
            provider: Optional[str] = None,
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            system_prompt: Optional[str] = None,
            tools: Optional[List[ToolDefinition]] = None,
            enable_safety_check: bool = True,
            metadata: Optional[Dict[str, Any]] = None,
        ):
            self.prompt = prompt
            self.messages = messages
            self.provider = provider
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.system_prompt = system_prompt
            self.tools = tools
            self.enable_safety_check = enable_safety_check
            self.metadata = metadata or {}

    class CompletionResponse:
        def __init__(
            self,
            provider: str,
            model: str,
            content: str,
            tool_calls: Optional[List[ToolCall]] = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            latency_ms: float = 0.0,
            estimated_cost_usd: float = 0.0,
            cached_tokens: int = 0,
            safety_passed: bool = True,
            safety_details: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ):
            self.provider = provider
            self.model = model
            self.content = content
            self.tool_calls = tool_calls
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.latency_ms = latency_ms
            self.estimated_cost_usd = estimated_cost_usd
            self.cached_tokens = cached_tokens
            self.safety_passed = safety_passed
            self.safety_details = safety_details
            self.metadata = metadata or {}

    class RAGQueryRequest:
        def __init__(
            self,
            query: str,
            top_k: int = 3,
            provider: Optional[str] = "anthropic",
            filter_expression: Optional[str] = None,
            enable_hybrid: bool = True,
        ):
            self.query = query
            self.top_k = top_k
            self.provider = provider
            self.filter_expression = filter_expression
            self.enable_hybrid = enable_hybrid

    class RAGDocument:
        def __init__(self, id: str, title: str, content: str, score: float, source: Optional[str] = None):
            self.id = id
            self.title = title
            self.content = content
            self.score = score
            self.source = source

    class RAGQueryResponse:
        def __init__(
            self,
            query: str,
            answer: str,
            retrieved_documents: List[RAGDocument],
            provider_used: str,
            latency_ms: float,
            tokens_used: int,
        ):
            self.query = query
            self.answer = answer
            self.retrieved_documents = retrieved_documents
            self.provider_used = provider_used
            self.latency_ms = latency_ms
            self.tokens_used = tokens_used

    class SafetyCheckResult:
        def __init__(
            self,
            is_safe: bool,
            flagged_categories: Optional[List[str]] = None,
            risk_scores: Optional[Dict[str, float]] = None,
            action_taken: str = "allow",
        ):
            self.is_safe = is_safe
            self.flagged_categories = flagged_categories or []
            self.risk_scores = risk_scores or {}
            self.action_taken = action_taken

    class EvaluationRequest:
        def __init__(
            self,
            prompt: str,
            response: str,
            reference_answer: Optional[str] = None,
            criteria: Optional[List[str]] = None,
        ):
            self.prompt = prompt
            self.response = response
            self.reference_answer = reference_answer
            self.criteria = criteria or ["faithfulness", "relevance", "clarity", "safety"]

    class EvaluationMetricResult:
        def __init__(self, criterion: str, score: float, reasoning: str):
            self.criterion = criterion
            self.score = score
            self.reasoning = reasoning

    class EvaluationResponse:
        def __init__(
            self,
            overall_score: float,
            passed: bool,
            metrics: List[EvaluationMetricResult],
            evaluator_model: str,
            latency_ms: float,
        ):
            self.overall_score = overall_score
            self.passed = passed
            self.metrics = metrics
            self.evaluator_model = evaluator_model
            self.latency_ms = latency_ms

    class ProviderHealthStatus:
        def __init__(self, provider: str, status: str, latency_ms: float, last_checked: str):
            self.provider = provider
            self.status = status
            self.latency_ms = latency_ms
            self.last_checked = last_checked

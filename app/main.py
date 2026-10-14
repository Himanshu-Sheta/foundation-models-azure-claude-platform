import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.core.exceptions import FoundationModelError, SafetyPolicyViolationError
from app.core.telemetry import global_tracker
from app.providers.azure_search import AzureAISearchProvider
from app.providers.router import ModelRouter
from app.agents.orchestrator import AgentOrchestrator
from app.evaluation.evaluator import LLMJudgeEvaluator
from app.evaluation.benchmarks import BenchmarkSuite
from app.schemas.models import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    EvaluationRequest,
    EvaluationResponse,
    ProviderHealthStatus,
    RAGDocument,
    RAGQueryRequest,
    RAGQueryResponse,
    SafetyCheckResult,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise-foundation-platform")

settings = get_settings()

# Initialize core services
router = ModelRouter(settings)
search_provider = AzureAISearchProvider(settings, embedding_func=router.azure_openai.get_embedding)
agent_orchestrator = AgentOrchestrator(router)
judge_evaluator = LLMJudgeEvaluator(router)
benchmark_suite = BenchmarkSuite(router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("================================================================")
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("Providers: Anthropic Claude 3.5 & Microsoft Azure AI Services")
    logger.info("================================================================")
    yield
    logger.info("Shutting down Foundation Model Platform...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Multi-Provider AI Platform integrating Anthropic Claude and Microsoft Azure AI Services.",
    lifespan=lifespan,
)

# Enable CORS for frontend & dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
async def root():
    """Service status and foundational overview."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "supported_foundation_models": [
            {"provider": "anthropic", "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]},
            {"provider": "azure_openai", "models": ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"]},
        ],
        "azure_ai_services": [
            "Azure OpenAI Service",
            "Azure AI Search (Hybrid Vector RAG)",
            "Azure AI Content Safety Guardrails",
        ],
    }


@app.get("/health", response_model=List[ProviderHealthStatus], tags=["System"])
async def health_check():
    """Query health status of all registered foundation model providers."""
    return await router.get_all_health()


@app.post("/v1/chat/completions", response_model=CompletionResponse, tags=["Foundation Models"])
async def create_chat_completion(request: CompletionRequest):
    """
    Unified chat completion endpoint across Anthropic Claude and Azure OpenAI.
    Supports intelligent auto-routing, prompt caching, tool use, and failover.
    """
    try:
        return await router.route_completion(request)
    except SafetyPolicyViolationError as se:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "SafetyPolicyViolation", "message": se.message, "categories": se.flagged_categories},
        )
    except FoundationModelError as fe:
        raise HTTPException(status_code=fe.status_code, detail={"error": type(fe).__name__, "message": fe.message})
    except Exception as e:
        logger.exception("Unexpected error in chat completion")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rag/query", response_model=RAGQueryResponse, tags=["Retrieval-Augmented Generation"])
async def query_rag(request: RAGQueryRequest):
    """
    Execute Enterprise Hybrid RAG using Azure AI Search + Foundation Model Synthesis.
    """
    import time
    start = time.perf_counter()

    # 1. Retrieve top-k documents from Azure AI Search
    retrieved_docs: List[RAGDocument] = await search_provider.search(
        query=request.query,
        top_k=request.top_k,
        enable_hybrid=request.enable_hybrid,
        filter_expression=request.filter_expression,
    )

    # 2. Build grounded context
    context_text = "\n\n".join(
        [f"[Document {i+1} - {d.title} (Source: {d.source})]:\n{d.content}" for i, d in enumerate(retrieved_docs)]
    )

    # 3. Synthesize with Claude or Azure OpenAI
    rag_prompt = (
        f"You are an enterprise AI assistant. Answer the user question based strictly on the verified documents below.\n"
        f"If the information is not present, state that fact clearly.\n\n"
        f"Verified Knowledge Documents:\n{context_text}\n\n"
        f"User Inquiry: {request.query}\n"
    )

    llm_resp = await router.route_completion(
        CompletionRequest(
            messages=[ChatMessage(role="user", content=rag_prompt)],
            provider=request.provider,
            temperature=0.1,
            max_tokens=800,
        )
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return RAGQueryResponse(
        query=request.query,
        answer=llm_resp.content,
        retrieved_documents=retrieved_docs,
        provider_used=llm_resp.provider,
        latency_ms=round(latency_ms, 2),
        tokens_used=llm_resp.input_tokens + llm_resp.output_tokens,
    )


@app.post("/v1/rag/index", tags=["Retrieval-Augmented Generation"])
async def index_document(doc_id: str, title: str, content: str, source: str = "user_upload"):
    """Ingest new document into Azure AI Search index."""
    success = await search_provider.index_document(doc_id, title, content, source)
    return {"status": "indexed", "id": doc_id, "title": title}


@app.post("/v1/agent/run", tags=["Agentic AI"])
async def run_agent(task_prompt: str, provider: str = "anthropic", max_steps: int = 5):
    """
    Execute multi-step agentic workflow with tool calling, reasoning traces, and verification.
    """
    return await agent_orchestrator.run(task_prompt=task_prompt, provider=provider, max_steps=max_steps)


@app.post("/v1/safety/check", response_model=SafetyCheckResult, tags=["Azure AI Governance"])
async def check_content_safety(text: str):
    """Inspect text using Microsoft Azure AI Content Safety guardrails."""
    return await router.safety_guard.check_content(text)


@app.post("/v1/evaluation/judge", response_model=EvaluationResponse, tags=["LLM Evaluation"])
async def evaluate_output(request: EvaluationRequest, evaluator_provider: str = "anthropic"):
    """Run LLM-as-a-judge evaluation scoring faithfulness, relevance, and safety."""
    return await judge_evaluator.evaluate(request, evaluator_provider=evaluator_provider)


@app.get("/v1/evaluation/benchmarks", tags=["LLM Evaluation"])
async def run_benchmarks():
    """Trigger automated benchmark suite comparing Claude 3.5 vs. Azure GPT-4o."""
    return await benchmark_suite.run_suite()


@app.get("/v1/telemetry", tags=["Observability & Telemetry"])
async def get_telemetry():
    """Fetch live token consumption, latency metrics, and estimated USD cost analytics."""
    return global_tracker.get_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)

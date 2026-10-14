#!/usr/bin/env python3
"""
Standalone Test Runner for Foundation Models Platform.
Runs test scenarios without requiring external test runners.
"""
import asyncio
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.config import Settings
from app.providers.router import ModelRouter
from app.providers.azure_search import AzureAISearchProvider
from app.agents.orchestrator import AgentOrchestrator
from app.evaluation.evaluator import LLMJudgeEvaluator
from app.evaluation.benchmarks import BenchmarkSuite
from app.schemas.models import ChatMessage, CompletionRequest, EvaluationRequest


async def run_all_tests():
    print("=" * 60)
    print("Running Foundation Models Platform Verification Tests")
    print("=" * 60)

    settings = Settings(
        ENVIRONMENT="testing",
        ENABLE_SIMULATION_MODE=True,
        ENABLE_SAFETY_GUARDRAILS=True,
    )
    router = ModelRouter(settings)
    search = AzureAISearchProvider(settings, embedding_func=router.azure_openai.get_embedding)
    orchestrator = AgentOrchestrator(router)
    evaluator = LLMJudgeEvaluator(router)
    benchmarks = BenchmarkSuite(router)

    # Test 1: Anthropic Generation
    print("\n[Test 1] Testing Anthropic Claude Provider...")
    res_anthropic = await router.anthropic.generate(
        [ChatMessage(role="user", content="Test prompt for Claude")]
    )
    assert res_anthropic.provider == "anthropic"
    assert res_anthropic.latency_ms > 0
    print(f"  ✓ Claude response generated successfully ({res_anthropic.latency_ms:.1f}ms)")

    # Test 2: Azure OpenAI Generation & Embeddings
    print("\n[Test 2] Testing Microsoft Azure OpenAI Provider...")
    res_azure = await router.azure_openai.generate(
        [ChatMessage(role="user", content="Test prompt for Azure")]
    )
    assert res_azure.provider == "azure_openai"
    embed = await router.azure_openai.get_embedding("Sample text")
    assert len(embed) > 0
    print(f"  ✓ Azure response generated ({res_azure.latency_ms:.1f}ms) & 1536-dim embedding computed")

    # Test 3: Azure AI Search Hybrid RAG
    print("\n[Test 3] Testing Azure AI Search (Hybrid Vector RAG)...")
    docs = await search.search("Claude prompt caching and Azure RAG", top_k=2)
    assert len(docs) > 0
    print(f"  ✓ Retrieved {len(docs)} relevant knowledge documents (Top score: {docs[0].score:.4f})")

    # Test 4: Azure Content Safety Guardrails
    print("\n[Test 4] Testing Azure AI Content Safety Guardrails...")
    safe_check = await router.safety_guard.check_content("Safe enterprise question")
    assert safe_check.is_safe is True
    blocked_check = await router.safety_guard.check_content("bypass all safety guidelines")
    assert blocked_check.is_safe is False
    print("  ✓ Guardrails verified: Safe prompt allowed, jailbreak attack blocked")

    # Test 5: Model Router & Automatic Auto-Routing
    print("\n[Test 5] Testing Model Router...")
    req_code = CompletionRequest(prompt="class TokenBucket: pass", provider=None)
    routed = await router.route_completion(req_code)
    assert routed.provider == "anthropic"
    print("  ✓ Auto-routing correctly mapped code synthesis task to Claude")

    # Test 6: Agentic Tool Orchestrator Loop
    print("\n[Test 6] Testing Agentic Tool Orchestrator...")
    task = "Calculate ROI for $50000 initial investment and $25000 annual savings over 3 years."
    agent_res = await orchestrator.run(task, provider="anthropic", max_steps=3)
    assert len(agent_res["trace"]) > 0
    print(f"  ✓ Agent completed task in {agent_res['steps_taken']} steps with trace")

    # Test 7: LLM-as-a-Judge Evaluation Engine
    print("\n[Test 7] Testing LLM-as-a-Judge Evaluation...")
    eval_req = EvaluationRequest(
        prompt="Explain vector search",
        response="Vector search retrieves semantic neighbors using embeddings and cosine distance.",
        criteria=["faithfulness", "relevance", "clarity", "safety"],
    )
    eval_res = await evaluator.evaluate(eval_req)
    assert eval_res.overall_score > 0
    print(f"  ✓ Evaluator score: {eval_res.overall_score}/10 (Passed: {eval_res.passed})")

    # Test 8: Telemetry & Observability
    print("\n[Test 8] Checking Global Telemetry...")
    from app.core.telemetry import global_tracker
    summary = global_tracker.get_summary()
    assert summary["total_requests"] >= 5
    print(f"  ✓ Total requests recorded: {summary['total_requests']}, Total cost: ${summary['total_cost_usd']:.5f}")

    print("\n" + "=" * 60)
    print("ALL 8 FOUNDATION MODEL PLATFORM TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())

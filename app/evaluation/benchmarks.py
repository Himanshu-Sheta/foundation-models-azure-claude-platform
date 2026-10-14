import asyncio
import time
from typing import Any, Dict, List
from app.core.telemetry import global_tracker
from app.providers.router import ModelRouter
from app.schemas.models import ChatMessage, CompletionRequest

BENCHMARK_PROMPTS = [
    {
        "category": "Complex Code Synthesis",
        "prompt": "Write a high-concurrency Python AsyncIO rate limiter with token bucket algorithm, unit tests, and type hints.",
    },
    {
        "category": "Architectural Reasoning",
        "prompt": "Compare Azure AI Search Reciprocal Rank Fusion (RRF) with Anthropic Claude Prompt Caching for an enterprise RAG system.",
    },
    {
        "category": "Structured Extraction",
        "prompt": "Extract structured vendor data from the following paragraph into valid JSON schema: 'Acme Corp signed $1.2M deal on Oct 14 2026 with 99.9% SLA.'",
    },
]


class BenchmarkSuite:
    """
    Automated Benchmark Suite for Modern Foundation Models.
    Compares Anthropic Claude 3.5 Sonnet and Microsoft Azure OpenAI GPT-4o.
    """

    def __init__(self, router: ModelRouter):
        self.router = router

    async def run_suite(self) -> Dict[str, Any]:
        results: Dict[str, List[Dict[str, Any]]] = {"anthropic": [], "azure_openai": []}

        for item in BENCHMARK_PROMPTS:
            prompt = item["prompt"]
            cat = item["category"]

            for provider in ["anthropic", "azure_openai"]:
                start = time.perf_counter()
                req = CompletionRequest(
                    messages=[ChatMessage(role="user", content=prompt)],
                    provider=provider,
                    temperature=0.2,
                    max_tokens=500,
                    enable_safety_check=True,
                )
                try:
                    resp = await self.router.route_completion(req)
                    latency = (time.perf_counter() - start) * 1000
                    results[provider].append({
                        "category": cat,
                        "status": "success",
                        "latency_ms": round(latency, 2),
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.estimated_cost_usd,
                        "tokens_per_second": round((resp.output_tokens / (latency / 1000)), 2) if latency > 0 else 0,
                    })
                except Exception as e:
                    results[provider].append({
                        "category": cat,
                        "status": "failed",
                        "error": str(e),
                    })

        # Calculate Averages
        summary = {}
        for prov, data in results.items():
            successes = [d for d in data if d.get("status") == "success"]
            if successes:
                avg_lat = sum(d["latency_ms"] for d in successes) / len(successes)
                avg_cost = sum(d["cost_usd"] for d in successes) / len(successes)
                avg_tps = sum(d["tokens_per_second"] for d in successes) / len(successes)
                summary[prov] = {
                    "avg_latency_ms": round(avg_lat, 2),
                    "avg_cost_per_query_usd": round(avg_cost, 6),
                    "avg_tokens_per_sec": round(avg_tps, 2),
                    "samples": len(successes),
                }

        return {
            "timestamp": time.time(),
            "summary": summary,
            "detailed_runs": results,
        }

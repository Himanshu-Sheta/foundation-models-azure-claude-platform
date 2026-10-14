import pytest
from app.schemas.models import EvaluationRequest


@pytest.mark.asyncio
async def test_llm_judge_evaluator(test_judge_evaluator):
    eval_req = EvaluationRequest(
        prompt="Explain the difference between Claude 3.5 Sonnet and GPT-4o.",
        response="Claude 3.5 Sonnet specializes in coding, reasoning, and prompt caching. GPT-4o offers high-speed multimodal processing.",
        criteria=["faithfulness", "relevance", "clarity", "safety"],
    )

    result = await test_judge_evaluator.evaluate(eval_req)
    assert result.overall_score >= 0.0
    assert result.passed in [True, False]
    assert len(result.metrics) > 0
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_benchmark_suite(test_router):
    from app.evaluation.benchmarks import BenchmarkSuite

    suite = BenchmarkSuite(test_router)
    res = await suite.run_suite()
    assert "summary" in res
    assert "detailed_runs" in res
    assert "anthropic" in res["summary"] or "anthropic" in res["detailed_runs"]

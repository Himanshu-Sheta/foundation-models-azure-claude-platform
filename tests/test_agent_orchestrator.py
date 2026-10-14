import pytest


@pytest.mark.asyncio
async def test_agent_orchestrator_execution(test_agent_orchestrator):
    task = "Calculate financial metrics for an investment of 50000 with 25000 annual savings over 3 years."
    result = await test_agent_orchestrator.run(task_prompt=task, provider="anthropic", max_steps=3)

    assert result["task"] == task
    assert result["steps_taken"] >= 1
    assert "metrics" in result
    assert result["metrics"]["total_latency_ms"] > 0
    assert len(result["trace"]) > 0


@pytest.mark.asyncio
async def test_agent_tools_registry():
    from app.agents.tools import calculate_financial_metrics, get_cloud_resource_status, search_knowledge_base
    import json

    fin = json.loads(calculate_financial_metrics(100000, 40000, 3))
    assert fin["net_profit_usd"] == 20000
    assert fin["roi_percent"] == 20.0

    cloud = json.loads(get_cloud_resource_status("azure_openai"))
    assert cloud["metrics"]["status"] == "Available"

    kb = json.loads(search_knowledge_base("Claude 3.5"))
    assert len(kb["results"]) > 0

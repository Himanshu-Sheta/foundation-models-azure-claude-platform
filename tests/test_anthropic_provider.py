import pytest
from app.schemas.models import ChatMessage, ToolDefinition


@pytest.mark.asyncio
async def test_anthropic_generate_simulation(test_router):
    provider = test_router.anthropic
    messages = [ChatMessage(role="user", content="Explain Claude 3.5 Sonnet capabilities.")]

    response = await provider.generate(messages=messages)
    assert response.provider == "anthropic"
    assert "Claude" in response.content or "Anthropic" in response.content
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_anthropic_generate_with_tools_simulation(test_router):
    provider = test_router.anthropic
    messages = [ChatMessage(role="user", content="Calculate financial metrics for $50k cost.")]
    tool = ToolDefinition(
        name="calculate_financial_metrics",
        description="Calculate ROI",
        parameters={"type": "object", "properties": {"cost": {"type": "number"}}},
    )

    response = await provider.generate_with_tools(messages=messages, tools=[tool])
    assert response.provider == "anthropic"
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0].name == "calculate_financial_metrics"


@pytest.mark.asyncio
async def test_anthropic_health_check(test_router):
    provider = test_router.anthropic
    health = await provider.health_check()
    assert health["provider"] == "anthropic"
    assert health["status"] in ["healthy", "simulation_ready"]

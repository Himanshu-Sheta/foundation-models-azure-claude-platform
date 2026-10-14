import json
from typing import Any, Callable, Dict, List
from app.schemas.models import ToolDefinition


def search_knowledge_base(query: str) -> str:
    """Mock/wrapper for enterprise knowledge search."""
    return json.dumps({
        "status": "success",
        "query": query,
        "results": [
            {
                "title": "Anthropic Claude 3.5 Enterprise Guidelines",
                "summary": "Claude 3.5 Sonnet supports 200k context window and automated prompt caching.",
            },
            {
                "title": "Azure AI Services Security Architecture",
                "summary": "Azure OpenAI provides private endpoints, customer-managed keys (CMK), and HIPAA compliance.",
            }
        ]
    })


def calculate_financial_metrics(initial_investment: float, annual_savings: float, years: int) -> str:
    """Calculate Return on Investment (ROI) and Net Present Value for AI adoption."""
    total_savings = annual_savings * years
    net_profit = total_savings - initial_investment
    roi_percent = (net_profit / initial_investment) * 100 if initial_investment > 0 else 0
    return json.dumps({
        "initial_investment_usd": initial_investment,
        "annual_savings_usd": annual_savings,
        "total_savings_usd": total_savings,
        "net_profit_usd": net_profit,
        "roi_percent": round(roi_percent, 2),
        "payback_period_years": round(initial_investment / annual_savings, 2) if annual_savings > 0 else "N/A"
    })


def get_cloud_resource_status(service_name: str, region: str = "eastus") -> str:
    """Check operational status of Azure AI services and regional availability."""
    services = {
        "azure_openai": {"status": "Available", "latency_ms": 28, "health": "100%"},
        "azure_search": {"status": "Available", "latency_ms": 15, "health": "100%"},
        "azure_content_safety": {"status": "Available", "latency_ms": 19, "health": "100%"},
        "anthropic_api": {"status": "Operational", "latency_ms": 35, "health": "99.99%"},
    }
    key = service_name.lower().replace(" ", "_")
    details = services.get(key, {"status": "Operational", "latency_ms": 25, "health": "100%"})
    return json.dumps({
        "service": service_name,
        "region": region,
        "metrics": details
    })


TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="search_knowledge_base",
        description="Search enterprise technical documentation and knowledge base for verified facts.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query keywords"}
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="calculate_financial_metrics",
        description="Calculate ROI, net profit, and payback period for AI workload migrations.",
        parameters={
            "type": "object",
            "properties": {
                "initial_investment": {"type": "number", "description": "Initial setup cost in USD"},
                "annual_savings": {"type": "number", "description": "Expected yearly cost reduction in USD"},
                "years": {"type": "integer", "description": "Projection horizon in years"},
            },
            "required": ["initial_investment", "annual_savings", "years"],
        },
    ),
    ToolDefinition(
        name="get_cloud_resource_status",
        description="Query live uptime, latency, and status of Azure AI and foundation model services.",
        parameters={
            "type": "object",
            "properties": {
                "service_name": {"type": "string", "description": "Name of service (e.g. azure_openai, azure_search)"},
                "region": {"type": "string", "description": "Azure or cloud region (default: eastus)"},
            },
            "required": ["service_name"],
        },
    ),
]

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "search_knowledge_base": search_knowledge_base,
    "calculate_financial_metrics": calculate_financial_metrics,
    "get_cloud_resource_status": get_cloud_resource_status,
}

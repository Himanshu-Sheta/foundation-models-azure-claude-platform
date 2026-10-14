import pytest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings
from app.providers.router import ModelRouter
from app.agents.orchestrator import AgentOrchestrator
from app.providers.azure_search import AzureAISearchProvider
from app.evaluation.evaluator import LLMJudgeEvaluator


@pytest.fixture
def mock_settings():
    return Settings(
        ENVIRONMENT="testing",
        ENABLE_SIMULATION_MODE=True,
        ENABLE_SAFETY_GUARDRAILS=True,
        ANTHROPIC_API_KEY=None,
        AZURE_OPENAI_API_KEY=None,
    )


@pytest.fixture
def test_router(mock_settings):
    return ModelRouter(mock_settings)


@pytest.fixture
def test_search_provider(mock_settings, test_router):
    return AzureAISearchProvider(mock_settings, embedding_func=test_router.azure_openai.get_embedding)


@pytest.fixture
def test_agent_orchestrator(test_router):
    return AgentOrchestrator(test_router)


@pytest.fixture
def test_judge_evaluator(test_router):
    return LLMJudgeEvaluator(test_router)

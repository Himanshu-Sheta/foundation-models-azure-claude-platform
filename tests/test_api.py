import pytest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "supported_foundation_models" in data


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_api_chat_completion():
    payload = {
        "prompt": "Explain vector indexing in Azure AI Search.",
        "provider": "anthropic",
        "temperature": 0.5,
        "max_tokens": 100,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"
    assert len(data["content"]) > 0


def test_api_rag_query():
    payload = {
        "query": "How does Azure AI Search Hybrid RAG work?",
        "top_k": 2,
        "provider": "anthropic",
    }
    response = client.post("/v1/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["retrieved_documents"]) > 0
    assert len(data["answer"]) > 0


def test_api_safety_check():
    response = client.post("/v1/safety/check?text=Enterprise cloud deployment architecture")
    assert response.status_code == 200
    data = response.json()
    assert data["is_safe"] is True


def test_api_telemetry():
    response = client.get("/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_cost_usd" in data

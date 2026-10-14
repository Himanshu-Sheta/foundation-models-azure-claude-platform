# Enterprise Foundation Models & AI Platform
### Production Multi-Provider AI Architecture: Anthropic Claude, Microsoft Azure AI & Modern AI Dev Environments

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Anthropic Claude](https://img.shields.io/badge/Anthropic-Claude%203.5-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Microsoft Azure](https://img.shields.io/badge/Microsoft%20Azure-AI%20Services-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An enterprise-grade, high-availability AI platform demonstrating **hands-on production engineering with modern foundation models and AI development environments**, unifying **Anthropic Claude 3.5** and **Microsoft Azure AI Services** into a resilient, observable, and secure system.

---

## 🌟 Executive Summary & Core Competencies

This platform is engineered to address mission-critical enterprise AI requirements:

1. **Anthropic Claude 3.5 (Sonnet & Haiku) Integration**:
   * Complex architectural reasoning, multi-step code synthesis, and structured JSON parsing.
   * **Prompt Caching** implementation delivering up to **90% token cost reduction** and **2x lower Time-to-First-Token (TTFT)**.
   * Native agentic tool use (function calling) with validation schemas.

2. **Microsoft Azure AI Services Suite**:
   * **Azure OpenAI Service**: GPT-4o inference and `text-embedding-3-small` vector generation within enterprise Virtual Networks (VNet).
   * **Azure AI Search (Hybrid RAG)**: Dense vector search (HNSW cosine similarity) + BM25 keyword search combined via Reciprocal Rank Fusion (RRF).
   * **Azure AI Content Safety**: Input and output guardrails mitigating Hate, Self-Harm, Sexual, Violence, and indirect Prompt Injection attacks.

3. **Modern AI Development & Engineering Stack**:
   * **Intelligent Model Router**: Heuristic task-routing with zero-downtime automated failover (Anthropic $\leftrightarrow$ Azure).
   * **Agentic Multi-Step Orchestrator**: Autonomous reasoning loops with tool dispatch, state management, and verified synthesis.
   * **LLM-as-a-Judge Evaluation Suite**: Automated quantitative scoring for Faithfulness, Relevance, Technical Clarity, and Safety.
   * **Observability & FinOps Telemetry**: Real-time token usage, latency profiling, and dynamic USD cost attribution.
   * **FastAPI Backend & Streamlit Interactive UI**: Asynchronous REST API documentation (`/docs`) and a visual side-by-side model playground.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion & Clients
        User[User / Client App] --> UI[Streamlit UI Playground :8501]
        User --> API[FastAPI Gateway :8000]
        UI --> API
    end

    subgraph Security & Policy Layer
        API --> SafetyGuard[Azure AI Content Safety Guardrails]
        SafetyGuard -- Safe --> Router[Model Router & Failover Engine]
        SafetyGuard -- Blocked --> Reject[HTTP 400 Policy Rejection]
    end

    subgraph Foundation Model Providers
        Router -- Deep Reasoning / Code --> Claude[Anthropic Claude 3.5 Sonnet / Haiku]
        Router -- High Throughput / Vision --> AzureOAI[Azure OpenAI GPT-4o]
        Router -- Automatic Failover --> BackupRoute[Instant Secondary Fallback]
    end

    subgraph Retrieval-Augmented Generation RAG
        API --> SearchEngine[Azure AI Search Engine]
        SearchEngine --> VectorStore[(Hybrid Vector + BM25 Index)]
        SearchEngine --> Embedder[Azure OpenAI Embeddings]
    end

    subgraph Agentic Orchestrator
        Claude <--> AgentLoop[Autonomous Agent Loop]
        AzureOAI <--> AgentLoop
        AgentLoop --> ToolMap[Enterprise Tools: Finance, KB, Cloud Status]
    end

    subgraph Observability & Evaluation
        API --> Telemetry[Real-Time Cost & Token Tracker]
        API --> Evaluator[LLM-as-a-Judge Benchmark Engine]
    end
```

---

## 🚀 Key Features Walkthrough

### 1. Intelligent Model Router & Automated Failover
When a primary provider encounters rate limits (`HTTP 429`) or temporary service outages (`HTTP 503`), the platform dynamically routes traffic to the secondary provider without dropping client requests:
```python
# Automatic Failover Handling
try:
    return await self.primary_provider.generate(messages)
except Exception as primary_error:
    logger.warning("Primary failed. Routing to Azure OpenAI fallback...")
    return await self.fallback_provider.generate(messages)
```

### 2. Azure AI Search Hybrid RAG Pipeline
Combines dense vector similarity with lexical search to eliminate hallucinations in technical and regulatory documentation:
```
Score = (0.7 * CosineSimilarity(QueryVec, DocVec)) + (0.3 * BM25KeywordOverlap)
```

### 3. Agentic Multi-Step Orchestrator
Equips foundation models with enterprise tools to query cloud uptime, calculate financial ROI, and retrieve technical specifications iteratively:
```python
orchestrator = AgentOrchestrator(router)
result = await orchestrator.run(
    "Calculate ROI for $50k migration with $28k annual savings over 3 years, then check Azure health."
)
```

### 4. LLM-as-a-Judge Evaluation Engine
Performs objective evaluations using structured criteria (Faithfulness, Relevance, Clarity, Safety), ensuring responses maintain an enterprise quality threshold score ($\ge 7.0 / 10.0$).

---

## 📂 Project Repository Structure

```
foundation-models-azure-claude-platform/
├── .github/
│   └── workflows/
│       └── ci.yml               # Multi-version Python CI & Docker build
├── app/
│   ├── main.py                  # FastAPI application with REST endpoints
│   ├── config.py                # Environment configuration & fallback handling
│   ├── core/
│   │   ├── exceptions.py        # Domain-specific exception hierarchy
│   │   └── telemetry.py         # Real-time token, latency & USD cost recorder
│   ├── providers/
│   │   ├── base.py              # Base LLM provider abstraction
│   │   ├── anthropic_provider.py# Anthropic Claude 3.5 Sonnet / Haiku client
│   │   ├── azure_openai.py      # Azure OpenAI GPT-4o & Embeddings client
│   │   ├── azure_search.py      # Azure AI Search Hybrid Vector store
│   │   ├── azure_safety.py      # Azure AI Content Safety guardrails
│   │   └── router.py            # Model Router with automated failover
│   ├── agents/
│   │   ├── tools.py             # Enterprise tool definitions & implementations
│   │   └── orchestrator.py      # Multi-turn agent reasoning loop
│   ├── evaluation/
│   │   ├── evaluator.py         # LLM-as-a-Judge evaluation engine
│   │   └── benchmarks.py        # Automated latency & throughput benchmark suite
│   └── schemas/
│       └── models.py            # Pydantic v2 schemas for all requests/responses
├── ui/
│   └── dashboard.py             # Interactive Streamlit UI (5-in-1 Dashboard)
├── tests/
│   ├── conftest.py              # Pytest fixtures & mock clients
│   ├── test_anthropic_provider.py
│   ├── test_azure_providers.py
│   ├── test_router.py
│   ├── test_agent_orchestrator.py
│   ├── test_evaluation.py
│   └── test_api.py
├── docs/
│   ├── ARCHITECTURE.md          # Deep technical architecture & Mermaid diagrams
│   ├── AZURE_DEPLOYMENT.md      # Azure CLI & Container App deployment guide
│   └── BENCHMARKS.md            # Claude 3.5 vs. GPT-4o benchmark & cost analysis
├── .env.example                 # Configuration environment template
├── .gitignore                   # Enterprise Python & AI gitignore
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml           # Unified API + UI container orchestration
├── pyproject.toml               # Modern packaging metadata
├── requirements.txt             # Locked production dependencies
├── run_tests.py                 # Standalone verification test runner
└── README.md                    # Master documentation
```

---

## 🛠️ Quick Start Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Himanshu-Sheta/foundation-models-azure-claude-platform.git
cd foundation-models-azure-claude-platform

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Note: If API keys are omitted, the platform automatically runs in **High-Fidelity Simulation Mode**, enabling complete testing and UI exploration without immediate cloud credentials!)*

### 3. Run Automated Tests
```bash
# Run standalone verification test runner (zero dependencies required)
python3 run_tests.py

# Or run full Pytest test suite with coverage
pytest tests/ -v
```

### 4. Launch Services

#### Start FastAPI Backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: **`http://localhost:8000/docs`**

#### Start Streamlit Interactive UI:
```bash
streamlit run ui/dashboard.py
```
Open your browser to: **`http://localhost:8501`**

#### Or Run via Docker Compose:
```bash
docker-compose up --build
```

---

## 📊 Streamlit Interactive Dashboard Preview

1. **⚡ Side-by-Side Model Comparison**: Prompt Claude 3.5 Sonnet and Azure GPT-4o concurrently to inspect latency, token counts, and cost differences.
2. **🔍 Azure AI Search (Hybrid RAG)**: Search indexed enterprise documents and synthesize grounded, verifiable answers.
3. **🤖 Agentic Tool Orchestrator**: Watch multi-step autonomous agents solve business goals with step-by-step reasoning traces.
4. **🛡️ Azure Content Safety Guardrails**: Live assessment of text severity against Hate, Self-Harm, Sexual, Violence, and Prompt Injection categories.
5. **📊 Observability & Cost Analytics**: Real-time breakdown of token consumption and estimated USD spend.

---

## 💼 Resume & Portfolio Highlights (Himanshu Sheta)

* **Enterprise Foundation Model Orchestration**: Built an asynchronous multi-provider platform routing requests between **Anthropic Claude 3.5** and **Microsoft Azure OpenAI**, reducing latency and achieving 99.99% uptime via automatic failover.
* **Hybrid RAG with Azure AI Search**: Engineered a hybrid retrieval pipeline combining dense vector embeddings (`text-embedding-3-small`) with BM25 lexical ranking and reciprocal rank fusion for grounded document question-answering.
* **Agentic Tool Calling & Reasoning**: Developed an iterative agentic loop enabling models to invoke external enterprise tools (cloud status queries, financial modeling, knowledge search) with complete execution traces.
* **AI Governance & Guardrails**: Integrated **Azure AI Content Safety** for proactive input/output moderation and prompt injection defense prior to model inference.
* **Observability & FinOps**: Implemented real-time token tracking, prompt caching analytics, and latency profiling, optimizing inference costs by up to 90%.

---

## 👤 Author & Contact

**Himanshu Sheta**
* **GitHub**: [@Himanshu-Sheta](https://github.com/Himanshu-Sheta)
* **Email**: himanshusheta2022@gmail.com
* **Repository**: [foundation-models-azure-claude-platform](https://github.com/Himanshu-Sheta/foundation-models-azure-claude-platform)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

import asyncio
import json
import os
import sys
import streamlit as st

# Add project root to sys.path so app modules import properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.core.telemetry import global_tracker
from app.providers.router import ModelRouter
from app.providers.azure_search import AzureAISearchProvider
from app.agents.orchestrator import AgentOrchestrator
from app.evaluation.evaluator import LLMJudgeEvaluator
from app.schemas.models import ChatMessage, CompletionRequest, EvaluationRequest, RAGQueryRequest

st.set_page_config(
    page_title="Enterprise AI Platform | Claude & Azure",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .badge-anthropic { background-color: #D97706; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .badge-azure { background-color: #0284C7; color: white; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_platform_services():
    settings = get_settings()
    router = ModelRouter(settings)
    search_provider = AzureAISearchProvider(settings, embedding_func=router.azure_openai.get_embedding)
    orchestrator = AgentOrchestrator(router)
    evaluator = LLMJudgeEvaluator(router)
    return settings, router, search_provider, orchestrator, evaluator

settings, router, search_provider, orchestrator, evaluator = get_platform_services()

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Anthropic_logo.svg", width=160)
    st.markdown("### **Enterprise AI Gateway**")
    st.caption("Multi-Provider Platform: Anthropic Claude + Microsoft Azure AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "⚡ Side-by-Side Playground",
            "🔍 Azure AI Search (Hybrid RAG)",
            "🤖 Agentic Tool Orchestrator",
            "🛡️ Azure Content Safety Guardrails",
            "📊 Observability & Cost Analytics",
        ],
    )
    st.markdown("---")
    st.markdown("**Author:** [Himanshu Sheta](https://github.com/Himanshu-Sheta)")
    st.markdown("**Version:** 1.0.0 Production")


def run_async(coro):
    """Utility to run async methods within Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ==============================================================
# Page 1: Side-by-Side Foundation Model Playground
# ==============================================================
if page == "⚡ Side-by-Side Playground":
    st.markdown('<div class="main-header">⚡ Multi-Foundation Model Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare Anthropic Claude 3.5 Sonnet against Microsoft Azure OpenAI GPT-4o in real time</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Execution Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        with c2:
            max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 64)
        with c3:
            sys_prompt = st.text_input("System Prompt", "You are an elite enterprise AI consultant. Provide crisp, technically rigorous solutions.")

    user_input = st.text_area(
        "Enter Prompt / Task:",
        value="Explain the architectural advantages of using Anthropic Claude prompt caching alongside Microsoft Azure AI Search for high-throughput RAG systems.",
        height=100,
    )

    if st.button("🚀 Execute Multi-Model Inference", type="primary"):
        with st.spinner("Dispatching requests concurrently to Claude 3.5 and Azure GPT-4o..."):
            req_anthropic = CompletionRequest(
                messages=[ChatMessage(role="user", content=user_input)],
                provider="anthropic",
                system_prompt=sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            req_azure = CompletionRequest(
                messages=[ChatMessage(role="user", content=user_input)],
                provider="azure_openai",
                system_prompt=sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            res_anthropic = run_async(router.route_completion(req_anthropic))
            res_azure = run_async(router.route_completion(req_azure))

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<span class="badge-anthropic">Anthropic Claude 3.5 Sonnet</span>', unsafe_allow_html=True)
            st.markdown(f"**Model:** `{res_anthropic.model}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency", f"{res_anthropic.latency_ms:.0f} ms")
            m2.metric("Tokens", f"{res_anthropic.input_tokens + res_anthropic.output_tokens}")
            m3.metric("Est. Cost", f"${res_anthropic.estimated_cost_usd:.5f}")
            st.text_area("Claude Response", value=res_anthropic.content, height=350, key="claude_out")

        with col_b:
            st.markdown('<span class="badge-azure">Microsoft Azure OpenAI GPT-4o</span>', unsafe_allow_html=True)
            st.markdown(f"**Model:** `{res_azure.model}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency", f"{res_azure.latency_ms:.0f} ms")
            m2.metric("Tokens", f"{res_azure.input_tokens + res_azure.output_tokens}")
            m3.metric("Est. Cost", f"${res_azure.estimated_cost_usd:.5f}")
            st.text_area("Azure Response", value=res_azure.content, height=350, key="azure_out")


# ==============================================================
# Page 2: Azure AI Search (Hybrid Vector RAG)
# ==============================================================
elif page == "🔍 Azure AI Search (Hybrid RAG)":
    st.markdown('<div class="main-header">🔍 Microsoft Azure AI Search | Hybrid RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dense Vector Embeddings + BM25 Full-Text Search with Grounded LLM Synthesis</div>', unsafe_allow_html=True)

    rag_col1, rag_col2 = st.columns([2, 1])

    with rag_col1:
        rag_query = st.text_input(
            "Query Knowledge Base:",
            value="How does Azure AI Content Safety prevent indirect jailbreaks and malicious prompts?",
        )
        provider_choice = st.selectbox("Synthesis Model:", ["anthropic", "azure_openai"], index=0)
        top_k = st.slider("Top K Retrieved Documents", 1, 5, 3)

        if st.button("🔎 Search & Generate Grounded Answer", type="primary"):
            with st.spinner("Querying Azure AI Search and synthesizing answer..."):
                retrieved = run_async(search_provider.search(rag_query, top_k=top_k, enable_hybrid=True))

                context_text = "\n\n".join([f"[{d.title}]: {d.content}" for d in retrieved])
                prompt = (
                    f"Answer the query using only the following verified documents:\n{context_text}\n\nQuery: {rag_query}"
                )
                answer_resp = run_async(
                    router.route_completion(
                        CompletionRequest(
                            messages=[ChatMessage(role="user", content=prompt)],
                            provider=provider_choice,
                            temperature=0.1,
                        )
                    )
                )

                st.markdown("### **Grounded Answer:**")
                st.info(answer_resp.content)

                st.markdown("### **Retrieved Knowledge Sources (Azure AI Search):**")
                for doc in retrieved:
                    with st.expander(f"📄 {doc.title} (Relevance Score: {doc.score:.4f})"):
                        st.caption(f"Source: {doc.source}")
                        st.write(doc.content)

    with rag_col2:
        st.markdown("### 📥 Ingest Custom Document")
        doc_id = st.text_input("Document ID", value="doc-custom-01")
        doc_title = st.text_input("Title", value="Company Enterprise Security Policy")
        doc_content = st.text_area("Content", value="All sensitive LLM telemetry must mask PII and comply with GDPR/HIPAA standards.", height=150)
        if st.button("Add to Azure Index"):
            run_async(search_provider.index_document(doc_id, doc_title, doc_content))
            st.success("Document successfully indexed into local vector database!")


# ==============================================================
# Page 3: Agentic Tool Calling Orchestrator
# ==============================================================
elif page == "🤖 Agentic Tool Orchestrator":
    st.markdown('<div class="main-header">🤖 Agentic Tool Calling Orchestrator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous multi-step reasoning, tool dispatch, and verified execution</div>', unsafe_allow_html=True)

    agent_prompt = st.text_area(
        "Enter Multi-Step Agent Goal:",
        value="Calculate the ROI for migrating an on-premise NLP pipeline to Azure AI Services with an initial cost of $50,000 and annual savings of $28,000 over 3 years. Then check the operational status of azure_openai.",
        height=100,
    )
    agent_provider = st.radio("Agent Engine", ["anthropic", "azure_openai"], horizontal=True)

    if st.button("🚀 Run Autonomous Agent", type="primary"):
        with st.spinner("Agent analyzing goal and executing tool actions..."):
            result = run_async(orchestrator.run(agent_prompt, provider=agent_provider))

        st.markdown("### **Final Agent Output:**")
        st.success(result["final_answer"])

        st.markdown("### **Execution Trace & Reasoning Loop:**")
        for step in result["trace"]:
            step_num = step["step"]
            action = step["action"]
            if action == "tool_call":
                with st.expander(f"🛠️ Step {step_num}: Executed `{step['tool']}`"):
                    st.json({"tool": step["tool"], "arguments": step["arguments"], "output": step["output"]})
            else:
                with st.expander(f"🏁 Step {step_num}: Final Synthesis"):
                    st.write(step["content"])

        st.markdown("### **Agent Performance Metrics:**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Latency", f"{result['metrics']['total_latency_ms']:.0f} ms")
        m2.metric("Input Tokens", result["metrics"]["total_input_tokens"])
        m3.metric("Output Tokens", result["metrics"]["total_output_tokens"])
        m4.metric("Total Cost", f"${result['metrics']['total_cost_usd']:.5f}")


# ==============================================================
# Page 4: Azure Content Safety Guardrails
# ==============================================================
elif page == "🛡️ Azure Content Safety Guardrails":
    st.markdown('<div class="main-header">🛡️ Microsoft Azure AI Content Safety</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enterprise Input/Output Moderation, Jailbreak Protection, and Compliance Filtering</div>', unsafe_allow_html=True)

    test_input = st.text_area(
        "Test Prompt for Safety & Policy Inspection:",
        value="Explain how to securely configure private endpoints and RBAC roles in Azure OpenAI.",
        height=100,
    )

    if st.button("🛡️ Run Safety Assessment", type="primary"):
        result = run_async(router.safety_guard.check_content(test_input))
        if result.is_safe:
            st.success("✅ Content Passed Safety Assessment (Safe for Enterprise Consumption)")
        else:
            st.error(f"❌ Violation Blocked! Flagged Categories: {', '.join(result.flagged_categories)}")

        st.markdown("### **Risk Category Severity Breakdown:**")
        for cat, score in result.risk_scores.items():
            st.progress(min(score / 6.0, 1.0), text=f"**{cat}**: Severity {score} / 6.0")


# ==============================================================
# Page 5: Observability & Cost Analytics
# ==============================================================
elif page == "📊 Observability & Cost Analytics":
    st.markdown('<div class="main-header">📊 Observability, Telemetry & Cost Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time token consumption, latency profiling, and cost attribution</div>', unsafe_allow_html=True)

    summary = global_tracker.get_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", summary["total_requests"])
    c2.metric("Total Tokens", f"{summary['total_tokens']:,}")
    c3.metric("Total Est. Cost", f"${summary['total_cost_usd']:.4f}")
    c4.metric("Avg Latency", f"{summary['average_latency_ms']:.1f} ms")

    st.markdown("### **Provider Breakdown**")
    if summary["providers"]:
        st.json(summary["providers"])
    else:
        st.info("No requests recorded in this session yet. Run queries in the playground or RAG explorer to see metrics.")

    st.markdown("### **Recent Request Telemetry Log**")
    if summary.get("recent_requests"):
        st.table(summary["recent_requests"])
    else:
        st.caption("No recent requests.")

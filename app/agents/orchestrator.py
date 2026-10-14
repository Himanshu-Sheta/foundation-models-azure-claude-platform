import logging
import time
from typing import Any, Dict, List, Optional
from app.agents.tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from app.providers.router import ModelRouter
from app.schemas.models import ChatMessage, CompletionRequest

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Enterprise Agentic Orchestrator.
    Manages multi-turn reasoning loops, tool calling execution, and final synthesis.
    """

    def __init__(self, router: ModelRouter):
        self.router = router
        self.tools = TOOL_DEFINITIONS
        self.tool_map = TOOL_REGISTRY

    async def run(
        self,
        task_prompt: str,
        provider: Optional[str] = "anthropic",
        max_steps: int = 5,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agentic reasoning loop until completion or max_steps reached.
        """
        start_time = time.perf_counter()
        effective_system = system_prompt or (
            "You are an enterprise AI architect agent. Use your available tools when necessary "
            "to retrieve factual data, calculate metrics, or verify system status before answering."
        )

        history: List[ChatMessage] = [ChatMessage(role="user", content=task_prompt)]
        execution_trace: List[Dict[str, Any]] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for step in range(1, max_steps + 1):
            req = CompletionRequest(
                messages=history,
                provider=provider,
                system_prompt=effective_system,
                tools=self.tools,
                temperature=0.2,
                enable_safety_check=True,
            )

            resp = await self.router.route_completion(req)
            total_input_tokens += resp.input_tokens
            total_output_tokens += resp.output_tokens
            total_cost += resp.estimated_cost_usd

            # If no tools called, we reached final answer
            if not resp.tool_calls:
                execution_trace.append({
                    "step": step,
                    "action": "final_synthesis",
                    "content": resp.content,
                    "provider": resp.provider,
                    "latency_ms": resp.latency_ms,
                })
                break

            # Handle Tool Calls
            for tool_call in resp.tool_calls:
                tool_fn = self.tool_map.get(tool_call.name)
                if tool_fn:
                    try:
                        tool_output = tool_fn(**tool_call.arguments)
                    except Exception as e:
                        tool_output = f"Tool execution failed: {str(e)}"
                else:
                    tool_output = f"Unknown tool: {tool_call.name}"

                execution_trace.append({
                    "step": step,
                    "action": "tool_call",
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": tool_output,
                    "provider": resp.provider,
                })

                # Append tool observation
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=resp.content or f"Executing {tool_call.name}...",
                    )
                )
                history.append(
                    ChatMessage(
                        role="user",
                        content=f"[Tool Result for {tool_call.name}]: {tool_output}",
                        name=tool_call.name,
                    )
                )

        # If loop finished after tool calls, generate final synthesis
        if execution_trace and execution_trace[-1]["action"] == "tool_call":
            history.append(
                ChatMessage(
                    role="user",
                    content="Please summarize the final answer based on the tool results above.",
                )
            )
            final_req = CompletionRequest(
                messages=history,
                provider=provider,
                system_prompt=effective_system,
                tools=None,
                temperature=0.3,
            )
            final_resp = await self.router.route_completion(final_req)
            total_input_tokens += final_resp.input_tokens
            total_output_tokens += final_resp.output_tokens
            total_cost += final_resp.estimated_cost_usd

            final_text = final_resp.content
            execution_trace.append({
                "step": len(execution_trace) + 1,
                "action": "final_synthesis",
                "content": final_text,
                "provider": final_resp.provider,
                "latency_ms": final_resp.latency_ms,
            })
        else:
            final_text = execution_trace[-1]["content"] if execution_trace else ""

        total_latency = (time.perf_counter() - start_time) * 1000

        return {
            "task": task_prompt,
            "final_answer": final_text,
            "steps_taken": len(execution_trace),
            "trace": execution_trace,
            "metrics": {
                "total_latency_ms": round(total_latency, 2),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_usd": round(total_cost, 6),
            },
        }

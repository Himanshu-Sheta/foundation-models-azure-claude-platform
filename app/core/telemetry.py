import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List


# Pricing per 1M tokens (USD)
PRICING_TABLE = {
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    # Azure OpenAI
    "gpt-4o": {"input": 5.00, "output": 15.00, "cache_read": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_read": 0.0},
    # Fallback default
    "default": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
}


@dataclass
class RequestRecord:
    timestamp: float
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    success: bool
    error_type: str = ""


class TelemetryTracker:
    """
    Enterprise Observability & Cost Tracking for Multi-Foundation Model System.
    """
    def __init__(self):
        self._lock = Lock()
        self.records: List[RequestRecord] = []
        self.provider_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "successful": 0,
                "failed": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_latency_ms": 0.0,
            }
        )

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> float:
        """Calculate approximate USD cost based on token counts."""
        pricing = PRICING_TABLE.get(model, PRICING_TABLE["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cache_cost = (cached_tokens / 1_000_000) * pricing["cache_read"]
        return round(input_cost + output_cost + cache_cost, 6)

    def record_request(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        success: bool = True,
        error_type: str = "",
    ) -> float:
        cost = self.calculate_cost(model, input_tokens, output_tokens, cached_tokens)

        record = RequestRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            success=success,
            error_type=error_type,
        )

        with self._lock:
            self.records.append(record)
            stats = self.provider_stats[provider]
            stats["requests"] += 1
            if success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            stats["total_tokens"] += (input_tokens + output_tokens)
            stats["total_cost"] += cost
            stats["total_latency_ms"] += latency_ms

        return cost

    def get_summary(self) -> Dict[str, Any]:
        """Return executive observability summary."""
        with self._lock:
            total_reqs = len(self.records)
            if total_reqs == 0:
                return {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "average_latency_ms": 0.0,
                    "success_rate_percent": 100.0,
                    "providers": {},
                }

            successful = sum(1 for r in self.records if r.success)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in self.records)
            total_cost = sum(r.cost_usd for r in self.records)
            avg_latency = sum(r.latency_ms for r in self.records) / total_reqs

            provider_breakdown = {}
            for p, s in self.provider_stats.items():
                p_reqs = s["requests"]
                provider_breakdown[p] = {
                    "requests": p_reqs,
                    "success_rate": round((s["successful"] / p_reqs * 100), 2) if p_reqs else 0,
                    "total_tokens": s["total_tokens"],
                    "total_cost_usd": round(s["total_cost"], 4),
                    "avg_latency_ms": round(s["total_latency_ms"] / p_reqs, 2) if p_reqs else 0,
                }

            return {
                "total_requests": total_reqs,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "average_latency_ms": round(avg_latency, 2),
                "success_rate_percent": round((successful / total_reqs * 100), 2),
                "providers": provider_breakdown,
                "recent_requests": [
                    {
                        "time": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
                        "provider": r.provider,
                        "model": r.model,
                        "latency_ms": round(r.latency_ms, 1),
                        "tokens": r.input_tokens + r.output_tokens,
                        "cost": r.cost_usd,
                        "status": "Success" if r.success else f"Failed: {r.error_type}",
                    }
                    for r in self.records[-10:]
                ],
            }


# Global singleton tracker
global_tracker = TelemetryTracker()

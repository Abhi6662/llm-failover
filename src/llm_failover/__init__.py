"""LLM Failover - OpenAI-compatible client with automatic endpoint failover."""

__version__ = "0.5.0"

from .client import EndpointConfig, FailoverClient, parse_retry_after
from .telemetry import Telemetry
from .circuit_breaker import CircuitBreaker, CircuitState, StateChangeCallback

__all__ = [
    "EndpointConfig",
    "FailoverClient",
    "parse_retry_after",
    "Telemetry",
    "CircuitBreaker",
    "CircuitState",
    "StateChangeCallback",
]

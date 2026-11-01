"""LLM Failover - OpenAI-compatible client with automatic endpoint failover."""

__version__ = "0.3.0"

from .client import EndpointConfig, FailoverClient
from .telemetry import Telemetry
from .circuit_breaker import CircuitBreaker, CircuitState

__all__ = ["EndpointConfig", "FailoverClient", "Telemetry", "CircuitBreaker", "CircuitState"]

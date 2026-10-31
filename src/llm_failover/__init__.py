"""LLM Failover - OpenAI-compatible client with automatic endpoint failover."""

__version__ = "0.1.0"

from .client import EndpointConfig, FailoverClient
from .telemetry import Telemetry

__all__ = ["EndpointConfig", "FailoverClient", "Telemetry"]

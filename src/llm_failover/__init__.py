"""LLM Failover - OpenAI-compatible client with automatic endpoint failover."""

__version__ = "0.1.0"

from .client import FailoverClient
from .telemetry import Telemetry

__all__ = ["FailoverClient", "Telemetry"]

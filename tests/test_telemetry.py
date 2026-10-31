"""Tests for Telemetry."""

import tempfile
from pathlib import Path
import pytest
from llm_failover import Telemetry


def test_telemetry_initialization():
    """Test telemetry initialization."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        telemetry = Telemetry(db_path=tmp.name)
        assert Path(tmp.name).exists()


def test_log_and_stats():
    """Test logging and retrieving stats."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        telemetry = Telemetry(db_path=tmp.name)
        
        # Log some requests
        telemetry.log("https://api.test.com", tokens=100, latency_ms=250)
        telemetry.log("https://api.test.com", tokens=150, latency_ms=300)
        
        # Check stats
        stats = telemetry.get_stats()
        assert stats["total_tokens"] == 250
        assert stats["request_count"] == 2
        assert stats["avg_latency_ms"] > 0

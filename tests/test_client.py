"""Tests for FailoverClient."""

import pytest
from llm_failover import FailoverClient, EndpointConfig


def test_client_initialization():
    """Test client initialization."""
    endpoints = [
        EndpointConfig(url="https://api.example.com/v1", api_key="test-key")
    ]
    client = FailoverClient(endpoints=endpoints)
    assert len(client.endpoints) == 1


def test_empty_endpoints_raises():
    """Test that empty endpoints list raises ValueError."""
    with pytest.raises(ValueError, match="At least one endpoint required"):
        FailoverClient(endpoints=[])


def test_cooldown_tracking():
    """Test cooldown mechanism."""
    endpoints = [
        EndpointConfig(url="https://api.example.com/v1", api_key="test-key", cooldown_seconds=10)
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Initially available
    assert client._is_available(endpoints[0])
    
    # Set cooldown
    client._set_cooldown(endpoints[0])
    
    # Now unavailable
    assert not client._is_available(endpoints[0])

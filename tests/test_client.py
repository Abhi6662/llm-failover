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


def test_backoff_time():
    """Test backoff calculation with jitter."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key")
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Backoff should increase with attempts
    backoff_0 = client._get_backoff_time(0)
    backoff_2 = client._get_backoff_time(2)
    
    assert backoff_0 < backoff_2
    assert 1 <= backoff_0 <= 2  # 2^0 + jitter
    assert 4 <= backoff_2 <= 5  # 2^2 + jitter


def test_endpoint_health():
    """Test endpoint health summary."""
    endpoints = [
        EndpointConfig(url="https://api.test1.com/v1", api_key="test-key"),
        EndpointConfig(url="https://api.test2.com/v1", api_key="test-key")
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Initially all available
    health = client.endpoint_health()
    assert len(health) == 2
    assert health["https://api.test1.com/v1"]["available"] is True
    
    # Set cooldown on first endpoint
    client._set_cooldown(endpoints[0])
    health = client.endpoint_health()
    assert health["https://api.test1.com/v1"]["available"] is False
    assert health["https://api.test2.com/v1"]["available"] is True

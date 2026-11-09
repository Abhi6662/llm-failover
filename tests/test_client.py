"""Tests for FailoverClient."""

import pytest
from unittest.mock import MagicMock, patch
from llm_failover import FailoverClient, EndpointConfig
from llm_failover.client import parse_retry_after


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


# --- Retry-After header tests ---


def test_parse_retry_after_seconds():
    """Test parsing Retry-After header in seconds format."""
    assert parse_retry_after("120") == 120
    assert parse_retry_after("  60  ") == 60
    assert parse_retry_after("0") == 0


def test_parse_retry_after_http_date():
    """Test parsing Retry-After header in HTTP-date format."""
    from email.utils import formatdate
    import time
    
    # Future date
    future_time = time.time() + 300
    http_date = formatdate(future_time, usegmt=True)
    result = parse_retry_after(http_date)
    assert result is not None
    assert 290 <= result <= 310  # Allow some tolerance


def test_parse_retry_after_invalid():
    """Test parsing invalid Retry-After header returns None."""
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("invalid") is None
    assert parse_retry_after("not-a-date") is None


def test_set_cooldown_uses_retry_after():
    """Test that _set_cooldown respects Retry-After header."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Set cooldown with Retry-After = 30 seconds
    client._set_cooldown(endpoints[0], retry_after_seconds=30)
    
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 30
    
    # Should be approximately 30 seconds (allow small delta)
    assert abs(cooldown_until - expected) < 2


def test_set_cooldown_caps_retry_after():
    """Test that Retry-After is capped at 300 seconds."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Set cooldown with huge Retry-After = 1000 seconds (should cap at 300)
    client._set_cooldown(endpoints[0], retry_after_seconds=1000)
    
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 300
    
    assert abs(cooldown_until - expected) < 2


def test_set_cooldown_fallback_to_default():
    """Test that _set_cooldown falls back to default when retry_after is None."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)
    
    # Set cooldown without Retry-After (None)
    client._set_cooldown(endpoints[0], retry_after_seconds=None)
    
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 60
    
    assert abs(cooldown_until - expected) < 2


# --- Integration tests with mocked HTTP responses ---


def test_chat_completion_uses_retry_after_header():
    """Test that chat_completion respects Retry-After header from 429 response."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "45"}
    mock_response.raise_for_status.side_effect = Exception("Rate limited")

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("llm_failover.client.httpx.Client", return_value=mock_client):
        try:
            client.chat_completion(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            )
        except RuntimeError:
            pass  # Expected - all endpoints failed

    # Check that cooldown was set to 45 seconds (Retry-After) not 60 (default)
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 45
    assert abs(cooldown_until - expected) < 2


def test_chat_completion_uses_retry_after_http_date():
    """Test that chat_completion respects Retry-After header in HTTP-date format."""
    from email.utils import formatdate
    import time
    
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)

    future_time = time.time() + 120
    http_date = formatdate(future_time, usegmt=True)

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": http_date}
    mock_response.raise_for_status.side_effect = Exception("Rate limited")

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("llm_failover.client.httpx.Client", return_value=mock_client):
        try:
            client.chat_completion(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            )
        except RuntimeError:
            pass  # Expected - all endpoints failed

    # Check that cooldown was set to ~120 seconds (Retry-After) not 60 (default)
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 120
    assert abs(cooldown_until - expected) < 5  # More tolerance for date parsing


def test_http_status_error_extracts_retry_after():
    """Test that HTTPStatusError extracts Retry-After header."""
    import httpx
    
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)

    mock_response = MagicMock()
    mock_response.headers = {"retry-after": "90"}
    mock_response.status_code = 429

    http_error = httpx.HTTPStatusError(
        "Rate limited", request=MagicMock(), response=mock_response
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = http_error
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("llm_failover.client.httpx.Client", return_value=mock_client):
        try:
            client.chat_completion(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            )
        except RuntimeError:
            pass  # Expected - all endpoints failed

    # Check that cooldown was set to 90 seconds (Retry-After) not 60 (default)
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 90
    assert abs(cooldown_until - expected) < 2


def test_no_retry_after_falls_back_to_default():
    """Test that missing Retry-After header falls back to default cooldown."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key", cooldown_seconds=60)
    ]
    client = FailoverClient(endpoints=endpoints)

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}  # No Retry-After header
    mock_response.raise_for_status.side_effect = Exception("Rate limited")

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("llm_failover.client.httpx.Client", return_value=mock_client):
        try:
            client.chat_completion(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            )
        except RuntimeError:
            pass  # Expected - all endpoints failed

    # Check that cooldown was set to 60 seconds (default) not Retry-After
    import time
    cooldown_until = client._cooldowns[endpoints[0].url]
    expected = time.time() + 60
    assert abs(cooldown_until - expected) < 2

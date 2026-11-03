"""Tests for streaming chat completions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm_failover import FailoverClient, EndpointConfig
from llm_failover.circuit_breaker import CircuitBreaker, CircuitState


def test_chat_completion_stream_flag():
    """Test that stream=True returns async generator."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key")
    ]
    client = FailoverClient(endpoints=endpoints)
    result = client.chat_completion(
        [{"role": "user", "content": "Hello"}],
        model="gpt-4",
        stream=True,
    )
    assert hasattr(result, "__aiter__")


def test_chat_completion_stream_default():
    """Test that stream=False (default) returns dict (mocked)."""
    endpoints = [
        EndpointConfig(url="https://api.test.com/v1", api_key="test-key")
    ]
    client = FailoverClient(endpoints=endpoints)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}], "usage": {"total_tokens": 10}}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("llm_failover.client.httpx.Client", return_value=mock_client):
        result = client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )

    assert result == {"choices": [{"message": {"content": "Hi"}}], "usage": {"total_tokens": 10}}


class _MockAsyncResponse:
    """Properly mock async response for streaming."""
    def __init__(self, sse_lines):
        self._lines = list(sse_lines)
        self.status_code = 200
        
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._lines:
            return self._lines.pop(0)
        raise StopAsyncIteration
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    def raise_for_status(self):
        pass
    
    def aiter_lines(self):
        """Return self as async iterator for SSE line iteration (httpx decodes to str)."""
        return self


class _MockStreamContext:
    """Mock async context manager for client.stream()"""
    def __init__(self, response):
        self._response = response
    
    async def __aenter__(self):
        return self._response
    
    async def __aexit__(self, *args):
        pass


def _make_streaming_mock(sse_lines):
    """Create a properly mocked async client for streaming."""
    mock_response = _MockAsyncResponse(sse_lines.copy())
    mock_stream_ctx = _MockStreamContext(mock_response)
    
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    return mock_client


@pytest.mark.asyncio
async def test_streaming_happy_path():
    """Test streaming returns chunks from a healthy endpoint."""
    endpoints = [EndpointConfig(url="https://api.test.com/v1", api_key="test-key")]
    client = FailoverClient(endpoints=endpoints)

    # httpx's aiter_lines() decodes bytes to str, so yield str
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        'data: [DONE]\n\n',
    ]

    mock_client = _make_streaming_mock(sse_lines)

    with patch("llm_failover.client.httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        ):
            chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0] == {"choices": [{"delta": {"content": "Hello"}}]}
    assert chunks[1] == {"choices": [{"delta": {"content": " world"}}]}


@pytest.mark.asyncio
async def test_streaming_endpoint_failover():
    """Test streaming fails over to second endpoint when first fails."""
    endpoints = [
        EndpointConfig(url="https://api.fail.com/v1", api_key="key1"),
        EndpointConfig(url="https://api.ok.com/v1", api_key="key2"),
    ]
    client = FailoverClient(endpoints=endpoints)

    # First endpoint raises on __aenter__
    mock_fail_client = AsyncMock()
    mock_fail_client.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
    mock_fail_client.__aexit__ = AsyncMock(return_value=None)

    # Second endpoint returns SSE
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        'data: [DONE]\n\n',
    ]
    mock_ok_client = _make_streaming_mock(sse_lines)

    call_count = {"n": 0}

    def client_factory(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return mock_fail_client
        return mock_ok_client

    with patch("llm_failover.client.httpx.AsyncClient", side_effect=client_factory):
        chunks = []
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        ):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == {"choices": [{"delta": {"content": "Hi"}}]}


@pytest.mark.asyncio
async def test_streaming_all_endpoints_fail():
    """Test streaming raises RuntimeError when all endpoints fail."""
    endpoints = [
        EndpointConfig(url="https://api.fail1.com/v1", api_key="key1"),
        EndpointConfig(url="https://api.fail2.com/v1", api_key="key2"),
    ]
    client = FailoverClient(endpoints=endpoints)

    mock_fail_client = AsyncMock()
    mock_fail_client.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
    mock_fail_client.__aexit__ = AsyncMock(return_value=None)

    with patch("llm_failover.client.httpx.AsyncClient", return_value=mock_fail_client):
        with pytest.raises(RuntimeError, match="All endpoints failed"):
            async for _ in client.chat_completion_stream(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            ):
                pass


@pytest.mark.asyncio
async def test_streaming_unavailable_endpoint_skipped():
    """Test that endpoints in cooldown are skipped during streaming."""
    endpoints = [
        EndpointConfig(url="https://api.cold.com/v1", api_key="key1", cooldown_seconds=99999),
        EndpointConfig(url="https://api.ok.com/v1", api_key="key2"),
    ]
    client = FailoverClient(endpoints=endpoints)

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        'data: [DONE]\n\n',
    ]
    mock_ok_client = _make_streaming_mock(sse_lines)

    call_count = {"n": 0}

    def client_factory(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise AssertionError("Cooled-down endpoint should be skipped")
        return mock_ok_client

    with patch("llm_failover.client.httpx.AsyncClient", side_effect=client_factory):
        chunks = []
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        ):
            chunks.append(chunk)

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_streaming_records_circuit_failure():
    """Test that streaming failure records circuit breaker failure."""
    circuit_breaker = CircuitBreaker(failure_threshold=1)
    endpoints = [EndpointConfig(url="https://api.fail.com/v1", api_key="test-key")]
    client = FailoverClient(endpoints=endpoints, circuit_breaker=circuit_breaker)

    mock_fail_client = AsyncMock()
    mock_fail_client.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
    mock_fail_client.__aexit__ = AsyncMock(return_value=None)

    with patch("llm_failover.client.httpx.AsyncClient", return_value=mock_fail_client):
        try:
            async for _ in client.chat_completion_stream(
                [{"role": "user", "content": "Hello"}],
                model="gpt-4",
            ):
                pass
        except RuntimeError:
            pass

    assert client.circuit_breaker.get_state("https://api.fail.com/v1") == CircuitState.OPEN


@pytest.mark.asyncio
async def test_streaming_parses_done_signal():
    """Test that [DONE] signal stops iteration."""
    endpoints = [EndpointConfig(url="https://api.test.com/v1", api_key="test-key")]
    client = FailoverClient(endpoints=endpoints)

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"A"}}]}\n\n',
        'data: [DONE]\n\n',
        'data: {"choices":[{"delta":{"content":"B"}}]}\n\n',  # Should NOT be yielded
    ]

    mock_client = _make_streaming_mock(sse_lines)

    with patch("llm_failover.client.httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        ):
            chunks.append(chunk)

    assert len(chunks) == 1  # Stops at [DONE], never sees "B"


@pytest.mark.asyncio
async def test_streaming_empty_lines_skipped():
    """Test that blank lines are skipped during SSE parsing."""
    endpoints = [EndpointConfig(url="https://api.test.com/v1", api_key="test-key")]
    client = FailoverClient(endpoints=endpoints)

    sse_lines = [
        '\n\n',  # blank line
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        '\n',  # another blank
        'data: [DONE]\n\n',
    ]

    mock_client = _make_streaming_mock(sse_lines)

    with patch("llm_failover.client.httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4",
        ):
            chunks.append(chunk)

    assert len(chunks) == 1


def test_parse_sse_line_done():
    """Test _parse_sse_line recognizes [DONE]."""
    client = FailoverClient([EndpointConfig(url="https://x.com", api_key="k")])
    result = client._parse_sse_line("data: [DONE]")
    assert result == {"done": True}


def test_parse_sse_line_json():
    """Test _parse_sse_line parses JSON data."""
    client = FailoverClient([EndpointConfig(url="https://x.com", api_key="k")])
    result = client._parse_sse_line('data: {"foo":"bar"}')
    assert result == {"foo": "bar"}


def test_parse_sse_line_ignores_non_data():
    """Test _parse_sse_line ignores lines not starting with data:."""
    client = FailoverClient([EndpointConfig(url="https://x.com", api_key="k")])
    assert client._parse_sse_line(": comment") is None
    assert client._parse_sse_line("") is None
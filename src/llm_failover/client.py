"""OpenAI-compatible client with automatic failover."""

import json
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from pydantic import BaseModel
from .circuit_breaker import CircuitBreaker


def parse_retry_after(header_value: Optional[str]) -> Optional[int]:
    """
    Parse Retry-After header value into seconds.

    Supports two formats per RFC 7231:
    - Seconds: "120" (delay in seconds)
    - HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT" (absolute time)

    Args:
        header_value: Raw Retry-After header value

    Returns:
        Seconds to wait, or None if parsing fails
    """
    if not header_value:
        return None

    header_value = header_value.strip()

    # Try seconds format (integer)
    try:
        return int(header_value)
    except ValueError:
        pass

    # Try HTTP-date format
    try:
        # Parse HTTP date (RFC 7231 / RFC 1123)
        from email.utils import parsedate_to_datetime
        retry_date = parsedate_to_datetime(header_value)
        if retry_date:
            delay = int(retry_date.timestamp() - time.time())
            return max(0, delay)
    except Exception:
        pass

    return None


class EndpointConfig(BaseModel):
    """Endpoint configuration."""
    url: str
    api_key: str
    cooldown_seconds: int = 60
    timeout_seconds: int = 30


class FailoverClient:
    """OpenAI-compatible client with automatic endpoint rotation."""

    def __init__(self, endpoints: List[EndpointConfig], telemetry: Optional[Any] = None,
                 circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize client with multiple endpoints.

        Args:
            endpoints: List of endpoint configurations
            telemetry: Optional telemetry tracker
            circuit_breaker: Optional circuit breaker (default: new instance)
        """
        if not endpoints:
            raise ValueError("At least one endpoint required")
        self.endpoints = endpoints
        self.telemetry = telemetry
        self._cooldowns: Dict[str, float] = {}
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def _is_available(self, endpoint: EndpointConfig) -> bool:
        """Check if endpoint is available (not in cooldown and circuit closed)."""
        cooldown_until = self._cooldowns.get(endpoint.url, 0)
        cooldown_ok = time.time() >= cooldown_until
        circuit_ok = self.circuit_breaker.is_available(endpoint.url)
        return cooldown_ok and circuit_ok

    def _set_cooldown(self, endpoint: EndpointConfig, retry_after_seconds: Optional[int] = None) -> None:
        """Set cooldown for endpoint.

        Args:
            endpoint: The endpoint to set cooldown on
            retry_after_seconds: Optional Retry-After value from response header (takes precedence)
        """
        if retry_after_seconds is not None:
            # Use Retry-After header if provided, with a reasonable cap
            self._cooldowns[endpoint.url] = time.time() + min(retry_after_seconds, 300)
        else:
            self._cooldowns[endpoint.url] = time.time() + endpoint.cooldown_seconds

    def _get_backoff_time(self, attempt: int) -> float:
        """Calculate backoff time with jitter.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Backoff time in seconds with jitter
        """
        base_delay = min(60, 2 ** attempt)
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter

    def _parse_sse_line(self, line: str) -> Optional[Dict]:
        """Parse a single SSE line into a dict."""
        if not line.startswith("data: "):
            return None
        data = line[6:].strip()
        if data == "[DONE]":
            return {"done": True}
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async def _stream_from_endpoint(
        self,
        endpoint: EndpointConfig,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion from a single endpoint.

        Yields:
            Parsed SSE data chunks from the streaming response
        """
        start_time = time.time()
        total_tokens = 0

        async with httpx.AsyncClient(timeout=endpoint.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{endpoint.url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {endpoint.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": True,
                    **kwargs
                }
            ) as response:
                if response.status_code in (429, 502, 503):
                    # Check for Retry-After header
                    retry_after = None
                    if "retry-after" in response.headers:
                        retry_after = parse_retry_after(response.headers["retry-after"])
                    self._set_cooldown(endpoint, retry_after)
                    self.circuit_breaker.record_failure(endpoint.url)
                    raise RuntimeError(f"HTTP {response.status_code}")

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    parsed = self._parse_sse_line(line)
                    if parsed is None:
                        continue
                    if parsed.get("done"):
                        break

                    # Extract token from chunk
                    choices = parsed.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            total_tokens += len(content.split())

                    yield parsed

                latency_ms = int((time.time() - start_time) * 1000)

                # Record success in circuit breaker
                self.circuit_breaker.record_success(endpoint.url)

                if self.telemetry:
                    self.telemetry.log(endpoint.url, total_tokens, latency_ms)

    def endpoint_health(self) -> Dict[str, Dict]:
        """Get health summary of all endpoints.

        Returns:
            Dictionary mapping endpoint URLs to their health status
        """
        health = {}
        current_time = time.time()

        for endpoint in self.endpoints:
            cooldown_until = self._cooldowns.get(endpoint.url, 0)
            cooldown_ok = current_time >= cooldown_until

            # Get circuit breaker health
            circuit_health = self.circuit_breaker.get_health(endpoint.url)

            health[endpoint.url] = {
                "available": cooldown_ok and circuit_health["available"],
                "cooldown_remaining": max(0, int(cooldown_until - current_time)),
                "circuit_state": circuit_health["state"],
                "circuit_failures": circuit_health["failures"]
            }

            if "recovery_in" in circuit_health:
                health[endpoint.url]["circuit_recovery_in"] = circuit_health["recovery_in"]

        return health

    def chat_completion(self, messages: List[Dict[str, str]], model: str,
                       max_tokens: int = 100, stream: bool = False, **kwargs) -> Any:
        """Send chat completion request with automatic failover.

        Args:
            messages: Chat messages
            model: Model identifier
            max_tokens: Maximum tokens
            stream: If True, return async generator for streaming
            **kwargs: Additional parameters

        Returns:
            API response dictionary (non-streaming) or async generator (streaming)

        Raises:
            RuntimeError: If all endpoints fail
        """
        if stream:
            return self.chat_completion_stream(messages, model, max_tokens, **kwargs)

        last_error = None

        for endpoint in self.endpoints:
            if not self._is_available(endpoint):
                continue

            try:
                start_time = time.time()

                with httpx.Client(timeout=endpoint.timeout_seconds) as client:
                    response = client.post(
                        f"{endpoint.url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {endpoint.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            **kwargs
                        }
                    )

                    latency_ms = int((time.time() - start_time) * 1000)

                    if response.status_code in (429, 502, 503):
                        # Check for Retry-After header
                        retry_after = None
                        if "retry-after" in response.headers:
                            retry_after = parse_retry_after(response.headers["retry-after"])
                        self._set_cooldown(endpoint, retry_after)
                        self.circuit_breaker.record_failure(endpoint.url)
                        last_error = f"HTTP {response.status_code}"
                        continue

                    response.raise_for_status()
                    result = response.json()

                    # Record success in circuit breaker
                    self.circuit_breaker.record_success(endpoint.url)

                    if self.telemetry:
                        tokens = result.get("usage", {}).get("total_tokens", 0)
                        self.telemetry.log(endpoint.url, tokens, latency_ms)

                    return result

            except httpx.TimeoutException as e:
                self._set_cooldown(endpoint)
                self.circuit_breaker.record_failure(endpoint.url)
                last_error = str(e)
                continue
            except httpx.HTTPStatusError as e:
                # Extract Retry-After header if present
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    if "retry-after" in e.response.headers:
                        retry_after = parse_retry_after(e.response.headers["retry-after"])
                self._set_cooldown(endpoint, retry_after)
                self.circuit_breaker.record_failure(endpoint.url)
                last_error = str(e)
                continue
            except Exception as e:
                self.circuit_breaker.record_failure(endpoint.url)
                last_error = str(e)
                continue

        raise RuntimeError(f"All endpoints failed. Last error: {last_error}")

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 100,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion with automatic failover.

        Args:
            messages: Chat messages
            model: Model identifier
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Parsed SSE data chunks from the streaming response

        Raises:
            RuntimeError: If all endpoints fail
        """
        last_error = None

        for endpoint in self.endpoints:
            if not self._is_available(endpoint):
                continue

            try:
                async for chunk in self._stream_from_endpoint(
                    endpoint, messages, model, max_tokens, **kwargs
                ):
                    yield chunk
                return  # Success - exit the method

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                self._set_cooldown(endpoint)
                self.circuit_breaker.record_failure(endpoint.url)
                last_error = str(e)
                continue
            except Exception as e:
                self.circuit_breaker.record_failure(endpoint.url)
                last_error = str(e)
                continue

        raise RuntimeError(f"All endpoints failed. Last error: {last_error}")
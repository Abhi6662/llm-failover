"""OpenAI-compatible client with automatic failover."""

import random
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel


class EndpointConfig(BaseModel):
    """Endpoint configuration."""
    url: str
    api_key: str
    cooldown_seconds: int = 60
    timeout_seconds: int = 30


class FailoverClient:
    """OpenAI-compatible client with automatic endpoint rotation."""
    
    def __init__(self, endpoints: List[EndpointConfig], telemetry: Optional[Any] = None):
        """Initialize client with multiple endpoints.
        
        Args:
            endpoints: List of endpoint configurations
            telemetry: Optional telemetry tracker
        """
        if not endpoints:
            raise ValueError("At least one endpoint required")
        self.endpoints = endpoints
        self.telemetry = telemetry
        self._cooldowns: Dict[str, float] = {}
    
    def _is_available(self, endpoint: EndpointConfig) -> bool:
        """Check if endpoint is available (not in cooldown)."""
        cooldown_until = self._cooldowns.get(endpoint.url, 0)
        return time.time() >= cooldown_until
    
    def _set_cooldown(self, endpoint: EndpointConfig) -> None:
        """Set cooldown for endpoint."""
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
    
    def endpoint_health(self) -> Dict[str, Dict]:
        """Get health summary of all endpoints.
        
        Returns:
            Dictionary mapping endpoint URLs to their health status
        """
        health = {}
        current_time = time.time()
        
        for endpoint in self.endpoints:
            cooldown_until = self._cooldowns.get(endpoint.url, 0)
            is_available = current_time >= cooldown_until
            
            health[endpoint.url] = {
                "available": is_available,
                "cooldown_remaining": max(0, int(cooldown_until - current_time))
            }
        
        return health
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str, 
                       max_tokens: int = 100, **kwargs) -> Dict[str, Any]:
        """Send chat completion request with automatic failover.
        
        Args:
            messages: Chat messages
            model: Model identifier
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If all endpoints fail
        """
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
                        self._set_cooldown(endpoint)
                        last_error = f"HTTP {response.status_code}"
                        continue
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if self.telemetry:
                        tokens = result.get("usage", {}).get("total_tokens", 0)
                        self.telemetry.log(endpoint.url, tokens, latency_ms)
                    
                    return result
                    
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                self._set_cooldown(endpoint)
                last_error = str(e)
                continue
            except Exception as e:
                last_error = str(e)
                continue
        
        raise RuntimeError(f"All endpoints failed. Last error: {last_error}")

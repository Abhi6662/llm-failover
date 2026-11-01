"""Circuit breaker for endpoint health management."""

import time
from enum import Enum
from typing import Dict


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for tracking endpoint health state.
    
    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Too many failures, block requests until timeout
    - HALF_OPEN: Testing if endpoint recovered, limited requests allowed
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._states: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
    
    def get_state(self, endpoint_url: str) -> CircuitState:
        """Get current circuit state for endpoint.
        
        Args:
            endpoint_url: Endpoint URL
            
        Returns:
            Current circuit state
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self._states.get(endpoint_url) == CircuitState.OPEN:
            opened_at = self._opened_at.get(endpoint_url, 0)
            if time.time() >= opened_at + self.recovery_timeout:
                self._states[endpoint_url] = CircuitState.HALF_OPEN
        
        return self._states.get(endpoint_url, CircuitState.CLOSED)
    
    def record_success(self, endpoint_url: str) -> None:
        """Record successful request.
        
        Args:
            endpoint_url: Endpoint URL
        """
        current_state = self.get_state(endpoint_url)
        
        if current_state == CircuitState.HALF_OPEN:
            # Recovery confirmed, close circuit
            self._states[endpoint_url] = CircuitState.CLOSED
            self._failure_counts[endpoint_url] = 0
        elif current_state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_counts[endpoint_url] = 0
    
    def record_failure(self, endpoint_url: str) -> None:
        """Record failed request.
        
        Args:
            endpoint_url: Endpoint URL
        """
        current_state = self.get_state(endpoint_url)
        
        if current_state == CircuitState.HALF_OPEN:
            # Recovery failed, reopen circuit
            self._states[endpoint_url] = CircuitState.OPEN
            self._opened_at[endpoint_url] = time.time()
            return
        
        # Increment failure count
        failures = self._failure_counts.get(endpoint_url, 0) + 1
        self._failure_counts[endpoint_url] = failures
        
        # Open circuit if threshold exceeded
        if failures >= self.failure_threshold:
            self._states[endpoint_url] = CircuitState.OPEN
            self._opened_at[endpoint_url] = time.time()
    
    def is_available(self, endpoint_url: str) -> bool:
        """Check if endpoint is available for requests.
        
        Args:
            endpoint_url: Endpoint URL
            
        Returns:
            True if requests allowed, False if circuit is open
        """
        state = self.get_state(endpoint_url)
        return state != CircuitState.OPEN
    
    def get_health(self, endpoint_url: str) -> Dict:
        """Get health summary for endpoint.
        
        Args:
            endpoint_url: Endpoint URL
            
        Returns:
            Health summary with state and failure count
        """
        state = self.get_state(endpoint_url)
        failures = self._failure_counts.get(endpoint_url, 0)
        
        health = {
            "state": state.value,
            "failures": failures,
            "available": state != CircuitState.OPEN
        }
        
        if state == CircuitState.OPEN:
            opened_at = self._opened_at.get(endpoint_url, 0)
            recovery_in = max(0, int((opened_at + self.recovery_timeout) - time.time()))
            health["recovery_in"] = recovery_in
        
        return health

"""Tests for CircuitBreaker."""

import time
import pytest
from llm_failover.circuit_breaker import CircuitBreaker, CircuitState


def test_initial_state_closed():
    """Test circuit starts in CLOSED state."""
    breaker = CircuitBreaker()
    assert breaker.get_state("https://api.test.com") == CircuitState.CLOSED
    assert breaker.is_available("https://api.test.com") is True


def test_circuit_opens_after_threshold():
    """Test circuit opens after failure threshold."""
    breaker = CircuitBreaker(failure_threshold=3)
    endpoint = "https://api.test.com"
    
    # Record failures up to threshold
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED
    
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED
    
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.OPEN
    assert breaker.is_available(endpoint) is False


def test_success_resets_failure_count():
    """Test success resets failure count in CLOSED state."""
    breaker = CircuitBreaker(failure_threshold=3)
    endpoint = "https://api.test.com"
    
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED
    
    breaker.record_success(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED
    
    # Should need 3 more failures to open
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED


def test_circuit_transitions_to_half_open():
    """Test circuit transitions to HALF_OPEN after timeout."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    endpoint = "https://api.test.com"
    
    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    assert breaker.get_state(endpoint) == CircuitState.HALF_OPEN
    assert breaker.is_available(endpoint) is True


def test_half_open_success_closes_circuit():
    """Test success in HALF_OPEN state closes circuit."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    endpoint = "https://api.test.com"
    
    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    
    # Wait and enter HALF_OPEN
    time.sleep(1.1)
    assert breaker.get_state(endpoint) == CircuitState.HALF_OPEN
    
    # Success should close circuit
    breaker.record_success(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.CLOSED


def test_half_open_failure_reopens_circuit():
    """Test failure in HALF_OPEN state reopens circuit."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    endpoint = "https://api.test.com"
    
    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    
    # Wait and enter HALF_OPEN
    time.sleep(1.1)
    assert breaker.get_state(endpoint) == CircuitState.HALF_OPEN
    
    # Failure should reopen circuit
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.OPEN
    assert breaker.is_available(endpoint) is False


def test_health_summary():
    """Test health summary includes state and failure count."""
    breaker = CircuitBreaker(failure_threshold=3)
    endpoint = "https://api.test.com"
    
    # Initial health
    health = breaker.get_health(endpoint)
    assert health["state"] == "closed"
    assert health["failures"] == 0
    assert health["available"] is True
    
    # After failures
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    health = breaker.get_health(endpoint)
    assert health["failures"] == 2
    assert health["available"] is True
    
    # After opening
    breaker.record_failure(endpoint)
    health = breaker.get_health(endpoint)
    assert health["state"] == "open"
    assert health["available"] is False
    assert "recovery_in" in health


def test_multiple_endpoints_independent():
    """Test circuit breakers for different endpoints are independent."""
    breaker = CircuitBreaker(failure_threshold=2)
    endpoint1 = "https://api1.test.com"
    endpoint2 = "https://api2.test.com"
    
    # Fail first endpoint
    breaker.record_failure(endpoint1)
    breaker.record_failure(endpoint1)
    
    # First endpoint should be open
    assert breaker.get_state(endpoint1) == CircuitState.OPEN
    assert breaker.is_available(endpoint1) is False
    
    # Second endpoint should still be closed
    assert breaker.get_state(endpoint2) == CircuitState.CLOSED
    assert breaker.is_available(endpoint2) is True

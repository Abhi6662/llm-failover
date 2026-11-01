"""Tests for CircuitBreaker."""

import time
import pytest
from llm_failover.circuit_breaker import CircuitBreaker, CircuitState, StateChangeCallback


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


# --- Event callback tests ---


def test_state_change_callback_on_open():
    """Test callback fires when circuit opens."""
    events = []

    def callback(endpoint, old_state, new_state):
        events.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, on_state_change=callback)
    endpoint = "https://api.test.com"

    # First failure - no state change (still CLOSED)
    breaker.record_failure(endpoint)
    assert len(events) == 0

    # Second failure - should transition to OPEN
    breaker.record_failure(endpoint)
    assert len(events) == 1
    assert events[0] == (endpoint, CircuitState.CLOSED, CircuitState.OPEN)


def test_state_change_callback_on_half_open():
    """Test callback fires when circuit transitions to HALF_OPEN."""
    events = []

    def callback(endpoint, old_state, new_state):
        events.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, on_state_change=callback)
    endpoint = "https://api.test.com"

    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert len(events) == 1  # CLOSED -> OPEN

    # Wait for recovery timeout
    time.sleep(1.1)

    # get_state should transition to HALF_OPEN
    state = breaker.get_state(endpoint)
    assert state == CircuitState.HALF_OPEN
    assert len(events) == 2
    assert events[1] == (endpoint, CircuitState.OPEN, CircuitState.HALF_OPEN)


def test_state_change_callback_on_close():
    """Test callback fires when circuit closes from HALF_OPEN."""
    events = []

    def callback(endpoint, old_state, new_state):
        events.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, on_state_change=callback)
    endpoint = "https://api.test.com"

    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)

    # Wait and enter HALF_OPEN
    time.sleep(1.1)
    breaker.get_state(endpoint)  # Trigger transition

    # Success should close circuit
    breaker.record_success(endpoint)
    assert len(events) == 3
    assert events[2] == (endpoint, CircuitState.HALF_OPEN, CircuitState.CLOSED)


def test_state_change_callback_on_reopen():
    """Test callback fires when HALF_OPEN failure reopens circuit."""
    events = []

    def callback(endpoint, old_state, new_state):
        events.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, on_state_change=callback)
    endpoint = "https://api.test.com"

    # Open circuit
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)

    # Wait and enter HALF_OPEN
    time.sleep(1.1)
    breaker.get_state(endpoint)

    # Failure should reopen circuit
    breaker.record_failure(endpoint)
    assert len(events) == 3
    assert events[2] == (endpoint, CircuitState.HALF_OPEN, CircuitState.OPEN)


def test_multiple_callbacks():
    """Test multiple callbacks all fire."""
    events1 = []
    events2 = []

    def callback1(endpoint, old_state, new_state):
        events1.append((endpoint, old_state, new_state))

    def callback2(endpoint, old_state, new_state):
        events2.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, on_state_change=callback1)
    breaker.add_state_change_callback(callback2)
    endpoint = "https://api.test.com"

    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)

    assert len(events1) == 1
    assert len(events2) == 1
    assert events1[0] == events2[0] == (endpoint, CircuitState.CLOSED, CircuitState.OPEN)


def test_add_callback_after_init():
    """Test adding callback after initialization."""
    events = []

    def callback(endpoint, old_state, new_state):
        events.append((endpoint, old_state, new_state))

    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    endpoint = "https://api.test.com"

    # Fail twice without callback - no events
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert len(events) == 0

    # Add callback
    breaker.add_state_change_callback(callback)

    # Next transition should fire
    breaker.record_failure(endpoint)  # This won't cause transition (already OPEN)
    assert len(events) == 0

    # Wait and let it transition to HALF_OPEN
    time.sleep(1.1)
    breaker.get_state(endpoint)
    assert len(events) == 1
    assert events[0] == (endpoint, CircuitState.OPEN, CircuitState.HALF_OPEN)


def test_callback_error_does_not_break_breaker():
    """Test that callback exceptions don't break circuit breaker logic."""
    def bad_callback(endpoint, old_state, new_state):
        raise RuntimeError("callback error")

    breaker = CircuitBreaker(failure_threshold=2, on_state_change=bad_callback)
    endpoint = "https://api.test.com"

    # Should not raise despite callback error
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)

    # Circuit should still be open
    assert breaker.get_state(endpoint) == CircuitState.OPEN
    assert breaker.is_available(endpoint) is False


def test_no_callback_by_default():
    """Test that callback is optional and works without it."""
    breaker = CircuitBreaker(failure_threshold=2)
    endpoint = "https://api.test.com"

    # Should work normally without any callbacks
    breaker.record_failure(endpoint)
    breaker.record_failure(endpoint)
    assert breaker.get_state(endpoint) == CircuitState.OPEN


def test_state_change_callback_type_export():
    """Test StateChangeCallback type is exported."""
    from llm_failover import StateChangeCallback
    assert StateChangeCallback is not None
    # Just verify it's a valid type annotation
    cb: StateChangeCallback = lambda e, o, n: None
    assert callable(cb)
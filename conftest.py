"""Pytest configuration shared across the test suite.

Living at the repo root, this file makes pytest add the repo root to
``sys.path`` so ``import src`` works, and provides an autouse fixture that
resets the global Kalman track-ID counter before each test so that track-ID
assertions are deterministic.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_kalman_counter():
    """Reset ``KalmanBoxTracker.count`` (a class-level global) around each
    test. Guarded so tests that don't need the tracker still run even if
    ``filterpy`` isn't installed."""
    try:
        from src.tracker import KalmanBoxTracker
    except Exception:
        yield
        return

    KalmanBoxTracker.count = 0
    yield
    KalmanBoxTracker.count = 0

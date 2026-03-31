"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from btom_engine.config import DEFAULT_HYPOTHESES
from btom_engine.engine import BTOMEngine


@pytest.fixture
def clean_engine(tmp_path):
    """Create a fresh BTOMEngine with a temp state file. No carryover."""
    state_path = tmp_path / "test_state.json"
    engine = BTOMEngine(state_path=state_path, hypotheses=dict(DEFAULT_HYPOTHESES))
    engine.reset()
    return engine


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires live LLM connection (LM Studio)")
    config.addinivalue_line("markers", "integration: integration tests through full pipeline")

"""
Simplified pytest configuration for end-to-end tests only.
Avoids importing problematic modules.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_test_dir():
    """Create a temporary test directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_cache_dir():
    """Get path to test cache directory."""
    return Path("data/cache/test_model_states")


@pytest.fixture
def test_weights_dir():
    """Get path to test weights directory."""
    return Path("tests/data/weights/analyze")
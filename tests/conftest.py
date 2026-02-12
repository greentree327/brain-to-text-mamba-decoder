"""
Pytest configuration and fixtures for Brain-to-Text Decoder tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def dummy_neural_data():
    """Create dummy neural data for testing."""
    batch_size = 2
    seq_len = 100
    neural_dim = 513
    
    return torch.randn(batch_size, seq_len, neural_dim, dtype=torch.float32)


@pytest.fixture
def dummy_day_idx():
    """Create dummy day indices."""
    return torch.tensor([0, 1])


@pytest.fixture
def model_config():
    """Standard model configuration for testing."""
    return {
        'neural_dim': 513,
        'n_units': 256,
        'n_days': 5,
        'n_classes': 40,
        'n_layers': 2,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session", autouse=True)
def check_dependencies():
    """Check that all required dependencies are installed."""
    try:
        import mamba_ssm
        print("✓ mamba-ssm available")
    except ImportError:
        print("⚠ mamba-ssm not available (expected for CPU-only environments)")

"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import os
from pathlib import Path
from hypothesis import settings, Verbosity

# Configure hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=200, verbosity=Verbosity.verbose)
settings.load_profile("default")

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_clinical_text():
    """Load sample clinical text with abnormal values."""
    with open(FIXTURES_DIR / "sample_clinical_text.txt", "r") as f:
        return f.read()


@pytest.fixture
def sample_clinical_text_normal():
    """Load sample clinical text with normal values."""
    with open(FIXTURES_DIR / "sample_clinical_text_normal.txt", "r") as f:
        return f.read()


@pytest.fixture
def sample_clinical_text_mixed():
    """Load sample clinical text with mixed values."""
    with open(FIXTURES_DIR / "sample_clinical_text_mixed.txt", "r") as f:
        return f.read()


@pytest.fixture
def sample_lab_report_path(fixtures_dir):
    """Return path to sample lab report PDF (to be created)."""
    return fixtures_dir / "sample_lab_report.pdf"


@pytest.fixture
def sample_xray_path(fixtures_dir):
    """Return path to sample X-ray image (to be created)."""
    return fixtures_dir / "sample_xray.jpg"


@pytest.fixture
def sample_image_png_path(fixtures_dir):
    """Return path to sample PNG image (to be created)."""
    return fixtures_dir / "sample_image.png"

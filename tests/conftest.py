"""
Pytest configuration and fixtures.
"""
import sys
from pathlib import Path

import pytest

# Add src to path so imports work
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def pytest_collection_modifyitems(config, items):
    """Run post_processing tests before coil_optimization to avoid matplotlib state corruption.

    coil_optimization 3D plots can leave matplotlib in a bad state that causes
    'Figure' object has no attribute 'items' in post_processing plot tests.
    Running post_processing first avoids this.
    """
    post_processing = [i for i in items if "post_processing" in i.nodeid]
    others = [i for i in items if i not in post_processing]
    items[:] = post_processing + others


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures before and after each test to prevent state leakage.

    Without this, figures from earlier tests (e.g. coil_optimization 3D plots)
    can corrupt matplotlib's internal state and cause 'Figure' object has no
    attribute 'items' in later tests (e.g. post_processing plot tests).
    """
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


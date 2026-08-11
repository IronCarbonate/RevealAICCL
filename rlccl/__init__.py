"""Evolved CCL - clean and modular collective scheduling components."""

__version__ = '1.0.0'

from .config import get_config, AVAILABLE_TOPOLOGIES


def __getattr__(name):
    """Load PyTorch model classes lazily so traffic tooling stays NumPy-only."""
    if name in {"SlotLevelPolicy", "ECDUGNNLayer"}:
        from .models import ECDUGNNLayer, SlotLevelPolicy

        return {"SlotLevelPolicy": SlotLevelPolicy, "ECDUGNNLayer": ECDUGNNLayer}[name]
    raise AttributeError(name)

__all__ = [
    'SlotLevelPolicy',
    'ECDUGNNLayer',
    'get_config',
    'AVAILABLE_TOPOLOGIES',
]

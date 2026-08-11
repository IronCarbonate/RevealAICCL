"""Configuration management for Evolved CCL."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STRATEGY_DIR = OUTPUT_DIR / "strategies"
XML_DIR = OUTPUT_DIR / "xml"

# Create directories if they don't exist
for dir_path in [CHECKPOINT_DIR, OUTPUT_DIR, STRATEGY_DIR, XML_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Default training configuration
DEFAULT_TRAIN_CONFIG = {
    'epochs': 30,
    'ppo_epochs': 10,
    'batch_target': 500,
    'lr': 3e-4,
    'hidden_dim': 128,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'mini_batch_size': 16,
    'max_grad_norm': 0.5,
}

# Available topologies
AVAILABLE_TOPOLOGIES = [
    'Rear4GPU',
    'Rear8GPU_NoSwitch_Test',
    'Heterogeneous_12GPU',
    'Heterogeneous_16GPU_3Server',
    'Heterogeneous_6GPU_Ring',
]

# Traffic patterns
TRAFFIC_PATTERNS = {
    'AllGather': 0.3,
    'All-to-All-V': 0.5,
    'AllToAll': 0.2,
}

def get_config():
    """Get default configuration."""
    return DEFAULT_TRAIN_CONFIG.copy()

def get_topology_path(topology_name):
    """Get path to topology JSON file."""
    return DATA_DIR / f"{topology_name}.json"

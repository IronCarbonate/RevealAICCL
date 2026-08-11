"""Training module for PPO-based optimization."""

from .buffer import SlotBuffer
from .ppo_trainer import train_epoch, evaluate_model, compute_gae_advantages
from .sequence_sampler import SequenceDatasetConfig, build_sequence_problems

__all__ = [
    'SlotBuffer',
    'train_epoch',
    'evaluate_model',
    'compute_gae_advantages',
    'SequenceDatasetConfig',
    'build_sequence_problems',
]

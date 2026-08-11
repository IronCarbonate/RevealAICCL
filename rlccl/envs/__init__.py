"""Environment and problem instance definitions."""

from .problem import ProblemInstance, TopologyInfo
from .evaluator import evaluate_schedule, load_topology_info, generate_train_test_split
from .sequence_env import TrafficSequenceRunner


def __getattr__(name):
    """Load the PyTorch decoder only when a caller requests it."""
    if name in {'SlotDecoder', 'solve_with_model'}:
        from .decoder import SlotDecoder, solve_with_model

        return {'SlotDecoder': SlotDecoder, 'solve_with_model': solve_with_model}[name]
    raise AttributeError(name)

__all__ = [
    'ProblemInstance',
    'TopologyInfo', 
    'evaluate_schedule',
    'load_topology_info',
    'generate_train_test_split',
    'SlotDecoder',
    'solve_with_model',
    'TrafficSequenceRunner',
]

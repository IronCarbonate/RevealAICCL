"""Traffic-sequence data structures and history-only moment utilities."""

from .matrix_utils import (
    scenario_to_traffic_matrix,
    traffic_matrix_to_scenario,
    validate_traffic_matrix,
)
from .context_views import mean_only_context, zero_moment_context
from .moment_estimator import SlidingMomentEstimator
from .moment_validation import (
    compute_window_moments,
    relative_l2_error,
    validate_sequence_moment_bounds,
)
from .process_generator import TrafficProcessConfig, generate_traffic_sequence
from .long_horizon_generator import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    SPATIAL_MODES,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
    generate_same_moment_group,
)
from .types import MomentBounds, MomentContext, TrafficSequence

__all__ = [
    "MomentBounds",
    "MomentContext",
    "LongHorizonTrafficConfig",
    "LONG_HORIZON_FAMILIES",
    "SAME_MOMENT_VARIANTS",
    "SPATIAL_MODES",
    "TrafficProcessConfig",
    "TrafficSequence",
    "SlidingMomentEstimator",
    "mean_only_context",
    "zero_moment_context",
    "compute_window_moments",
    "generate_traffic_sequence",
    "generate_long_horizon_sequence",
    "generate_same_moment_group",
    "relative_l2_error",
    "scenario_to_traffic_matrix",
    "traffic_matrix_to_scenario",
    "validate_sequence_moment_bounds",
    "validate_traffic_matrix",
]

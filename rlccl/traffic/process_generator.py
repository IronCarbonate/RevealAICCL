"""Moment-bounded, non-Gaussian All-to-All-V traffic process generator."""

from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from .moment_validation import validate_sequence_moment_bounds
from .types import MomentBounds, TrafficSequence


@dataclass
class TrafficProcessConfig:
    num_nodes: int
    sequence_length: int
    window_size: int
    mean_level: float
    std_level: float
    max_entry: int
    epsilon_mean: float
    epsilon_var: float
    family: str
    seed: int
    max_generation_attempts: int = 100
    mean_ref: np.ndarray | None = None
    var_ref: np.ndarray | None = None
    topology_name: str = "unknown"
    validation_stride: int = 1
    allowed_violation_fraction: float = 0.0

    FAMILIES: ClassVar[tuple[str, ...]] = (
        "smooth_ar",
        "alternating_burst",
        "moving_hotspot",
        "sparse_switching",
        "bimodal",
        "heavy_tail_clipped",
    )

    def __post_init__(self) -> None:
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if self.window_size <= 1 or self.sequence_length < self.window_size:
            raise ValueError("sequence_length must be >= window_size > 1")
        if self.mean_level < 0 or self.std_level < 0:
            raise ValueError("mean_level and std_level must be nonnegative")
        if self.max_entry <= 0:
            raise ValueError("max_entry must be positive")
        if self.family not in self.FAMILIES:
            raise ValueError(
                f"Unknown traffic family {self.family!r}; expected one of {self.FAMILIES}"
            )
        if self.max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")
        if self.validation_stride <= 0:
            raise ValueError("validation_stride must be positive")
        if not 0.0 <= self.allowed_violation_fraction <= 1.0:
            raise ValueError("allowed_violation_fraction must be in [0, 1]")


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    std = centered.std(ddof=0)
    if std < 1e-12:
        raise ValueError("Family profile must have nonzero variance")
    return centered / std


def _family_profile(family: str, window_size: int) -> np.ndarray:
    """Return a period with family-specific shape and standardized moments."""
    t = np.arange(window_size, dtype=np.float64)
    if family == "smooth_ar":
        raw = np.sin(2.0 * np.pi * t / window_size)
        raw += 0.30 * np.sin(4.0 * np.pi * t / window_size + 0.4)
    elif family == "alternating_burst":
        raw = np.where((t.astype(int) % 2) == 0, -1.0, 1.0)
        raw += 0.15 * np.where((t.astype(int) % 4) < 2, -1.0, 1.0)
    elif family == "moving_hotspot":
        # A compact smooth pulse; destination-dependent phase shifts make it move.
        distance = np.minimum(t, window_size - t)
        raw = np.exp(-0.5 * (distance / max(window_size / 8.0, 1.0)) ** 2)
    elif family == "sparse_switching":
        active = max(1, window_size // 4)
        # Sparse active positions ride on a small background ripple.  The ripple
        # gives integer calibration enough degrees of freedom to preserve tight
        # target moments after rounding, while the high positions still switch.
        raw = 0.20 * np.sin(2.0 * np.pi * t / window_size)
        raw[:active] += 1.5
    elif family == "bimodal":
        raw = np.full(window_size, -1.0, dtype=np.float64)
        raw[window_size // 2 :] = 1.0
    elif family == "heavy_tail_clipped":
        raw = np.zeros(window_size, dtype=np.float64)
        raw[0] = max(4.0, np.sqrt(window_size))
        if window_size >= 8:
            raw[window_size // 2] = 1.0
    else:  # guarded by TrafficProcessConfig
        raise ValueError(f"Unsupported family: {family}")
    return _standardize(raw)


def _calibrate_integer_period(
    profile: np.ndarray,
    target_mean: float,
    target_var: float,
    max_entry: int,
) -> np.ndarray:
    """Search a small affine grid, then clip/round on the final integer period."""
    target_std = float(np.sqrt(max(target_var, 0.0)))
    best_values = None
    best_score = float("inf")

    mean_offsets = np.linspace(-0.75, 0.75, 13)
    scale_factors = np.linspace(0.50, 1.50, 21) if target_std > 0 else np.array([0.0])
    for offset in mean_offsets:
        for factor in scale_factors:
            values = np.rint(
                np.clip(target_mean + offset + target_std * factor * profile, 0, max_entry)
            ).astype(np.int64)
            actual_mean = float(values.mean())
            actual_var = float(values.var(ddof=0))
            mean_scale = max(abs(target_mean), 1.0)
            var_scale = max(abs(target_var), 1.0)
            score = ((actual_mean - target_mean) / mean_scale) ** 2
            score += ((actual_var - target_var) / var_scale) ** 2
            if score < best_score:
                best_score = score
                best_values = values
    assert best_values is not None
    return best_values


def _references(config: TrafficProcessConfig) -> tuple[np.ndarray, np.ndarray]:
    shape = (config.num_nodes, config.num_nodes)
    if config.mean_ref is None:
        mean_ref = np.full(shape, config.mean_level, dtype=np.float64)
        np.fill_diagonal(mean_ref, 0.0)
    else:
        mean_ref = np.asarray(config.mean_ref, dtype=np.float64).copy()
    if config.var_ref is None:
        var_ref = np.full(shape, config.std_level**2, dtype=np.float64)
        np.fill_diagonal(var_ref, 0.0)
    else:
        var_ref = np.asarray(config.var_ref, dtype=np.float64).copy()

    if mean_ref.shape != shape or var_ref.shape != shape:
        raise ValueError(f"mean_ref and var_ref must both have shape {shape}")
    if not np.all(np.isfinite(mean_ref)) or not np.all(np.isfinite(var_ref)):
        raise ValueError("Reference moments must be finite")
    if np.any(mean_ref < 0) or np.any(var_ref < 0):
        raise ValueError("Reference mean and variance must be nonnegative")
    if np.any(np.diag(mean_ref) != 0) or np.any(np.diag(var_ref) != 0):
        raise ValueError("Reference moment diagonals must be zero")
    return mean_ref, var_ref


def _phase_for_pair(
    family: str,
    src: int,
    dst: int,
    num_nodes: int,
    window_size: int,
    rng: np.random.Generator,
    sequence_phase: int,
) -> int:
    if family == "moving_hotspot":
        return (dst * window_size // num_nodes + sequence_phase) % window_size
    if family == "sparse_switching":
        return (src * num_nodes + dst + sequence_phase) % window_size
    if family == "alternating_burst":
        return (src + dst + sequence_phase) % window_size
    if family == "heavy_tail_clipped":
        return (src * 3 + dst * 5 + sequence_phase) % window_size
    return int(rng.integers(0, window_size))


def _generate_candidate(
    config: TrafficProcessConfig,
    mean_ref: np.ndarray,
    var_ref: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    profile = _family_profile(config.family, config.window_size)
    period = np.zeros(
        (config.window_size, config.num_nodes, config.num_nodes), dtype=np.int64
    )
    # Make the family-wide phase seed-addressable (adjacent seeds differ even
    # for two-cycle alternating profiles); pair-specific random phases below
    # still use the local Generator where appropriate.
    sequence_phase = int(config.seed % config.window_size)
    calibrated_cache: dict[tuple[float, float], np.ndarray] = {}
    for src in range(config.num_nodes):
        for dst in range(config.num_nodes):
            if src == dst:
                continue
            key = (float(mean_ref[src, dst]), float(var_ref[src, dst]))
            if key not in calibrated_cache:
                calibrated_cache[key] = _calibrate_integer_period(
                    profile, key[0], key[1], config.max_entry
                )
            values = calibrated_cache[key]
            phase = _phase_for_pair(
                config.family,
                src,
                dst,
                config.num_nodes,
                config.window_size,
                rng,
                sequence_phase,
            )
            period[:, src, dst] = np.roll(values, phase)

    repeats = int(np.ceil(config.sequence_length / config.window_size))
    sequence = np.tile(period, (repeats, 1, 1))[: config.sequence_length]
    return [matrix.copy() for matrix in sequence]


def _config_metadata(config: TrafficProcessConfig) -> dict[str, object]:
    values = asdict(config)
    for name in ("mean_ref", "var_ref"):
        if values[name] is not None:
            values[name] = np.asarray(values[name]).tolist()
    return values


def generate_traffic_sequence(config: TrafficProcessConfig) -> TrafficSequence:
    """Generate and strictly validate a final integer traffic sequence."""
    mean_ref, var_ref = _references(config)
    bounds = MomentBounds(config.epsilon_mean, config.epsilon_var)
    rng = np.random.default_rng(config.seed)
    last_diagnostics = None

    for attempt in range(1, config.max_generation_attempts + 1):
        matrices = _generate_candidate(config, mean_ref, var_ref, rng)
        sequence = TrafficSequence(
            sequence_id=f"{config.family}-seed{config.seed}",
            topology_name=config.topology_name,
            family=config.family,
            seed=config.seed,
            matrices=matrices,
            mean_ref=mean_ref,
            var_ref=var_ref,
            bounds=bounds,
            metadata={
                "schema_version": 1,
                "window_size": config.window_size,
                "validation_stride": config.validation_stride,
                "allowed_violation_fraction": config.allowed_violation_fraction,
                "generation_attempt": attempt,
                "generator_config": _config_metadata(config),
            },
        )
        diagnostics = validate_sequence_moment_bounds(sequence)
        if diagnostics["passed"]:
            sequence.metadata["validation"] = {
                key: value for key, value in diagnostics.items() if key != "windows"
            }
            return sequence
        last_diagnostics = diagnostics

    assert last_diagnostics is not None
    raise RuntimeError(
        "Unable to generate a moment-bounded integer traffic sequence after "
        f"{config.max_generation_attempts} attempts for family={config.family!r}; "
        f"max_mean_error={last_diagnostics['max_mean_error']:.6f}, "
        f"max_var_error={last_diagnostics['max_var_error']:.6f}, "
        f"bounds=({config.epsilon_mean}, {config.epsilon_var})"
    )

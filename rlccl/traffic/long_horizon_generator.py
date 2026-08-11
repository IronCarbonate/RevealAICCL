"""Long-horizon stochastic traffic processes for Phase B.

This module is deliberately independent from ``process_generator``.  The six
legacy families remain unchanged.  Long-horizon traffic is constructed as a
scalar total-load process ``S_t`` and a normalized spatial process ``P_t``, then
integerized with a capacity-aware largest-deficit allocator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np

from .matrix_utils import validate_traffic_matrix
from .types import MomentBounds, TrafficSequence


LONG_HORIZON_FAMILIES = (
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
)

SAME_MOMENT_VARIANTS = (
    "smooth",
    "random_switching",
    "long_regime",
    "shock_recovery",
)

SPATIAL_MODES = (
    "balanced",
    "single_source_hotspot",
    "single_destination_hotspot",
    "dual_hotspot",
    "sparse_support",
    "cross_group_concentration",
    "hotspot_random_walk",
)


@dataclass(frozen=True)
class LongHorizonTrafficConfig:
    num_nodes: int = 4
    sequence_length: int = 1024
    family: str = "regime_switching_long"
    seed: int = 42
    mean_level: float = 2.0
    std_level: float = 1.5
    max_entry: int = 8
    base_level: float | None = None
    ar_coefficient: float = 0.80
    short_noise_scale: float = 0.06
    regime_levels: tuple[float, float, float] = (0.75, 1.00, 1.25)
    regime_dwell_range: tuple[int, int] = (32, 512)
    shock_probability: float = 0.003
    shock_magnitude: float = 1.50
    shock_duration_range: tuple[int, int] = (4, 16)
    recovery_rate: float = 0.94
    minimum_total: float = 4.0
    maximum_total: float | None = None
    short_window: int = 16
    medium_window: int = 128
    long_window: int = 512
    medium_epsilon_mean: float = 0.60
    medium_epsilon_var: float = 6.00
    medium_allowed_violation_fraction: float = 0.20
    medium_matrix_epsilon_mean: float = 1.10
    medium_matrix_epsilon_var: float = 3.50
    long_epsilon_mean: float = 0.30
    long_epsilon_var: float = 3.00
    long_allowed_violation_fraction: float = 0.05
    long_matrix_epsilon_mean: float = 0.70
    long_matrix_epsilon_var: float = 2.50
    spatial_mode: str | None = None
    hotspot_dwell_range: tuple[int, int] = (8, 96)
    hotspot_strength_range: tuple[float, float] = (2.5, 5.0)
    source_groups: tuple[int, ...] | None = None
    dynamics_variant: str | None = None
    calibration_candidates: int = 3
    topology_name: str = "Rear4GPU"

    FAMILIES: ClassVar[tuple[str, ...]] = LONG_HORIZON_FAMILIES
    SPATIAL_MODES: ClassVar[tuple[str, ...]] = SPATIAL_MODES
    SAME_MOMENT_VARIANTS: ClassVar[tuple[str, ...]] = SAME_MOMENT_VARIANTS

    def __post_init__(self) -> None:
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be at least two")
        if self.sequence_length < self.short_window or self.short_window <= 1:
            raise ValueError("sequence_length must be >= short_window > 1")
        if self.family not in LONG_HORIZON_FAMILIES:
            raise ValueError(f"Unsupported long-horizon family: {self.family}")
        if self.mean_level <= 0 or self.std_level < 0 or self.max_entry <= 0:
            raise ValueError("mean_level/max_entry must be positive and std_level nonnegative")
        if not -0.999 < self.ar_coefficient < 0.999:
            raise ValueError("ar_coefficient must be in (-0.999, 0.999)")
        if not 0.0 <= self.shock_probability <= 1.0:
            raise ValueError("shock_probability must be in [0, 1]")
        if not 0.0 < self.recovery_rate < 1.0:
            raise ValueError("recovery_rate must be in (0, 1)")
        if self.spatial_mode is not None and self.spatial_mode not in SPATIAL_MODES:
            raise ValueError(f"Unsupported spatial_mode: {self.spatial_mode}")
        if self.dynamics_variant is not None and self.dynamics_variant not in SAME_MOMENT_VARIANTS:
            raise ValueError(f"Unsupported dynamics_variant: {self.dynamics_variant}")
        for name, bounds in (
            ("regime_dwell_range", self.regime_dwell_range),
            ("shock_duration_range", self.shock_duration_range),
            ("hotspot_dwell_range", self.hotspot_dwell_range),
        ):
            if len(bounds) != 2 or bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError(f"{name} must be a positive ordered pair")
        for name, fraction in (
            ("medium_allowed_violation_fraction", self.medium_allowed_violation_fraction),
            ("long_allowed_violation_fraction", self.long_allowed_violation_fraction),
        ):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.calibration_candidates <= 0:
            raise ValueError("calibration_candidates must be positive")
        if self.source_groups is not None and len(self.source_groups) != self.num_nodes:
            raise ValueError("source_groups must contain one group id per node")


def _base_total(config: LongHorizonTrafficConfig) -> float:
    off_diagonal = config.num_nodes * (config.num_nodes - 1)
    return float(config.base_level if config.base_level is not None else config.mean_level * off_diagonal)


def _maximum_total(config: LongHorizonTrafficConfig) -> float:
    physical = float(config.max_entry * config.num_nodes * (config.num_nodes - 1))
    return min(physical, float(config.maximum_total)) if config.maximum_total is not None else physical


def _ar_noise(length: int, phi: float, sigma: np.ndarray | float, rng: np.random.Generator) -> np.ndarray:
    scale = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (length,))
    values = np.zeros(length, dtype=np.float64)
    innovation_factor = np.sqrt(max(1.0 - phi * phi, 1e-6))
    for index in range(1, length):
        values[index] = phi * values[index - 1] + rng.normal(0.0, scale[index] * innovation_factor)
    return values


def _dwell_schedule(
    length: int,
    labels: tuple[Any, ...],
    dwell_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.empty(length, dtype=object)
    records: list[dict[str, Any]] = []
    position = 0
    state_index = int(rng.integers(0, len(labels)))
    while position < length:
        dwell = int(rng.integers(dwell_range[0], dwell_range[1] + 1))
        end = min(length, position + dwell)
        label = labels[state_index]
        values[position:end] = label
        records.append({"state": label, "start": position, "end": end, "length": end - position})
        alternatives = [index for index in range(len(labels)) if index != state_index]
        state_index = int(rng.choice(alternatives))
        position = end
    return values, records


def _same_moment_latent(
    length: int,
    variant: str,
    config: LongHorizonTrafficConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"dynamics_variant": variant}
    if variant == "smooth":
        latent = _ar_noise(length, 0.975, 1.0, rng)
    elif variant == "random_switching":
        states, records = _dwell_schedule(length, (-1.0, 1.0), (4, 32), rng)
        latent = states.astype(np.float64) + _ar_noise(length, 0.4, 0.20, rng)
        metadata["variant_dwell_records"] = records
    elif variant == "long_regime":
        states, records = _dwell_schedule(length, (-1.0, 0.0, 1.0), (64, 256), rng)
        latent = states.astype(np.float64) + _ar_noise(length, 0.85, 0.15, rng)
        metadata["variant_dwell_records"] = records
    elif variant == "shock_recovery":
        latent = _ar_noise(length, 0.75, 0.18, rng)
        flags = np.zeros(length, dtype=np.int8)
        starts = sorted(set(int(value) for value in rng.integers(32, max(33, length - 32), size=max(1, length // 512))))
        for start in starts:
            amplitude = float(rng.uniform(3.0, 5.0))
            duration = int(rng.integers(4, 13))
            for offset in range(duration):
                if start + offset < length:
                    latent[start + offset] += amplitude
                    flags[start + offset] = 1
            residual = amplitude
            cursor = start + duration
            while cursor < length and residual > 0.05:
                residual *= config.recovery_rate
                latent[cursor] += residual
                cursor += 1
        metadata["variant_shock_flags"] = flags.astype(int).tolist()
    else:
        raise ValueError(f"Unknown same-moment variant: {variant}")
    centered = latent - latent.mean()
    std = float(centered.std(ddof=0))
    if std <= 1e-12:
        raise RuntimeError("same-moment latent process became constant")
    return centered / std, metadata


def _total_process(
    config: LongHorizonTrafficConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    length = config.sequence_length
    base = _base_total(config)
    metadata: dict[str, Any] = {}
    latent_regime = np.full(length, "normal", dtype=object)
    shock_flags = np.zeros(length, dtype=np.int8)
    shock_component = np.zeros(length, dtype=np.float64)
    volatility_state = np.full(length, "calm", dtype=object)

    if config.family == "regime_switching_long":
        latent_regime, records = _dwell_schedule(
            length, ("low", "normal", "high"), config.regime_dwell_range, rng
        )
        level_map = dict(zip(("low", "normal", "high"), config.regime_levels))
        baseline = np.asarray([base * level_map[state] for state in latent_regime], dtype=np.float64)
        total = baseline + _ar_noise(length, config.ar_coefficient, base * config.short_noise_scale, rng)
        metadata["regime_dwell_records"] = records
    elif config.family == "stochastic_volatility":
        volatility_state, records = _dwell_schedule(length, ("calm", "volatile"), (32, 256), rng)
        sigma = np.where(volatility_state == "calm", base * 0.025, base * 0.18)
        total = base + _ar_noise(length, config.ar_coefficient, sigma, rng)
        latent_regime = volatility_state.copy()
        metadata["volatility_dwell_records"] = records
    elif config.family == "rare_shock_recovery":
        total = base + _ar_noise(length, config.ar_coefficient, base * config.short_noise_scale, rng)
        starts: list[dict[str, Any]] = []
        cursor = 1
        while cursor < length:
            if rng.random() < config.shock_probability:
                duration = int(rng.integers(config.shock_duration_range[0], config.shock_duration_range[1] + 1))
                amplitude = base * config.shock_magnitude * float(rng.uniform(0.8, 1.2))
                end = min(length, cursor + duration)
                shock_component[cursor:end] += amplitude
                shock_flags[cursor:end] = 1
                residual = amplitude
                recovery_end = end
                while recovery_end < length and residual > base * 0.02:
                    residual *= config.recovery_rate
                    shock_component[recovery_end] += residual
                    recovery_end += 1
                starts.append(
                    {"start": cursor, "end": end, "duration": end - cursor, "amplitude": amplitude, "recovery_end": recovery_end}
                )
                cursor = recovery_end
            else:
                cursor += 1
        if not starts:
            cursor = int(rng.integers(max(1, length // 4), max(2, 3 * length // 4)))
            duration = min(config.shock_duration_range[1], max(1, length - cursor))
            amplitude = base * config.shock_magnitude
            end = min(length, cursor + duration)
            shock_component[cursor:end] += amplitude
            shock_flags[cursor:end] = 1
            residual = amplitude
            recovery_end = end
            while recovery_end < length and residual > base * 0.02:
                residual *= config.recovery_rate
                shock_component[recovery_end] += residual
                recovery_end += 1
            starts.append({"start": cursor, "end": end, "duration": end - cursor, "amplitude": amplitude, "recovery_end": recovery_end})
        total += shock_component
        latent_regime = np.where(shock_flags > 0, "shock", np.where(shock_component > 0, "recovery", "normal"))
        metadata["shock_records"] = starts
    elif config.family == "hotspot_random_walk":
        total = base + _ar_noise(length, config.ar_coefficient, base * 0.04, rng)
    elif config.family == "same_moments_different_dynamics":
        variant = config.dynamics_variant or SAME_MOMENT_VARIANTS[config.seed % len(SAME_MOMENT_VARIANTS)]
        latent, variant_metadata = _same_moment_latent(length, variant, config, rng)
        target_total_std = config.std_level * np.sqrt(config.num_nodes * (config.num_nodes - 1))
        total = base + target_total_std * latent
        metadata.update(variant_metadata)
        latent_regime = np.full(length, variant, dtype=object)
        if "variant_shock_flags" in variant_metadata:
            shock_flags = np.asarray(variant_metadata["variant_shock_flags"], dtype=np.int8)
    else:
        raise ValueError(config.family)

    # Global shift is a finite calibration step: it preserves dynamics and does
    # not reject rare events merely because short windows are unusual.
    total = total + (base - float(total.mean()))
    total = np.clip(total, config.minimum_total, _maximum_total(config))
    metadata.update(
        {
            "latent_regime": latent_regime.tolist(),
            "shock_flags": shock_flags.astype(int).tolist(),
            "shock_component": shock_component.tolist(),
            "volatility_state": volatility_state.tolist(),
            "target_total_mean": base,
        }
    )
    return total, metadata


def _random_walk_hotspots(
    config: LongHorizonTrafficConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    length, nodes = config.sequence_length, config.num_nodes
    sources = np.empty(length, dtype=np.int64)
    destinations = np.empty(length, dtype=np.int64)
    strengths = np.empty(length, dtype=np.float64)
    records: list[dict[str, Any]] = []
    source = int(rng.integers(nodes))
    destination = int(rng.integers(nodes))
    position = 0
    while position < length:
        dwell = int(rng.integers(config.hotspot_dwell_range[0], config.hotspot_dwell_range[1] + 1))
        end = min(length, position + dwell)
        strength = float(rng.uniform(*config.hotspot_strength_range))
        sources[position:end] = source
        destinations[position:end] = destination
        strengths[position:end] = strength
        records.append(
            {"start": position, "end": end, "length": end - position, "source": source, "destination": destination, "strength": strength}
        )
        source = (source + int(rng.choice((-1, 0, 1)))) % nodes
        destination = (destination + int(rng.choice((-1, 1)))) % nodes
        position = end
    return sources, destinations, strengths, records


def _spatial_process(
    config: LongHorizonTrafficConfig,
    total_metadata: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    length, nodes = config.sequence_length, config.num_nodes
    off_diagonal = ~np.eye(nodes, dtype=bool)
    probabilities = np.zeros((length, nodes, nodes), dtype=np.float64)
    sources, destinations, strengths, hotspot_records = _random_walk_hotspots(config, rng)
    groups = np.asarray(
        config.source_groups if config.source_groups is not None else tuple(index // max(1, nodes // 2) for index in range(nodes)),
        dtype=np.int64,
    )
    noise_state = np.zeros((nodes, nodes), dtype=np.float64)
    sparse_mask = off_diagonal.copy()
    sparse_refresh = 0
    regimes = np.asarray(total_metadata["latent_regime"], dtype=object)
    shock_component = np.asarray(total_metadata["shock_component"], dtype=np.float64)
    volatility = np.asarray(total_metadata["volatility_state"], dtype=object)
    modes: list[str] = []

    for step in range(length):
        if config.spatial_mode is not None:
            mode = config.spatial_mode
        elif config.family == "regime_switching_long":
            mode = {"low": "balanced", "normal": "single_source_hotspot", "high": "dual_hotspot"}[str(regimes[step])]
        elif config.family == "stochastic_volatility":
            mode = "sparse_support" if volatility[step] == "volatile" else "balanced"
        elif config.family == "rare_shock_recovery":
            mode = "dual_hotspot" if shock_component[step] > 0 else "balanced"
        elif config.family == "hotspot_random_walk":
            mode = "hotspot_random_walk"
        else:
            mode = "balanced"
        modes.append(mode)

        spatial_sigma = 0.05 if mode == "balanced" else 0.10
        noise_state = 0.92 * noise_state + rng.normal(0.0, spatial_sigma, size=(nodes, nodes))
        weights = np.exp(np.clip(noise_state, -1.5, 1.5))
        weights[~off_diagonal] = 0.0
        strength = strengths[step]
        source, destination = int(sources[step]), int(destinations[step])
        if mode == "single_source_hotspot":
            weights[source, :] *= strength
        elif mode == "single_destination_hotspot":
            weights[:, destination] *= strength
        elif mode == "dual_hotspot":
            weights[source, :] *= strength
            weights[:, destination] *= strength
        elif mode == "sparse_support":
            if step >= sparse_refresh:
                available = np.argwhere(off_diagonal)
                selected = rng.choice(len(available), size=max(nodes, len(available) // 3), replace=False)
                sparse_mask = np.zeros((nodes, nodes), dtype=bool)
                sparse_mask[tuple(available[selected].T)] = True
                sparse_refresh = step + int(rng.integers(8, 65))
            weights[~sparse_mask] *= 0.02
        elif mode == "cross_group_concentration":
            cross = groups[:, None] != groups[None, :]
            weights[cross & off_diagonal] *= strength
        elif mode == "hotspot_random_walk":
            weights[:, destination] *= strength
            weights[source, :] *= np.sqrt(strength)
        elif mode != "balanced":
            raise ValueError(mode)
        weights[~off_diagonal] = 0.0
        probabilities[step] = weights / weights.sum()

    return probabilities, {
        "spatial_mode": modes,
        "hotspot_source": sources.astype(int).tolist(),
        "hotspot_destination": destinations.astype(int).tolist(),
        "hotspot_strength": strengths.tolist(),
        "hotspot_dwell_records": hotspot_records,
        "source_groups": groups.astype(int).tolist(),
    }


def _allocate_integer_matrix(
    total: float,
    probabilities: np.ndarray,
    max_entry: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = probabilities.shape[0]
    off_diagonal = ~np.eye(nodes, dtype=bool)
    target = int(np.clip(np.rint(total), 0, max_entry * int(off_diagonal.sum())))
    raw = float(total) * probabilities
    clipped = np.clip(raw, 0.0, max_entry)
    result = np.zeros((nodes, nodes), dtype=np.int64)
    flat_probabilities = probabilities.ravel()
    flat_result = result.ravel()
    allowed = off_diagonal.ravel()
    jitter = rng.uniform(0.0, 1e-9, size=flat_result.size)
    for _ in range(target):
        available = allowed & (flat_result < max_entry)
        deficits = target * flat_probabilities - flat_result + jitter
        deficits[~available] = -np.inf
        index = int(np.argmax(deficits))
        if not np.isfinite(deficits[index]):
            break
        flat_result[index] += 1
    np.fill_diagonal(result, 0)
    return result, raw, clipped


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "variance": float(array.var(ddof=0)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def _rolling_scalar_constraint(
    values: np.ndarray,
    window: int,
    reference_mean: float,
    reference_variance: float,
    epsilon_mean: float | None,
    epsilon_variance: float | None,
    allowed_fraction: float | None,
) -> dict[str, Any]:
    if window > len(values):
        return {"available": False, "window_size": window, "num_windows": 0}
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    cumulative_sq = np.r_[0.0, np.cumsum(values * values, dtype=np.float64)]
    sums = cumulative[window:] - cumulative[:-window]
    sums_sq = cumulative_sq[window:] - cumulative_sq[:-window]
    means = sums / window
    variances = np.maximum(sums_sq / window - means * means, 0.0)
    mean_error = np.abs(means - reference_mean) / max(abs(reference_mean), 1.0)
    variance_error = np.abs(variances - reference_variance) / max(abs(reference_variance), 1.0)
    result: dict[str, Any] = {
        "available": True,
        "window_size": window,
        "num_windows": int(len(means)),
        "mean_error_mean": float(mean_error.mean()),
        "mean_error_max": float(mean_error.max()),
        "variance_error_mean": float(variance_error.mean()),
        "variance_error_max": float(variance_error.max()),
        "epsilon_mean": epsilon_mean,
        "epsilon_variance": epsilon_variance,
        "allowed_violation_fraction": allowed_fraction,
    }
    if epsilon_mean is None or epsilon_variance is None or allowed_fraction is None:
        result.update({"hard_or_soft_bound_applied": False, "violation_fraction": None, "passed": True})
    else:
        violations = (mean_error > epsilon_mean) | (variance_error > epsilon_variance)
        fraction = float(violations.mean())
        result.update(
            {
                "hard_or_soft_bound_applied": True,
                "violation_fraction": fraction,
                "violation_magnitude": {
                    "mean_excess_max": float(np.maximum(mean_error - epsilon_mean, 0.0).max()),
                    "variance_excess_max": float(np.maximum(variance_error - epsilon_variance, 0.0).max()),
                },
                "passed": fraction <= allowed_fraction,
            }
        )
    return result


def _rolling_matrix_constraint(
    stack: np.ndarray,
    window: int,
    mean_ref: np.ndarray,
    var_ref: np.ndarray,
    epsilon_mean: float | None,
    epsilon_variance: float | None,
    allowed_fraction: float | None,
) -> dict[str, Any]:
    if window > len(stack):
        return {"available": False, "window_size": window, "num_windows": 0}
    values = stack.astype(np.float64, copy=False)
    cumulative = np.concatenate(
        [np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)], axis=0
    )
    cumulative_sq = np.concatenate(
        [np.zeros((1,) + values.shape[1:]), np.cumsum(values * values, axis=0)], axis=0
    )
    sums = cumulative[window:] - cumulative[:-window]
    sums_sq = cumulative_sq[window:] - cumulative_sq[:-window]
    means = sums / float(window)
    variances = np.maximum(sums_sq / float(window) - means * means, 0.0)
    axes = tuple(range(1, means.ndim))
    mean_errors = np.sqrt(np.sum((means - mean_ref) ** 2, axis=axes)) / (
        float(np.linalg.norm(mean_ref)) + 1e-8
    )
    variance_errors = np.sqrt(np.sum((variances - var_ref) ** 2, axis=axes)) / (
        float(np.linalg.norm(var_ref)) + 1e-8
    )
    result: dict[str, Any] = {
        "available": True,
        "window_size": window,
        "num_windows": int(len(means)),
        "mean_error_mean": float(mean_errors.mean()),
        "mean_error_max": float(mean_errors.max()),
        "variance_error_mean": float(variance_errors.mean()),
        "variance_error_max": float(variance_errors.max()),
        "epsilon_mean": epsilon_mean,
        "epsilon_variance": epsilon_variance,
        "allowed_violation_fraction": allowed_fraction,
    }
    if epsilon_mean is None or epsilon_variance is None or allowed_fraction is None:
        result.update({"hard_or_soft_bound_applied": False, "violation_fraction": None, "passed": True})
    else:
        violations = (mean_errors > epsilon_mean) | (variance_errors > epsilon_variance)
        fraction = float(violations.mean())
        result.update(
            {
                "hard_or_soft_bound_applied": True,
                "violation_fraction": fraction,
                "violation_magnitude": {
                    "mean_excess_max": float(np.maximum(mean_errors - epsilon_mean, 0.0).max()),
                    "variance_excess_max": float(
                        np.maximum(variance_errors - epsilon_variance, 0.0).max()
                    ),
                },
                "passed": fraction <= allowed_fraction,
            }
        )
    return result


def _build_candidate(
    config: LongHorizonTrafficConfig,
    candidate_index: int,
) -> TrafficSequence:
    seed_sequence = np.random.SeedSequence([config.seed, candidate_index])
    total_seed, spatial_seed, allocation_seed = seed_sequence.spawn(3)
    total_rng = np.random.default_rng(total_seed)
    spatial_rng = np.random.default_rng(spatial_seed)
    allocation_rng = np.random.default_rng(allocation_seed)
    total_float, total_metadata = _total_process(config, total_rng)
    probabilities, spatial_metadata = _spatial_process(config, total_metadata, spatial_rng)
    matrices: list[np.ndarray] = []
    pre_clip: list[np.ndarray] = []
    post_clip: list[np.ndarray] = []
    for total, distribution in zip(total_float, probabilities):
        matrix, raw, clipped = _allocate_integer_matrix(total, distribution, config.max_entry, allocation_rng)
        validate_traffic_matrix(matrix)
        matrices.append(matrix)
        pre_clip.append(raw)
        post_clip.append(clipped)
    stack = np.stack(matrices)
    realized_total = stack.sum(axis=(1, 2)).astype(np.float64)
    off_diagonal = ~np.eye(config.num_nodes, dtype=bool)
    mean_ref = np.full((config.num_nodes, config.num_nodes), config.mean_level, dtype=np.float64)
    var_ref = np.full((config.num_nodes, config.num_nodes), config.std_level**2, dtype=np.float64)
    np.fill_diagonal(mean_ref, 0.0)
    np.fill_diagonal(var_ref, 0.0)
    total_reference_mean = _base_total(config)
    # A configured reference, not an empirical future statistic.  This keeps the
    # reference safe if a later phase chooses to expose it to a policy.
    total_reference_variance = float(
        (config.std_level * np.sqrt(config.num_nodes * (config.num_nodes - 1))) ** 2
    )
    if config.family == "rare_shock_recovery":
        # Rare shocks intentionally raise the process-level variance.  Use a
        # parameter-derived target (not a realized/future statistic), otherwise
        # valid shock windows are mislabeled against the calm-family target.
        total_reference_variance = max(
            total_reference_variance,
            (0.30 * total_reference_mean) ** 2,
        )
    constraints = {
        "short": _rolling_scalar_constraint(
            realized_total, config.short_window, total_reference_mean, total_reference_variance, None, None, None
        ),
        "medium": _rolling_scalar_constraint(
            realized_total,
            config.medium_window,
            total_reference_mean,
            total_reference_variance,
            config.medium_epsilon_mean,
            config.medium_epsilon_var,
            config.medium_allowed_violation_fraction,
        ),
        "long": _rolling_scalar_constraint(
            realized_total,
            config.long_window,
            total_reference_mean,
            total_reference_variance,
            config.long_epsilon_mean,
            config.long_epsilon_var,
            config.long_allowed_violation_fraction,
        ),
    }
    matrix_constraints = {
        "short": _rolling_matrix_constraint(
            stack, config.short_window, mean_ref, var_ref, None, None, None
        ),
        "medium": _rolling_matrix_constraint(
            stack,
            config.medium_window,
            mean_ref,
            var_ref,
            config.medium_matrix_epsilon_mean,
            config.medium_matrix_epsilon_var,
            config.medium_allowed_violation_fraction,
        ),
        "long": _rolling_matrix_constraint(
            stack,
            config.long_window,
            mean_ref,
            var_ref,
            config.long_matrix_epsilon_mean,
            config.long_matrix_epsilon_var,
            config.long_allowed_violation_fraction,
        ),
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator_kind": "long-horizon generator",
        "generator_name": "long_horizon_v1",
        "family": config.family,
        "seed": config.seed,
        "sequence_length": config.sequence_length,
        "window_size": config.short_window,
        "generator_config": asdict(config),
        "metadata_usage": "audit/evaluation only; latent state is not a policy input",
        "total_traffic": realized_total.astype(int).tolist(),
        "total_traffic_pre_integer": total_float.tolist(),
        "pre_clip_statistics": _stats(np.stack(pre_clip)[:, off_diagonal]),
        "post_clip_statistics": _stats(np.stack(post_clip)[:, off_diagonal]),
        "post_integer_statistics": _stats(stack[:, off_diagonal]),
        "diagonal_zeroing": {
            "pre_nonzero_count": 0,
            "post_nonzero_count": 0,
            "note": "P_t diagonal is structurally zero before integerization",
        },
        "spatial_distribution_validation": {
            "minimum_probability": float(probabilities.min()),
            "maximum_probability": float(probabilities.max()),
            "maximum_sum_error": float(
                np.max(np.abs(probabilities.sum(axis=(1, 2)) - 1.0))
            ),
            "diagonal_nonzero_count": int(
                np.count_nonzero(np.diagonal(probabilities, axis1=1, axis2=2))
            ),
        },
        "multi_scale_constraints": constraints,
        "matrix_multi_scale_constraints": matrix_constraints,
        "reference_total_mean": total_reference_mean,
        "reference_total_variance": total_reference_variance,
        "generation_attempt": candidate_index + 1,
        **total_metadata,
        **spatial_metadata,
    }
    variant = metadata.get("dynamics_variant")
    suffix = f"-{variant}" if variant else ""
    return TrafficSequence(
        sequence_id=f"{config.family}{suffix}-seed{config.seed}",
        topology_name=config.topology_name,
        family=config.family,
        seed=config.seed,
        matrices=matrices,
        mean_ref=mean_ref,
        var_ref=var_ref,
        bounds=MomentBounds(config.long_epsilon_mean, config.long_epsilon_var),
        metadata=metadata,
    )


def _constraint_penalty(sequence: TrafficSequence) -> float:
    penalty = 0.0
    for constraint_name in ("multi_scale_constraints", "matrix_multi_scale_constraints"):
        constraints = sequence.metadata[constraint_name]
        for level in ("medium", "long"):
            diagnostic = constraints[level]
            if not diagnostic["available"]:
                continue
            penalty += max(
                float(diagnostic["violation_fraction"])
                - float(diagnostic["allowed_violation_fraction"]),
                0.0,
            )
    return penalty


def generate_long_horizon_sequence(config: LongHorizonTrafficConfig) -> TrafficSequence:
    """Generate with finite calibration candidates; never filter one short shock."""
    best: TrafficSequence | None = None
    best_penalty = float("inf")
    for candidate_index in range(config.calibration_candidates):
        candidate = _build_candidate(config, candidate_index)
        penalty = _constraint_penalty(candidate)
        if penalty < best_penalty:
            best, best_penalty = candidate, penalty
        if penalty <= 0.0:
            break
    assert best is not None
    best.metadata["calibration_candidates_evaluated"] = int(best.metadata["generation_attempt"])
    best.metadata["constraint_penalty"] = float(best_penalty)
    best.metadata["constraint_status"] = "passed" if best_penalty <= 0.0 else "best_effort"
    return best


def generate_same_moment_group(config: LongHorizonTrafficConfig) -> dict[str, TrafficSequence]:
    """Generate four dynamics with shared configured mean/variance targets."""
    if config.family != "same_moments_different_dynamics":
        raise ValueError("generate_same_moment_group requires same_moments_different_dynamics")
    values = asdict(config)
    return {
        variant: generate_long_horizon_sequence(
            LongHorizonTrafficConfig(**{**values, "dynamics_variant": variant})
        )
        for variant in SAME_MOMENT_VARIANTS
    }

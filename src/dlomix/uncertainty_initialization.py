"""Utilities for empirical initialization of uncertainty-aware intensity heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


LOG_INTENSITY_EPSILON = 1e-7
_HEAD_COUNT = 6


class StreamingChannelStatistics:
    """Numerically stable per-channel statistics for flattened fragment targets."""

    def __init__(self, channels: int = _HEAD_COUNT, epsilon: float = LOG_INTENSITY_EPSILON):
        if channels <= 0:
            raise ValueError("channels must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.channels = int(channels)
        self.epsilon = float(epsilon)
        self.rows = 0
        self.valid_count = np.zeros(self.channels, dtype=np.int64)
        self.present_count = np.zeros(self.channels, dtype=np.int64)
        self._mean = np.zeros(self.channels, dtype=np.float64)
        self._m2 = np.zeros(self.channels, dtype=np.float64)

    def update(self, targets: np.ndarray) -> None:
        """Accumulate one target matrix with cleavage-major, channel-minor layout."""
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D target matrix, got shape {values.shape}.")
        if values.shape[1] % self.channels != 0:
            raise ValueError(
                f"Target width {values.shape[1]} is not divisible by {self.channels} channels."
            )

        self.rows += int(values.shape[0])
        for channel in range(self.channels):
            channel_values = values[:, channel :: self.channels]
            valid = channel_values >= 0.0
            present = channel_values > 0.0
            self.valid_count[channel] += int(valid.sum())

            positive_values = channel_values[present]
            batch_count = int(positive_values.size)
            if batch_count == 0:
                continue

            self.present_count[channel] += batch_count
            log_values = np.log(positive_values + self.epsilon)
            batch_mean = float(log_values.mean())
            batch_m2 = float(np.square(log_values - batch_mean).sum())

            previous_count = int(self.present_count[channel]) - batch_count
            total_count = previous_count + batch_count
            if previous_count == 0:
                self._mean[channel] = batch_mean
                self._m2[channel] = batch_m2
                continue

            delta = batch_mean - self._mean[channel]
            self._mean[channel] += delta * batch_count / total_count
            self._m2[channel] += batch_m2 + delta * delta * previous_count * batch_count / total_count

    def to_dict(self) -> Dict[str, Any]:
        presence_probability = []
        mean_log_intensity = []
        marginal_variance = []
        channels = []

        for channel in range(self.channels):
            valid_count = int(self.valid_count[channel])
            present_count = int(self.present_count[channel])
            probability: Optional[float] = (
                float(present_count / valid_count) if valid_count else None
            )
            mean: Optional[float] = float(self._mean[channel]) if present_count else None
            variance: Optional[float] = (
                float(self._m2[channel] / present_count) if present_count else None
            )
            absent_count = valid_count - present_count

            presence_probability.append(probability)
            mean_log_intensity.append(mean)
            marginal_variance.append(variance)
            channels.append(
                {
                    "channel": channel,
                    "valid_count": valid_count,
                    "present_count": present_count,
                    "absent_count": absent_count,
                    "presence_probability": probability,
                    "mean_log_intensity": mean,
                    "marginal_log_intensity_variance": variance,
                }
            )

        return {
            "schema_version": 1,
            "epsilon": self.epsilon,
            "fragment_channels": self.channels,
            "ordering": "cleavage-major flattened targets; channel = flat_index % fragment_channels",
            "rows_scanned": self.rows,
            "variance_estimator": "population",
            "mean_log_intensity": mean_log_intensity,
            "presence_probability": presence_probability,
            "marginal_log_intensity_variance": marginal_variance,
            "channels": channels,
        }


def save_statistics(path: str | Path, statistics: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_values(stats_path: str | Path, key: str, count: int) -> np.ndarray:
    with Path(stats_path).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    values = stats.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{stats_path} does not contain a sequence named {key!r}.")
    if len(values) != count:
        raise ValueError(
            f"{stats_path} contains {len(values)} values for {key!r}; expected {count}."
        )
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (count,) or not np.isfinite(values_array).all():
        raise ValueError(f"{stats_path} contains non-finite values for {key!r}.")
    return values_array


def load_log_variance_prior_centers(
    stats_path: str | Path, channel_count: int = _HEAD_COUNT
) -> np.ndarray:
    """Load one residual log-variance prior center per fragment channel."""
    return _load_values(stats_path, "log_variance_prior_centers", int(channel_count))


def _output_bias(model: torch.nn.Module, head_name: str) -> torch.nn.Parameter:
    head = getattr(model, head_name, None)
    output_dense = getattr(head, "output_dense", None)
    bias = getattr(output_dense, "bias", None)
    if bias is None:
        raise ValueError(f"Model head {head_name!r} does not expose output_dense.bias.")
    if bias.numel() != _HEAD_COUNT:
        raise ValueError(
            f"Model head {head_name!r} has {bias.numel()} outputs; expected {_HEAD_COUNT}."
        )
    return bias


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    # log(exp(x) - 1), expressed stably for positive x.
    return values + np.log(-np.expm1(-values))


def apply_empirical_head_initialization(
    model: torch.nn.Module,
    *,
    mean_mode: str = "default",
    presence_mode: str = "default",
    variance_mode: str = "default",
    stats_path: str | Path | None = None,
    variance_parameterization: str = "log_var",
    min_variance: float = 1e-4,
    probability_epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Optionally copy empirical channel statistics into uncertainty-head biases.

    The variance statistic is the marginal target variance in log-intensity
    space. It is intentionally not a residual-variance prior center.
    """
    mean_mode = str(mean_mode).strip().lower()
    presence_mode = str(presence_mode).strip().lower()
    variance_mode = str(variance_mode).strip().lower()
    variance_parameterization = str(variance_parameterization).strip().lower()

    if mean_mode not in {"default", "empirical"}:
        raise ValueError("MEAN_HEAD_INIT must be default or empirical.")
    if presence_mode not in {"default", "empirical"}:
        raise ValueError("PRESENCE_HEAD_INIT must be default or empirical.")
    if variance_mode not in {"default", "empirical_marginal"}:
        raise ValueError("VARIANCE_HEAD_INIT must be default or empirical_marginal.")
    if variance_parameterization not in {"log_var", "softplus_variance"}:
        raise ValueError("variance_parameterization must be log_var or softplus_variance.")
    if min_variance <= 0:
        raise ValueError("min_variance must be positive.")
    if not 0 < probability_epsilon < 0.5:
        raise ValueError("probability_epsilon must lie strictly between 0 and 0.5.")

    use_stats = any(mode != "default" for mode in (mean_mode, presence_mode, variance_mode))
    if not use_stats:
        return {}
    if not stats_path:
        raise ValueError("HEAD_INIT_STATS is required when an empirical head initializer is enabled.")

    applied: Dict[str, Any] = {
        "head_init_stats": str(stats_path),
        "mean_head_init": mean_mode,
        "presence_head_init": presence_mode,
        "variance_head_init": variance_mode,
    }

    with torch.no_grad():
        if mean_mode == "empirical":
            values = _load_values(stats_path, "mean_log_intensity", _HEAD_COUNT)
            bias = _output_bias(model, "log_mean_regressor")
            bias.copy_(torch.as_tensor(values, device=bias.device, dtype=bias.dtype))
            applied["mean_head_bias"] = values.tolist()

        if presence_mode == "empirical":
            probabilities = _load_values(stats_path, "presence_probability", _HEAD_COUNT)
            if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                raise ValueError("presence_probability values must lie in [0, 1].")
            probabilities = np.clip(probabilities, probability_epsilon, 1.0 - probability_epsilon)
            values = np.log(probabilities) - np.log1p(-probabilities)
            bias = _output_bias(model, "presence_regressor")
            bias.copy_(torch.as_tensor(values, device=bias.device, dtype=bias.dtype))
            applied["presence_head_probability"] = probabilities.tolist()
            applied["presence_head_bias"] = values.tolist()

        if variance_mode == "empirical_marginal":
            marginal_variance = _load_values(
                stats_path, "marginal_log_intensity_variance", _HEAD_COUNT
            )
            if np.any(marginal_variance <= 0.0):
                raise ValueError("marginal_log_intensity_variance values must be positive.")
            if variance_parameterization == "log_var":
                values = np.log(marginal_variance)
            else:
                unconstrained_variance = marginal_variance - float(min_variance)
                if np.any(unconstrained_variance <= 0.0):
                    raise ValueError(
                        "Empirical marginal variance must exceed MIN_VARIANCE for softplus_variance initialization."
                    )
                values = _inverse_softplus(unconstrained_variance)
            bias = _output_bias(model, "log_var_regressor")
            bias.copy_(torch.as_tensor(values, device=bias.device, dtype=bias.dtype))
            applied["marginal_target_variance"] = marginal_variance.tolist()
            applied["variance_head_bias"] = values.tolist()

    return applied

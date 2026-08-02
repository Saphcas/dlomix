"""Estimate channel-wise residual log-variance priors from a trained UA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DLOMIX_BACKEND", "torch")

import numpy as np
import torch
from tqdm.auto import tqdm

from dlomix.data import StreamingFragmentIonIntensityDataset
from dlomix.models import PrositIntensityUncertaintyPredictor
from dlomix.uncertainty_initialization import LOG_INTENSITY_EPSILON, save_statistics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Parquet split used to estimate residuals.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--reservoir-size", type=int, default=250_000)
    parser.add_argument("--variance-parameterization", choices=("log_var", "softplus_variance"), default="log_var")
    parser.add_argument("--min-variance", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--embedding-output-dim", type=int, default=16)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--latent-dropout-rate", type=float, default=0.1)
    parser.add_argument("--recurrent-layer-sizes", type=int, nargs=2, default=(256, 512))
    parser.add_argument("--regressor-layer-size", type=int, default=512)
    parser.add_argument("--len-fion", type=int, default=6)
    parser.add_argument("--channels", type=int, default=6)
    return parser.parse_args()


def _move_and_cast(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    result["modified_sequence"] = result["modified_sequence"].long()
    for key in (
        "intensities_raw",
        "collision_energy_aligned_normed",
        "precursor_charge_onehot",
        "mod_loss",
        "delta_mass",
    ):
        if key in result and torch.is_tensor(result[key]):
            result[key] = result[key].float()
    return result


class ResidualChannelStatistics:
    """Exact residual moments plus deterministic bounded-sample MAD estimates."""

    def __init__(self, channels: int = 6, reservoir_size: int = 250_000):
        self.channels = channels
        self.reservoir_size = reservoir_size
        self.count = np.zeros(channels, dtype=np.int64)
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)
        self.samples = [[] for _ in range(channels)]
        self.rng = np.random.default_rng(0)

    def update_channel(self, channel: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if not values.size:
            return
        previous = int(self.count[channel])
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(np.square(values - batch_mean).sum())
        total = previous + batch_count
        if previous:
            delta = batch_mean - self.mean[channel]
            self.mean[channel] += delta * batch_count / total
            self.m2[channel] += batch_m2 + delta * delta * previous * batch_count / total
        else:
            self.mean[channel] = batch_mean
            self.m2[channel] = batch_m2
        self.count[channel] = total
        sample = self.samples[channel]
        for offset, value in enumerate(values, start=1):
            seen = previous + offset
            if len(sample) < self.reservoir_size:
                sample.append(float(value))
            else:
                slot = int(self.rng.integers(seen))
                if slot < self.reservoir_size:
                    sample[slot] = float(value)

    def update(self, residuals: np.ndarray) -> None:
        for channel in range(self.channels):
            self.update_channel(channel, residuals[:, channel :: self.channels])

    def to_dict(self, min_variance: float) -> dict:
        channels = []
        residual_variance = []
        residual_mean_squared = []
        robust_mad_variance = []
        centers = []
        for channel in range(self.channels):
            count = int(self.count[channel])
            variance = float(self.m2[channel] / count) if count else None
            mean = float(self.mean[channel]) if count else None
            mean_squared = variance + mean * mean if variance is not None else None
            sample = np.asarray(self.samples[channel], dtype=np.float64)
            if sample.size:
                median = float(np.median(sample))
                mad = float(np.median(np.abs(sample - median)))
                robust_variance = float((1.4826 * mad) ** 2)
            else:
                median = mad = robust_variance = None
            center = (
                float(np.log(max(mean_squared, min_variance)))
                if mean_squared is not None
                else None
            )
            residual_variance.append(variance)
            residual_mean_squared.append(mean_squared)
            robust_mad_variance.append(robust_variance)
            centers.append(center)
            channels.append(
                {
                    "channel": channel,
                    "present_count": count,
                    "residual_mean": mean,
                    "residual_variance": variance,
                    "residual_mean_squared": mean_squared,
                    "robust_mad": mad,
                    "robust_mad_variance": robust_variance,
                    "mad_reservoir_count": int(sample.size),
                    "log_variance_prior_center": center,
                }
            )
        if any(value is None or not np.isfinite(value) for value in centers):
            raise ValueError("At least one channel has no finite residual variance.")
        return {
            "schema_version": 1,
            "epsilon": LOG_INTENSITY_EPSILON,
            "fragment_channels": self.channels,
            "ordering": "cleavage-major flattened targets; channel = flat_index % fragment_channels",
            "residual_variance_estimator": "population centered variance",
            "residual_mean_squared_estimator": "E[r^2] = Var(r) + E[r]^2",
            "robust_variance_estimator": "squared scaled MAD from deterministic reservoir sample",
            "log_variance_prior_center_estimator": "log(max(E[r^2], min_variance))",
            "residual_variance": residual_variance,
            "residual_mean_squared": residual_mean_squared,
            "robust_mad_variance": robust_mad_variance,
            "log_variance_prior_centers": centers,
            "channels": channels,
        }


def main() -> int:
    args = _parse_args()
    if not args.data.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Both --data and --checkpoint must exist.")
    if args.reservoir_size <= 0 or args.min_variance <= 0:
        raise ValueError("--reservoir-size and --min-variance must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StreamingFragmentIonIntensityDataset(
        train_path=str(args.data),
        val_path=str(args.data),
        test_path=str(args.data),
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_buffer_size=1,
        seed=0,
        with_termini=True,
        encoding_scheme="unmod",
        model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
        features_to_extract=["mod_loss", "delta_mass"],
        sequence_column="modified_sequence",
        label_column="intensities_raw",
        parquet_read_batch_size=50_000,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=1,
        persistent_workers=False,
        in_order=True,
    )
    if args.channels <= 0 or args.len_fion != args.channels:
        raise ValueError("--channels must be positive and match --len-fion.")
    model = PrositIntensityUncertaintyPredictor(
        seq_length=args.max_seq_len,
        embedding_output_dim=args.embedding_output_dim,
        dropout_rate=args.dropout_rate,
        latent_dropout_rate=args.latent_dropout_rate,
        recurrent_layers_sizes=tuple(args.recurrent_layer_sizes),
        regressor_layer_size=args.regressor_layer_size,
        len_fion=args.len_fion,
        use_prosit_ptm_features=True,
        input_keys={"SEQUENCE_KEY": "modified_sequence"},
        meta_data_keys={
            "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
            "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        },
        with_termini=True,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    output_bias_key = "log_mean_regressor.output_dense.bias"
    checkpoint_channels = state.get(output_bias_key)
    if checkpoint_channels is None:
        raise ValueError(
            "Checkpoint is not a compatible uncertainty-aware Prosit model: "
            f"missing {output_bias_key!r}."
        )
    if checkpoint_channels.numel() != args.channels:
        raise ValueError(
            "Checkpoint fragment-channel count does not match --channels: "
            f"checkpoint={checkpoint_channels.numel()}, requested={args.channels}."
        )
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint architecture does not match the requested uncertainty-aware "
            "Prosit model configuration."
        ) from error
    model.eval()

    stats = ResidualChannelStatistics(
        channels=args.channels, reservoir_size=args.reservoir_size
    )
    with torch.no_grad():
        iterator = tqdm(dataset.tensor_train_data, desc="Residual statistics", unit="batch")
        for batch_index, batch in enumerate(iterator, start=1):
            batch = _move_and_cast(batch, device)
            mean, _, _ = model(batch)
            target = batch["intensities_raw"]
            if target.shape[-1] % args.channels:
                raise ValueError(
                    "Target width must be divisible by --channels: "
                    f"width={target.shape[-1]}, channels={args.channels}."
                )
            present = target > 0
            residual = torch.zeros_like(target)
            residual[present] = (
                torch.log(target[present] + LOG_INTENSITY_EPSILON) - mean[present]
            )
            # Update per channel directly so -1 and zero targets never enter the statistics.
            for channel in range(args.channels):
                channel_mask = present[:, channel::args.channels]
                values = (
                    residual[:, channel::args.channels][channel_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if values.size:
                    stats.update_channel(channel, values)
            if args.max_batches and batch_index >= args.max_batches:
                break

    output = stats.to_dict(args.min_variance)
    output.update(
        {
            "source_parquet": str(args.data),
            "checkpoint": str(args.checkpoint),
            "variance_parameterization": args.variance_parameterization,
            "min_variance": args.min_variance,
        }
    )
    save_statistics(args.output, output)
    print(f"Wrote {args.output}")
    for channel in output["channels"]:
        print(
            "channel={channel} count={present_count} residual_mean={residual_mean} "
            "residual_var={residual_variance} residual_mean_squared={residual_mean_squared} "
            "mad_var={robust_mad_variance} log_var_center={log_variance_prior_center}".format(**channel)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

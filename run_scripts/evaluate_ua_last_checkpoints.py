"""Evaluate the last epoch checkpoint from uncertainty-aware sweep runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from dlomix.data import StreamingFragmentIonIntensityDataset
from dlomix.models import PrositIntensityUncertaintyPredictor
from train_prosit_intensity_ptms_torch import (
    ColumnConfig,
    _cast_batch_types,
    _mae_from_log_mean,
    _move_batch_to_device,
    _spectral_angle_from_log_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int, default=0, help="Evaluate this exact epoch; 0 means last checkpoint.")
    return parser.parse_args()


def select_checkpoint(root: Path, epoch: int) -> Path | None:
    candidates = []
    for path in root.rglob("checkpoint_epoch_*.pth"):
        match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pth", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if epoch:
        for number, path in candidates:
            if number == epoch:
                return path
        return None
    return max(candidates, default=(0, None))[1]


def build_dataset(args: argparse.Namespace) -> StreamingFragmentIonIntensityDataset:
    return StreamingFragmentIonIntensityDataset(
        train_path=str(args.val),
        val_path=str(args.val),
        test_path=str(args.test),
        max_seq_len=32,
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
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=1,
        persistent_workers=False,
        in_order=True,
    )


def build_model(device: torch.device) -> PrositIntensityUncertaintyPredictor:
    return PrositIntensityUncertaintyPredictor(
        seq_length=32,
        embedding_output_dim=16,
        dropout_rate=0.3,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        len_fion=6,
        use_prosit_ptm_features=True,
        input_keys={"SEQUENCE_KEY": "modified_sequence"},
        meta_data_keys={
            "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
            "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        },
        with_termini=True,
    ).to(device)


def evaluate(model, iterator, columns, device, limit: int, desc: str) -> tuple[float, float, int]:
    mae_total = 0.0
    sa_total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in tqdm(iterator, desc=desc, unit="batch", leave=False):
            batch = _cast_batch_types(_move_batch_to_device(batch, device), columns)
            mean, _, presence = model(batch)
            mae_total += _mae_from_log_mean(batch[columns.label], mean, presence)
            sa_total += _spectral_angle_from_log_mean(batch[columns.label], mean, presence)
            batches += 1
            if limit and batches >= limit:
                break
    return mae_total / max(1, batches), sa_total / max(1, batches), batches


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    columns = ColumnConfig(
        sequence="modified_sequence",
        label="intensities_raw",
        collision_energy="collision_energy_aligned_normed",
        precursor_charge="precursor_charge_onehot",
    )
    dataset = build_dataset(args)
    rows = []
    roots = [root for parent in args.roots for root in (sorted(parent.iterdir()) if parent.is_dir() else [parent])]
    for root in roots:
        checkpoint = select_checkpoint(root, args.epoch)
        if checkpoint is None:
            print(f"Skipping {root}: no checkpoint_epoch_N.pth found")
            continue
        model = build_model(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        val = evaluate(
            model, dataset.tensor_val_data, columns, device, args.max_val_batches,
            f"{root.name} [val]",
        )
        test = evaluate(
            model, dataset.tensor_test_data, columns, device, args.max_test_batches,
            f"{root.name} [test]",
        )
        row = {
            "run": root.name,
            "checkpoint": str(checkpoint),
            "epoch": state.get("epoch", checkpoint.stem.rsplit("_", 1)[-1]),
            "val_mae": val[0], "val_spectral_angle": val[1], "val_batches": val[2],
            "test_mae": test[0], "test_spectral_angle": test[1], "test_batches": test[2],
        }
        rows.append(row)
        print(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

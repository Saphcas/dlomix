"""Stream empirical initialization statistics from Prosit intensity targets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DLOMIX_BACKEND", "torch")

import numpy as np
import pyarrow.parquet as pq

from dlomix.uncertainty_initialization import (
    LOG_INTENSITY_EPSILON,
    StreamingChannelStatistics,
    save_statistics,
)


def _target_matrix(record_batch, label_column: str) -> np.ndarray:
    column_index = record_batch.schema.get_field_index(label_column)
    if column_index < 0:
        raise ValueError(f"Parquet batch does not contain label column {label_column!r}.")
    labels = record_batch.column(column_index)
    offsets = labels.offsets.to_numpy(zero_copy_only=False)
    widths = np.diff(offsets)
    if widths.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if not np.all(widths == widths[0]):
        raise ValueError("Found ragged fragment-intensity targets; expected fixed-width vectors.")
    values = labels.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(labels), int(widths[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Training parquet file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file.")
    parser.add_argument("--label-column", default="intensities_raw")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--read-batch-size", type=int, default=50_000)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional cap for smoke tests; 0 scans all batches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.read_batch_size <= 0:
        raise ValueError("--read-batch-size must be positive.")
    if not args.train.is_file():
        raise FileNotFoundError(args.train)

    parquet = pq.ParquetFile(args.train)
    statistics = StreamingChannelStatistics(
        channels=args.channels,
        epsilon=LOG_INTENSITY_EPSILON,
    )
    for batch_index, record_batch in enumerate(
        parquet.iter_batches(batch_size=args.read_batch_size, columns=[args.label_column]),
        start=1,
    ):
        statistics.update(_target_matrix(record_batch, args.label_column))
        if args.max_batches and batch_index >= args.max_batches:
            break
        if batch_index % 20 == 0:
            print(f"Processed {statistics.rows:,} rows")

    output = statistics.to_dict()
    output["source_parquet"] = str(args.train)
    output["label_column"] = args.label_column
    save_statistics(args.output, output)
    print(f"Wrote {args.output}")
    for channel in output["channels"]:
        print(
            "channel={channel} valid={valid_count} present={present_count} "
            "p={presence_probability} mean_log={mean_log_intensity} "
            "marginal_var={marginal_log_intensity_variance}".format(**channel)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

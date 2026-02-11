from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StreamingConfig:
    max_seq_len: int
    batch_size: int
    shuffle: bool
    shuffle_buffer_size: int
    seed: int
    with_termini: bool
    encoding_scheme: str
    padding_value: str
    alphabet: Dict[str, int]
    model_features: List[str]
    features_to_extract: List[str]
    sequence_column: str
    label_column: str
    return_debug_tokens: bool
    # Number of rows to read from parquet per IO batch (before per-example processing).
    # This is independent from `batch_size` used for the training DataLoader batching.
    parquet_read_batch_size: int


def _buffered_shuffle(
    it: Iterable[Dict[str, Any]],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[Dict[str, Any]]:
    if buffer_size <= 1:
        yield from it
        return

    buffer: List[Dict[str, Any]] = []
    for item in it:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue

        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = item

    rng.shuffle(buffer)
    yield from buffer


def _ensure_padding_token(alphabet: Dict[str, int], padding_value: str) -> Dict[str, int]:
    extended = dict(alphabet)
    if padding_value not in extended:
        extended[padding_value] = 0
    return extended


def _filter_by_keep_mask(batch: Dict[str, Any], keep_mask: Sequence[bool]) -> Dict[str, Any]:
    kept: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list):
            kept[key] = [v for v, keep in zip(value, keep_mask) if keep]
        else:
            kept[key] = value
    return kept


class ParquetExampleStream:
    def __init__(self, path: str, columns: List[str], read_batch_size: int):
        self.path = path
        self.columns = columns
        self.read_batch_size = read_batch_size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self.path)
        for record_batch in pf.iter_batches(
            batch_size=self.read_batch_size, columns=self.columns
        ):
            yield record_batch.to_pydict()


class StreamingPeptideProcessor:
    def __init__(self, config: StreamingConfig):
        self.config = config

        from .dataset_utils import EncodingScheme
        from .processing.feature_extractors import AVAILABLE_FEATURE_EXTRACTORS
        from .processing.processors import (
            SequenceEncodingProcessor,
            SequencePaddingProcessor,
            SequenceParsingProcessor,
            SequencePTMRemovalProcessor,
        )
        from .processing.feature_extractors import LookupFeatureExtractor, FEATURE_EXTRACTORS_PARAMETERS

        self.encoding_scheme = EncodingScheme(config.encoding_scheme)
        self.extended_alphabet = _ensure_padding_token(config.alphabet, config.padding_value)

        self.sequence_parser = SequenceParsingProcessor(
            config.sequence_column, batched=True, with_termini=config.with_termini
        )

        self.ptm_remover: Optional[SequencePTMRemovalProcessor] = None
        if self.encoding_scheme == EncodingScheme.UNMOD:
            self.ptm_remover = SequencePTMRemovalProcessor(
                sequence_column_name=config.sequence_column, batched=True
            )

        self.encoder = SequenceEncodingProcessor(
            sequence_column_name=config.sequence_column,
            alphabet=self.extended_alphabet,
            batched=True,
            extend_alphabet=False,  # fixed vocab in streaming mode
            fallback_unmodified=False,
        )

        self.padder = SequencePaddingProcessor(
            sequence_column_name=config.sequence_column,
            batched=True,
            padding_index=self.extended_alphabet[config.padding_value],
            max_length=config.max_seq_len,
        )

        self.feature_extractors = []
        for feature_name in config.features_to_extract:
            feature_name = feature_name.lower()
            if feature_name not in AVAILABLE_FEATURE_EXTRACTORS:
                continue
            self.feature_extractors.append(
                LookupFeatureExtractor(
                    sequence_column_name=SequenceParsingProcessor.PARSED_COL_NAMES["seq"],
                    feature_column_name=feature_name,
                    **FEATURE_EXTRACTORS_PARAMETERS[feature_name],
                    max_length=config.max_seq_len,
                    batched=True,
                )
            )

    def process_record_batch(self, batch: Dict[str, Any], is_test_split: bool) -> Dict[str, Any]:
        from .processing.processors import SequencePaddingProcessor

        # `datasets.Dataset.map()` merges returned dicts into the existing example(s),
        # it does not replace the entire record. Some processors in this codebase
        # return only the updated column(s). We replicate that behavior here.
        def apply_and_merge(processor, data: Dict[str, Any]) -> Dict[str, Any]:
            updates = processor(data)
            if updates is not data:
                data.update(updates)
            return data

        batch = apply_and_merge(self.sequence_parser, batch)

        if self.ptm_remover is not None:
            batch = apply_and_merge(self.ptm_remover, batch)

        if self.config.return_debug_tokens:
            seqs = batch.get(self.config.sequence_column, [])
            debug_tokens = []
            for seq in seqs:
                seq = list(seq)
                if len(seq) >= self.config.max_seq_len:
                    seq = seq[: self.config.max_seq_len]
                else:
                    seq = seq + [self.config.padding_value] * (
                        self.config.max_seq_len - len(seq)
                    )
                debug_tokens.append(seq)
            batch["_debug_input_tokens"] = debug_tokens

        batch = apply_and_merge(self.encoder, batch)
        batch = apply_and_merge(self.padder, batch)

        if not is_test_split:
            keep_mask = batch[SequencePaddingProcessor.KEEP_COLUMN_NAME]
            batch = _filter_by_keep_mask(batch, keep_mask)

        for extractor in self.feature_extractors:
            batch = apply_and_merge(extractor, batch)

        batch.pop(SequencePaddingProcessor.KEEP_COLUMN_NAME, None)
        return batch


def _to_torch(x: Any, dtype: Optional[str] = None):
    import torch

    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    if dtype is not None:
        t = t.to(getattr(torch, dtype))
    return t


class StreamingFragmentIonIntensityDataset:
    """
    Parquet -> PyTorch streaming dataset for Prosit intensity, avoiding HuggingFace `datasets` caching.

    This is intentionally minimal and reuses the existing processor implementations
    (ProForma parsing, PTM removal, encoding, padding, LookupFeatureExtractor PTM channels),
    but runs them on `pyarrow.parquet.ParquetFile.iter_batches()` record batches instead of
    `datasets.Dataset.map()`.
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str],
        test_path: Optional[str],
        *,
        max_seq_len: int = 32,
        batch_size: int = 8,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        with_termini: bool = True,
        encoding_scheme: str = "unmod",
        padding_value: str = "-",
        alphabet: Optional[Dict[str, int]] = None,
        model_features: Optional[List[str]] = None,
        features_to_extract: Optional[List[str]] = None,
        sequence_column: str = "modified_sequence",
        label_column: str = "intensities_raw",
        parquet_read_batch_size: int = 50_000,
        return_debug_tokens: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        from ..constants import ALPHABET_UNMOD

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        config = StreamingConfig(
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            with_termini=with_termini,
            encoding_scheme=encoding_scheme,
            padding_value=padding_value,
            alphabet=alphabet or dict(ALPHABET_UNMOD),
            model_features=model_features or [],
            features_to_extract=features_to_extract or [],
            sequence_column=sequence_column,
            label_column=label_column,
            return_debug_tokens=return_debug_tokens,
            parquet_read_batch_size=parquet_read_batch_size,
        )

        self.config = config
        self._processor = StreamingPeptideProcessor(config)

        self._dl_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    def _iter_split(self, path: str, *, is_test_split: bool) -> Iterator[Dict[str, Any]]:
        columns = [
            self.config.sequence_column,
            self.config.label_column,
            *self.config.model_features,
        ]
        stream = ParquetExampleStream(
            path, columns=columns, read_batch_size=self.config.parquet_read_batch_size
        )

        rng = random.Random(self.config.seed)

        def iter_examples() -> Iterator[Dict[str, Any]]:
            for record_batch in stream:
                processed = self._processor.process_record_batch(
                    record_batch, is_test_split=is_test_split
                )

                n = len(processed[self.config.sequence_column])
                for i in range(n):
                    ex: Dict[str, Any] = {}
                    ex[self.config.sequence_column] = _to_torch(
                        processed[self.config.sequence_column][i], dtype="long"
                    )
                    ex[self.config.label_column] = _to_torch(
                        processed[self.config.label_column][i], dtype="float32"
                    )

                    for feat in self.config.model_features:
                        ex[feat] = _to_torch(processed[feat][i], dtype="float32")

                    for feat in self.config.features_to_extract:
                        feat = feat.lower()
                        if feat in processed:
                            ex[feat] = _to_torch(processed[feat][i], dtype="float32")

                    if self.config.return_debug_tokens and "_debug_input_tokens" in processed:
                        ex["_debug_input_tokens"] = processed["_debug_input_tokens"][i]

                    yield ex

        examples_it: Iterable[Dict[str, Any]] = iter_examples()
        if self.config.shuffle and not is_test_split:
            examples_it = _buffered_shuffle(
                examples_it, buffer_size=self.config.shuffle_buffer_size, rng=rng
            )
        yield from examples_it

    def _collate_dicts(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if torch.is_tensor(vals[0]):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out

    @property
    def unknown_token_index(self) -> int:
        return int(getattr(self._processor.encoder, "unknown_token_index"))

    @property
    def tensor_train_data(self):
        import torch.utils.data

        class _Iter(torch.utils.data.IterableDataset):
            def __iter__(self_nonlocal):
                yield from self._iter_split(self.train_path, is_test_split=False)

        return torch.utils.data.DataLoader(
            _Iter(),
            shuffle=False,
            collate_fn=self._collate_dicts,
            **self._dl_kwargs,
        )

    @property
    def tensor_val_data(self):
        if self.val_path is None:
            raise ValueError("No val_path provided.")

        import torch.utils.data

        class _Iter(torch.utils.data.IterableDataset):
            def __iter__(self_nonlocal):
                yield from self._iter_split(self.val_path, is_test_split=False)

        return torch.utils.data.DataLoader(
            _Iter(),
            shuffle=False,
            collate_fn=self._collate_dicts,
            **self._dl_kwargs,
        )

    @property
    def tensor_test_data(self):
        if self.test_path is None:
            raise ValueError("No test_path provided.")

        import torch.utils.data

        class _Iter(torch.utils.data.IterableDataset):
            def __iter__(self_nonlocal):
                yield from self._iter_split(self.test_path, is_test_split=True)

        return torch.utils.data.DataLoader(
            _Iter(),
            shuffle=False,
            collate_fn=self._collate_dicts,
            **self._dl_kwargs,
        )

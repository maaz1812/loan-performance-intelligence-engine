"""
Chunked loading and deterministic loan-level sampling.

Implements dataset_usage.md Section 17.3 Rule 2 (chunk-based processing) and
the sampling decision recorded as Gap 3 in docs/documentation_analysis.md.

Why sample LOANS and never ROWS
-------------------------------
The corpus is ~870M loan-month records. Row-level sampling would shred each
loan's trajectory, which is exactly what forward-looking labels, rolling
features and survival analysis depend on. Sampling whole loans keeps every
selected loan's complete history intact and, as a bonus, delivers the
loan-level containment that dataset_usage.md Section 8.3 requires -- a loan is
either wholly in the panel or wholly absent, so it can never straddle the
train/validation boundary.

Why a HASH and not an RNG
-------------------------
A deterministic hash of the loan id means the same loans are selected on every
run, on every machine, without carrying a sampled-id list around. That is what
makes the modelling population reproducible (prd.md Section 5,
dataset_usage.md Section 16).
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..config import CFG
from .extract import stream_member
from .schema_detection import FIELDS, USED_IDX, USED_NAMES

# All 113 positional names, in order -- required so pandas can address the
# columns we prune to by integer position.
_ALL_NAMES: list[str] = [f.name for f in FIELDS]

# Everything is read as string and converted explicitly. Letting pandas infer
# types per chunk risks a column being int64 in one chunk and object in the
# next, which silently corrupts a concatenated panel.
_DTYPE = {name: "string" for name in USED_NAMES}


def stream_chunks(
    zip_path: Path,
    chunk_size: int | None = None,
    usecols: tuple[int, ...] = USED_IDX,
    names: tuple[str, ...] = USED_NAMES,
) -> Iterator[pd.DataFrame]:
    """
    Yield successive chunks of a raw quarterly file as DataFrames.

    The file is streamed straight out of its ZIP -- never extracted. Peak
    memory is one chunk. Column pruning via `usecols` is the single biggest
    throughput win available here: parsing 45 of 113 fields roughly triples
    read speed.
    """
    chunk_size = chunk_size or CFG.ingest.chunk_size
    dtype = {n: "string" for n in names}
    with stream_member(zip_path) as fh:
        reader = pd.read_csv(
            fh,
            sep="|",
            header=None,
            names=_ALL_NAMES,
            usecols=list(usecols),
            dtype=dtype,
            chunksize=chunk_size,
            na_filter=False,      # keep '' as '' so missingness is explicit
            engine="c",
            low_memory=False,
        )
        for chunk in reader:
            # read_csv returns columns in file order; reindex to our declared
            # order so downstream positional assumptions hold.
            yield chunk[list(names)]


# ---------------------------------------------------------------------------
# Deterministic loan sampling
# ---------------------------------------------------------------------------
_SPLITMIX_A = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_B = np.uint64(0x94D049BB133111EB)
_BUCKETS = np.uint64(1_000_000)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """
    SplitMix64 finaliser -- a strong, fully vectorised integer hash.

    Fannie Mae loan ids are sequential-ish 12-digit integers, so a naive
    `loan_id % N` sample would correlate with acquisition order and quietly
    bias the sample. Running the id through an avalanche mix removes that
    structure. uint64 arithmetic wraps by design here, which is what the
    algorithm expects.
    """
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * _SPLITMIX_A
        x = x ^ (x >> np.uint64(27))
        x = x * _SPLITMIX_B
        x = x ^ (x >> np.uint64(31))
    return x


def loan_hash_bucket(loan_ids: pd.Series, salt: str | None = None) -> np.ndarray:
    """Map each loan id to a stable bucket in [0, 1_000_000)."""
    salt = salt if salt is not None else CFG.ingest.sample_salt
    ids = pd.to_numeric(loan_ids, errors="coerce").fillna(0).astype("uint64").to_numpy()
    seed = np.uint64(zlib.crc32(salt.encode("utf-8")))
    with np.errstate(over="ignore"):
        mixed = _splitmix64(ids + seed)
    return (mixed % _BUCKETS).astype(np.uint64)


def sample_mask(
    loan_ids: pd.Series,
    rate: float | None = None,
    salt: str | None = None,
) -> np.ndarray:
    """
    Boolean mask selecting a deterministic `rate` fraction of distinct loans.

    Identical for a given (loan_id, rate, salt) on every run and every machine.
    """
    rate = rate if rate is not None else CFG.ingest.sample_rate
    if rate >= 1.0:
        return np.ones(len(loan_ids), dtype=bool)
    threshold = np.uint64(int(round(rate * float(_BUCKETS))))
    return loan_hash_bucket(loan_ids, salt) < threshold


def estimate_sample_rate(zip_path: Path, n_chunks: int = 2) -> dict:
    """
    Sanity-check the achieved sample rate on a couple of chunks.

    Guards against an off-by-one in the bucket threshold silently producing a
    sample an order of magnitude off target.
    """
    seen_loans: set[str] = set()
    kept_loans: set[str] = set()
    rows = kept_rows = 0
    for i, chunk in enumerate(stream_chunks(zip_path)):
        if i >= n_chunks:
            break
        ids = chunk["LOAN_ID"]
        m = sample_mask(ids)
        rows += len(chunk)
        kept_rows += int(m.sum())
        seen_loans.update(ids.unique().tolist())
        kept_loans.update(ids[m].unique().tolist())
    return {
        "rows_seen": rows,
        "rows_kept": kept_rows,
        "row_rate": kept_rows / rows if rows else 0.0,
        "loans_seen": len(seen_loans),
        "loans_kept": len(kept_loans),
        "loan_rate": len(kept_loans) / len(seen_loans) if seen_loans else 0.0,
        "target_rate": CFG.ingest.sample_rate,
    }


__all__ = [
    "stream_chunks", "loan_hash_bucket", "sample_mask", "estimate_sample_rate",
]

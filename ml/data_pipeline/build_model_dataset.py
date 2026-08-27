"""
Model-dataset builder: shards -> cleaned panel -> features -> time-aware splits.

This is the stage that sits between raw ingestion (dataset_builder.py) and
training. For each quarterly shard it runs the full chain:

    read shard  ->  clean_panel        (dataset_usage.md S5)
                ->  apply_rules        (S5.4 / S11, orchestrator.md S3.2)
                ->  build_features     (S6 features + S7 targets)
                ->  route rows to their time split (S8)

and appends the result to a set of ParquetWriters. Nothing larger than one
quarter is ever resident, which is what makes this run in ~2.5 GB of free RAM.

Outputs
-------
data/features/engineered_features.parquet   full feature table + targets
data/splits/train.parquet                   reporting_month <= 2020-12
data/splits/validation.parquet              2021
data/splits/test.parquet                    latest fully-labelled period
data/splits/holdout_2022.parquet            contiguous secondary holdout
data/processed/quarantine.parquet           every rejected row + reason code
data/processed/dq_scores.parquet            per-record data-quality scores

Why splits are written here rather than at train time
-----------------------------------------------------
The split is a property of the DATA, not of a model run. Materialising it once,
deterministically, means every model trains on byte-identical inputs and the
"no loan crosses the boundary" guarantee is established in exactly one place
and testable (tests/test_splits.py).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CFG, FEATURES_DIR, PROCESSED_DIR, REFERENCE_DIR, SPLITS_DIR
from .dataset_builder import SHARD_DIR
from .feature_engineering import build_features, feature_columns
from .preprocessing import clean_panel
from .supporting_data import load_servicer_updates
from .validation import apply_rules, batch_summary

TRAIN, VALID, TEST, HOLDOUT = "train", "validation", "test", "holdout_2022"


class _MultiWriter:
    """Append-only Parquet writers keyed by name, created lazily on first write."""

    def __init__(self) -> None:
        self._w: dict[str, pq.ParquetWriter] = {}
        self._paths: dict[str, Path] = {}
        self.rows: dict[str, int] = {}

    def write(self, name: str, path: Path, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if name not in self._w:
            self._w[name] = pq.ParquetWriter(path, tbl.schema, compression="snappy")
            self._paths[name] = path
            self.rows[name] = 0
        else:
            # Schemas must match across shards; align columns defensively.
            tbl = tbl.select(self._w[name].schema.names)
        self._w[name].write_table(tbl)
        self.rows[name] = self.rows.get(name, 0) + len(df)

    def close(self) -> None:
        for w in self._w.values():
            w.close()
        self._w.clear()


def resolve_data_end() -> pd.Timestamp:
    """
    The last reporting month present anywhere in the corpus.

    Read from the full-population aggregates so it reflects 100% of the data,
    not the sample. Label completeness depends on this, so it must be a
    corpus-level fact.
    """
    p = PROCESSED_DIR / "population_monthly_aggregates.parquet"
    if p.exists():
        pop = pd.read_parquet(p, columns=["reporting_month"])
        return pd.Timestamp(pop["reporting_month"].max())
    ends = []
    for f in sorted(SHARD_DIR.glob("monthly_*.parquet")):
        ends.append(pd.read_parquet(f, columns=["reporting_month"])["reporting_month"].max())
    return pd.Timestamp(max(ends))


def resolve_test_window(data_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    The "latest available period" test window, computed from the data.

    The 12-month forward horizon means the last month that can carry a COMPLETE
    default/prepayment label is 12 months before the data end. The test window is
    the 12 months ending there, so it is both the latest available and fully
    labelled. Computed rather than hard-coded, per the Gap 4 decision in
    docs/documentation_analysis.md.
    """
    last_labelled = (data_end - pd.DateOffset(months=CFG.rules.horizon_default)).replace(day=1)
    start = (last_labelled - pd.DateOffset(months=11)).replace(day=1)
    return start, last_labelled


def assign_split(months: pd.Series, test_start: pd.Timestamp,
                 test_end: pd.Timestamp) -> pd.Series:
    """
    Calendar-time split assignment. NEVER random (decision.md ADR-2).

    Because loans are sampled whole and every row of a loan carries the same
    loan_id, a loan can appear in more than one calendar window -- that is
    inherent to panel data and is exactly what the documentation intends: the
    model trains on a loan's PAST and is evaluated on a strictly LATER period.
    What must never happen is the same (loan, month) row, or a row whose label
    window overlaps the evaluation window, appearing on both sides. Label-horizon
    truncation (below) is what prevents that.
    """
    sp = CFG.split
    out = pd.Series(pd.NA, index=months.index, dtype="string")
    out = out.mask(months <= pd.Timestamp(sp.train_end), TRAIN)
    out = out.mask((months >= pd.Timestamp(sp.valid_start))
                   & (months <= pd.Timestamp(sp.valid_end)), VALID)
    out = out.mask((months >= pd.Timestamp(sp.holdout_2022_start))
                   & (months <= pd.Timestamp(sp.holdout_2022_end)), HOLDOUT)
    out = out.mask((months >= test_start) & (months <= test_end), TEST)
    return out


def _truncate_train_labels(df: pd.DataFrame, split: pd.Series) -> pd.Series:
    """
    Enforce dataset_usage.md S8.3: drop training rows whose forward label window
    crosses the training cutoff.

    Without this, a 2020-06 training row's 12-month default label would be
    computed from 2020-07..2021-06 data -- which lies inside the VALIDATION
    window. That is temporal leakage, and it is the specific failure mode ADR-2
    calls a disqualification condition.
    """
    train_end = pd.Timestamp(CFG.split.train_end)
    horizon = CFG.rules.horizon_default
    # Last safe training month = train_end minus the longest label horizon.
    safe_end = (train_end - pd.DateOffset(months=horizon)).replace(day=1)
    unsafe = (split == TRAIN) & (df["reporting_month"] > safe_end)
    return unsafe


def process_shard(
    quarter: str,
    data_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    writers: _MultiWriter,
    servicer_updates: pd.DataFrame | None,
    keep_feature_table: bool = True,
) -> dict:
    """Run the full chain for one quarterly shard."""
    mp = SHARD_DIR / f"monthly_{quarter}.parquet"
    sp = SHARD_DIR / f"static_{quarter}.parquet"
    if not mp.exists() or not sp.exists():
        return {"quarter": quarter, "ok": False, "error": "shard missing"}

    panel = pd.read_parquet(mp)
    static = pd.read_parquet(sp)

    # 1. clean --------------------------------------------------------------
    panel, quarantine = clean_panel(panel)
    if len(quarantine):
        writers.write("quarantine", PROCESSED_DIR / "quarantine.parquet",
                      quarantine.assign(source_quarter=quarter))

    # 2. validation rules ---------------------------------------------------
    su = None
    if servicer_updates is not None and len(servicer_updates):
        ids = set(panel["loan_id"].unique())
        su = servicer_updates[servicer_updates["loan_id"].astype("string").isin(ids)]
    flags = apply_rules(panel.reset_index(drop=True), static, su)
    summary = batch_summary(flags)

    dq_cols = ["dq_score", "dq_penalty", "n_hard_failures", "n_soft_failures",
               "exception_type", "reason"]
    panel = panel.reset_index(drop=True)
    for c in dq_cols:
        panel[c] = flags[c].values
    rule_cols = [c for c in flags.columns if c.startswith("R")]
    dq_out = panel[["loan_id", "reporting_month"]].copy()
    for c in dq_cols:
        dq_out[c] = flags[c].values
    for c in rule_cols:
        dq_out[c] = flags[c].values
    writers.write("dq", PROCESSED_DIR / "dq_scores.parquet",
                  dq_out.assign(source_quarter=quarter))
    del flags, dq_out
    gc.collect()

    # 3. features + targets -------------------------------------------------
    feat = build_features(panel, static, data_end, with_targets=True)
    del panel, static
    gc.collect()

    if keep_feature_table:
        writers.write("features", FEATURES_DIR / "engineered_features.parquet", feat)

    # 4. route to splits ----------------------------------------------------
    split = assign_split(feat["reporting_month"], test_start, test_end)
    unsafe = _truncate_train_labels(feat, split)
    n_truncated = int(unsafe.sum())
    split = split.mask(unsafe, pd.NA)
    feat["split"] = split

    counts = {}
    for name, path in ((TRAIN, SPLITS_DIR / "train.parquet"),
                       (VALID, SPLITS_DIR / "validation.parquet"),
                       (TEST, SPLITS_DIR / "test.parquet"),
                       (HOLDOUT, SPLITS_DIR / "holdout_2022.parquet")):
        sub = feat[feat["split"] == name]
        writers.write(name, path, sub)
        counts[name] = len(sub)

    res = {
        "quarter": quarter,
        "ok": True,
        "rows_in": int(len(feat)),
        "rows_truncated_at_boundary": n_truncated,
        "split_counts": counts,
        "dq": summary,
        "n_features": len(feature_columns(feat)),
    }
    del feat
    gc.collect()
    return res


def build_all(keep_feature_table: bool = True) -> dict:
    """Process every shard and assemble the modelling datasets."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for p in (list(SPLITS_DIR.glob("*.parquet"))
              + [FEATURES_DIR / "engineered_features.parquet",
                 PROCESSED_DIR / "quarantine.parquet",
                 PROCESSED_DIR / "dq_scores.parquet"]):
        if p.exists():
            p.unlink()

    data_end = resolve_data_end()
    test_start, test_end = resolve_test_window(data_end)
    servicer_updates = load_servicer_updates()

    quarters = sorted(p.stem.replace("monthly_", "")
                      for p in SHARD_DIR.glob("monthly_*.parquet"))
    writers = _MultiWriter()
    results: list[dict] = []

    print(f"data_end={data_end.date()}  "
          f"test_window={test_start.date()}..{test_end.date()}  "
          f"shards={len(quarters)}", flush=True)

    for q in quarters:
        r = process_shard(q, data_end, test_start, test_end, writers,
                          servicer_updates, keep_feature_table)
        results.append(r)
        if r["ok"]:
            sc = r["split_counts"]
            print(f"[ok] {q}  rows={r['rows_in']:>9,}  "
                  f"train={sc[TRAIN]:>8,} val={sc[VALID]:>7,} "
                  f"test={sc[TEST]:>7,} ho22={sc[HOLDOUT]:>7,}  "
                  f"trunc={r['rows_truncated_at_boundary']:>7,}  "
                  f"dq={r['dq']['batch_dq_score']:.2f}", flush=True)
        else:
            print(f"[FAIL] {q}: {r.get('error')}", flush=True)
    writers.close()

    ok = [r for r in results if r["ok"]]
    summary = {
        "data_end": str(data_end.date()),
        "test_window": [str(test_start.date()), str(test_end.date())],
        "shards_ok": len(ok),
        "shards_failed": [r["quarter"] for r in results if not r["ok"]],
        "rows_total": sum(r["rows_in"] for r in ok),
        "rows_truncated_at_boundary": sum(r["rows_truncated_at_boundary"] for r in ok),
        "split_rows": writers.rows,
        "n_features": max((r["n_features"] for r in ok), default=0),
        "config_version": CFG.config_version,
        "feature_set_version": CFG.feature_set_version,
        "per_shard": results,
    }
    (REFERENCE_DIR / "feature_build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-feature-table", action="store_true",
                    help="skip writing the full engineered_features.parquet")
    a = ap.parse_args()
    s = build_all(keep_feature_table=not a.no_feature_table)
    print(json.dumps({k: v for k, v in s.items() if k != "per_shard"},
                     indent=2, default=str))

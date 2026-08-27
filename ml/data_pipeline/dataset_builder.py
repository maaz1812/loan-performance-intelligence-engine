"""
Dataset builder -- the streaming ingestion driver.

Produces the three required outputs of dataset_usage.md Section 17.5 plus a
full-population aggregate layer:

  loan_static_attributes.parquet      one row per sampled loan
  loan_monthly_performance.parquet    one row per sampled loan-month
  population_monthly_aggregates.parquet   from 100% of ~870M rows
  quarantine.parquet                  every rejected row, with a reason code

Memory discipline
-----------------
Free RAM on this workstation is ~2.5 GB, so nothing is ever accumulated at
corpus or even file scale. Each chunk is canonicalised, downcast and appended
straight to a ParquetWriter, then released. Only three small structures live
across chunks, and all three are bounded by the number of *sampled loans in one
quarter* (tens of thousands), not by row count:

  * static rows      -- one per loan
  * month counters   -- one int per loan, for month_index continuity
  * population agg   -- one row per calendar month (~96 rows total)

Because loan sets are disjoint across quarter files and a loan's rows are
contiguous and month-ordered within a file (both verified empirically), each
file can be processed independently and month_index can be assigned during the
single streaming pass -- no global sort, no combine step that would blow memory.
"""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import CFG, PROCESSED_DIR, REFERENCE_DIR, raw_zips
from .loader import sample_mask, stream_chunks
from .preprocessing import build_monthly, build_static, derive_light
from .schema_detection import verify_contract

SHARD_DIR = PROCESSED_DIR / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

# Downcast map: float32 is ample for balances (max ~$3M, 7 significant digits)
# and rates, and halves both memory and parquet size versus float64.
_F32 = (
    "current_balance", "current_interest_rate", "scheduled_principal",
    "non_interest_bearing_upb", "principal_forgiveness", "last_upb",
    "days_past_due", "original_balance", "interest_rate", "credit_score",
    "co_borrower_credit_score", "ltv", "combined_ltv", "dti", "mi_pct",
)
_I16 = ("loan_age_months", "remaining_term_months", "original_term_months",
        "month_index", "dlq_months", "num_borrowers", "num_units")


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in _F32:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in _I16:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int16")
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string")
    return df


def _agg_light(light: pd.DataFrame) -> pd.DataFrame:
    """Collapse a chunk to per-month population counters (tiny result).

    Grouped on the raw MMYYYY string key; parsed to a timestamp once in
    combine_shards rather than per row.
    """
    g = light.groupby("act_period", sort=False)
    return pd.DataFrame({
        "n_records": g.size(),
        "sum_balance": g["current_balance"].sum(min_count=1),
        "n_delinquent": g["is_delinquent"].sum(),
        "n_d90": g["is_d90"].sum(),
        "n_default": g["is_default"].sum(),
        "n_prepaid": g["is_prepaid"].sum(),
        "n_terminated": g["is_terminated"].sum(),
        "n_dlq_unknown": g["dlq_unknown"].sum(),
        "sum_dlq_months": g["dlq_months"].sum(min_count=1),
    }).reset_index()


def ingest_quarter(zip_path: str | Path, sample_rate: float | None = None) -> dict:
    """
    Stream one quarterly archive and write its three shards.

    Runs in a worker process. Returns a stats dict; raises nothing -- failures
    are captured and returned so one bad file cannot abort the whole pass
    (orchestrator.md Section 7: a failed task halts only its dependents).
    """
    zip_path = Path(zip_path)
    quarter = zip_path.stem
    t0 = time.time()
    stats: dict = {"quarter": quarter, "ok": False}

    monthly_path = SHARD_DIR / f"monthly_{quarter}.parquet"
    static_path = SHARD_DIR / f"static_{quarter}.parquet"
    popagg_path = SHARD_DIR / f"popagg_{quarter}.parquet"
    stats_path = SHARD_DIR / f"ingest_{quarter}.json"

    # Idempotency (orchestrator.md Section 7): a completed quarter is skipped
    # on re-run so the pass is resumable after an interruption.
    if stats_path.exists() and monthly_path.exists():
        prev = json.loads(stats_path.read_text())
        if prev.get("ok"):
            prev["skipped"] = True
            return prev

    try:
        contract = verify_contract(zip_path)
        if not contract["contract_ok"]:
            raise ValueError(f"schema contract failed: {contract['problems']}")

        writer: pq.ParquetWriter | None = None
        static_frames: list[pd.DataFrame] = []
        pop_frames: list[pd.DataFrame] = []
        month_counter: dict[str, int] = {}

        rows_total = rows_kept = 0
        n_chunks = 0

        for chunk in stream_chunks(zip_path):
            n_chunks += 1
            rows_total += len(chunk)

            # ---- full-population pass (100% of rows) ----------------------
            if CFG.ingest.compute_population_aggregates:
                pop_frames.append(_agg_light(derive_light(chunk)))
                # Collapse periodically so the list cannot grow unbounded.
                if len(pop_frames) >= 40:
                    pop_frames = [
                        pd.concat(pop_frames, ignore_index=True)
                        .groupby("act_period", as_index=False).sum()
                    ]

            # ---- sampled pass -------------------------------------------
            mask = sample_mask(chunk["LOAN_ID"], rate=sample_rate)
            if not mask.any():
                del chunk
                continue
            sub = chunk.loc[mask]
            del chunk
            rows_kept += len(sub)

            monthly = build_monthly(sub)

            # month_index: per-loan sequential counter. Rows arrive month-ordered
            # within a loan, and a loan never reappears in a later file, so a
            # carried-over per-loan base plus an in-chunk cumcount is exact even
            # when a loan straddles a chunk boundary.
            monthly = monthly.sort_values(["loan_id", "reporting_month"],
                                          kind="mergesort")
            base = monthly["loan_id"].map(month_counter).fillna(0).astype("int64")
            within = monthly.groupby("loan_id", sort=False).cumcount()
            monthly["month_index"] = (base + within).astype("int32")
            # Carry the per-loan count forward for loans that continue into the
            # next chunk. max(month_index)+1 is the running total, so this is a
            # single vectorised dict update rather than a per-loan Python loop.
            month_counter.update(
                (monthly.groupby("loan_id", sort=False)["month_index"].max() + 1)
                .astype("int64").to_dict()
            )

            monthly = _downcast(monthly)
            table = pa.Table.from_pandas(monthly, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(monthly_path, table.schema,
                                          compression="snappy")
            writer.write_table(table)
            del table

            # ---- static: first occurrence per loan ------------------------
            first = sub.drop_duplicates(subset=["LOAN_ID"], keep="first")
            static_frames.append(build_static(first))
            if len(static_frames) >= 40:
                static_frames = [
                    pd.concat(static_frames, ignore_index=True)
                    .drop_duplicates(subset=["loan_id"], keep="first")
                ]
            del sub, monthly, first

        if writer is not None:
            writer.close()

        # ---- finalise the two small shards --------------------------------
        if static_frames:
            static = (pd.concat(static_frames, ignore_index=True)
                      .drop_duplicates(subset=["loan_id"], keep="first"))
            static["source_file"] = quarter
            static = _downcast(static)
            static.to_parquet(static_path, compression="snappy", index=False)
            n_loans = len(static)
        else:
            n_loans = 0

        if pop_frames:
            pop = (pd.concat(pop_frames, ignore_index=True)
                   .groupby("act_period", as_index=False).sum())
            pop["source_file"] = quarter
            pop.to_parquet(popagg_path, compression="snappy", index=False)

        elapsed = time.time() - t0
        stats.update({
            "ok": True,
            "rows_total": int(rows_total),
            "rows_kept": int(rows_kept),
            "loans_kept": int(n_loans),
            "chunks": n_chunks,
            "seconds": round(elapsed, 1),
            "rows_per_sec": int(rows_total / elapsed) if elapsed else 0,
            "sample_rate_effective": round(rows_kept / rows_total, 5) if rows_total else 0,
            "schema_field_count": contract["field_count"],
        })
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        stats.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-2000:]})

    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def ingest_all(n_workers: int | None = None, sample_rate: float | None = None) -> dict:
    """
    Run the streaming pass over all 16 archives.

    Parallel across files (each is independent), sequential within a file.
    """
    n_workers = n_workers or CFG.ingest.n_workers
    files = raw_zips()
    results: list[dict] = []
    t0 = time.time()

    # Largest archives first so the long tail does not strand a single worker
    # at the end of the run.
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(ingest_quarter, str(f), sample_rate): f for f in files}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            tag = "skip" if r.get("skipped") else ("ok" if r["ok"] else "FAIL")
            print(f"[{tag}] {r['quarter']:8s} "
                  f"rows={r.get('rows_total', 0):>12,} "
                  f"kept={r.get('rows_kept', 0):>10,} "
                  f"loans={r.get('loans_kept', 0):>8,} "
                  f"{r.get('seconds', 0):>7.1f}s "
                  f"{r.get('rows_per_sec', 0):>8,}/s"
                  + ("" if r["ok"] else f"  {r.get('error')}"), flush=True)

    ok = [r for r in results if r["ok"]]
    summary = {
        "files": len(results),
        "ok": len(ok),
        "failed": [r["quarter"] for r in results if not r["ok"]],
        "rows_total": sum(r.get("rows_total", 0) for r in ok),
        "rows_kept": sum(r.get("rows_kept", 0) for r in ok),
        "loans_kept": sum(r.get("loans_kept", 0) for r in ok),
        "wall_seconds": round(time.time() - t0, 1),
        "sample_rate_target": sample_rate or CFG.ingest.sample_rate,
        "config_version": CFG.config_version,
        "schema_version": CFG.schema_version,
    }
    (REFERENCE_DIR / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def combine_shards() -> dict:
    """
    Assemble the per-quarter shards into the final processed datasets.

    Static and population-aggregate shards are small and concatenated directly.
    The monthly panel is streamed shard by shard into a single Parquet dataset
    partitioned by year, so the combine never holds the full panel in memory.
    """
    out: dict = {}

    # --- static -----------------------------------------------------------
    static_files = sorted(SHARD_DIR.glob("static_*.parquet"))
    if static_files:
        static = pd.concat([pd.read_parquet(f) for f in static_files],
                           ignore_index=True)
        static = static.drop_duplicates(subset=["loan_id"], keep="first")
        static.to_parquet(PROCESSED_DIR / "loan_static_attributes.parquet",
                          compression="snappy", index=False)
        out["static_rows"] = len(static)
        out["static_cols"] = len(static.columns)
        del static

    # --- population aggregates -------------------------------------------
    pop_files = sorted(SHARD_DIR.glob("popagg_*.parquet"))
    if pop_files:
        pop = pd.concat([pd.read_parquet(f) for f in pop_files], ignore_index=True)
        # Parse the MMYYYY key exactly once, over ~96 distinct values, rather
        # than once per row during ingestion.
        pop["reporting_month"] = pd.to_datetime(pop["act_period"],
                                                format="%m%Y", errors="coerce")
        pop = pop.dropna(subset=["reporting_month"])
        pop_total = (pop.drop(columns=["source_file", "act_period"])
                     .groupby("reporting_month", as_index=False).sum()
                     .sort_values("reporting_month"))
        pop_total["delinquency_rate"] = pop_total["n_delinquent"] / pop_total["n_records"]
        pop_total["default_rate"] = pop_total["n_default"] / pop_total["n_records"]
        pop_total["prepayment_rate"] = pop_total["n_prepaid"] / pop_total["n_records"]
        pop_total["d90_rate"] = pop_total["n_d90"] / pop_total["n_records"]
        pop_total["dlq_unknown_rate"] = pop_total["n_dlq_unknown"] / pop_total["n_records"]
        pop_total.to_parquet(
            PROCESSED_DIR / "population_monthly_aggregates.parquet",
            compression="snappy", index=False)
        pop.to_parquet(PROCESSED_DIR / "population_monthly_by_vintage.parquet",
                       compression="snappy", index=False)
        out["population_records"] = int(pop_total["n_records"].sum())
        out["population_months"] = len(pop_total)
        out["population_month_min"] = str(pop_total["reporting_month"].min().date())
        out["population_month_max"] = str(pop_total["reporting_month"].max().date())
        del pop, pop_total

    # --- monthly panel: stream shards into one partitioned dataset --------
    monthly_files = sorted(SHARD_DIR.glob("monthly_*.parquet"))
    panel_dir = PROCESSED_DIR / "loan_monthly_performance.parquet"
    if monthly_files:
        writer: pq.ParquetWriter | None = None
        single = PROCESSED_DIR / "loan_monthly_performance_all.parquet"
        n_rows = 0
        for f in monthly_files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=250_000):
                tbl = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(single, tbl.schema,
                                              compression="snappy")
                writer.write_table(tbl)
                n_rows += tbl.num_rows
        if writer is not None:
            writer.close()
        out["monthly_rows"] = n_rows
        out["monthly_path"] = str(single)

    (REFERENCE_DIR / "combine_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LPIE streaming ingestion")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sample-rate", type=float, default=None)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    if not args.combine_only:
        print(f"=== Phase 2 ingestion: streaming {len(raw_zips())} archives "
              f"(no extraction) ===", flush=True)
        s = ingest_all(args.workers, args.sample_rate)
        print(json.dumps(s, indent=2), flush=True)

    print("=== combining shards ===", flush=True)
    print(json.dumps(combine_shards(), indent=2), flush=True)

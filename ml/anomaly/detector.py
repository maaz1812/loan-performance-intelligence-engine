"""
Anomaly detection: deterministic rules + Isolation Forest.

Implements prd.md Section 4.3, dataset_usage.md Section 11 and
orchestrator.md Section 3.4.

Two-layer design, because the documentation is explicit that neither layer alone
is sufficient (dataset_usage.md Section 11.2):

  Rule layer      catches structurally impossible states. Deterministic,
                  explainable, auditable -- what a regulator wants to see.
                  Owned by ml/data_pipeline/validation.py (13 rules).

  ML layer        Isolation Forest over the engineered numeric space, catching
                  statistically unusual COMBINATIONS that no enumerated rule
                  anticipates.

Training population
-------------------
orchestrator.md Section 3.4: the detector is "trained on validated 'clean'
records, scored against the full population." That asymmetry matters -- fitting
on the full population would teach the forest that the anomalies are normal,
which is exactly backwards. Records with any HARD rule failure are therefore
excluded from the fit but still scored.

Score fusion
------------
The rule and ML signals are combined into one `anomaly_score` in [0,1] so the
reviewer queue is a single ranked list (api_spec.md `/anomalies` sorts by score).
Fusion is a weighted max rather than a mean: a record that catastrophically
fails one deterministic rule must not have its score diluted by looking
statistically ordinary on every other axis.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import CFG, PROCESSED_DIR, REFERENCE_DIR
from ..data_pipeline.validation import HARD_RULES, RULE_BY_ID, RULES
from ..models.registry import ModelMetadata, model_dir, next_version, register

DETECTOR_NAME = "anomaly_model"

# Numeric feature space for the forest. Deliberately excludes anything derived
# from a target, and excludes raw identifiers.
ANOMALY_FEATURES = (
    "current_balance", "loan_utilization", "balance_reduction_ratio",
    "amortisation_gap", "term_elapsed_pct", "rate_gap", "days_past_due",
    "dlq_months", "dpd_delta_1m", "dpd_mean_3m", "dpd_max_6m",
    "balance_delta_1m", "balance_pct_change_1m", "cum_delinquent_months",
    "loan_age_months", "remaining_term_months", "credit_risk_score",
    "debt_burden", "equity_cushion", "credit_score", "ltv", "dti",
    "interest_rate", "deferred_balance_pct", "scheduled_principal",
)


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in ANOMALY_FEATURES if c in df.columns]


def fit_detector(
    df: pd.DataFrame,
    features: list[str] | None = None,
    contamination: float | None = None,
) -> tuple[IsolationForest, list[str], dict]:
    """
    Fit Isolation Forest on rule-clean records only.

    Returns (model, features, fit_info).
    """
    cfg = CFG.anomaly
    features = features or available_features(df)
    contamination = contamination if contamination is not None else cfg.contamination

    clean_mask = pd.Series(True, index=df.index)
    if "n_hard_failures" in df.columns:
        clean_mask = df["n_hard_failures"].fillna(0) == 0

    clean = df.loc[clean_mask, features]
    # Isolation Forest cannot consume NaN; median-impute using ONLY the clean
    # subset's medians, and keep them so scoring uses identical values.
    medians = clean.median(numeric_only=True)
    clean_f = clean.fillna(medians).replace([np.inf, -np.inf], 0.0)

    if len(clean_f) > cfg.max_samples:
        clean_f = clean_f.sample(cfg.max_samples, random_state=cfg.random_seed)

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=min(cfg.max_samples, len(clean_f)),
        contamination=contamination,
        random_state=cfg.random_seed,
        n_jobs=CFG.model.n_jobs,
        bootstrap=False,
    )
    model.fit(clean_f)

    info = {
        "n_features": len(features),
        "features": features,
        "n_train_rows": int(len(clean_f)),
        "n_clean_available": int(clean_mask.sum()),
        "n_excluded_hard_failures": int((~clean_mask).sum()),
        "contamination": contamination,
        "medians": {k: (None if pd.isna(v) else float(v)) for k, v in medians.items()},
        "seed": cfg.random_seed,
    }
    return model, features, info


def score(
    df: pd.DataFrame,
    model: IsolationForest,
    features: list[str],
    medians: dict | None = None,
) -> pd.DataFrame:
    """
    Score every record and fuse the rule and ML signals.

    Returns a frame with ml_anomaly_score, rule_score, anomaly_score,
    exception_type, reason and the top contributing fields.
    """
    X = df[features].copy()
    if medians:
        X = X.fillna(pd.Series({k: v for k, v in medians.items() if v is not None}))
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # decision_function: higher = more normal. Invert and min-max to [0,1] so a
    # larger score always means "more anomalous", matching the API contract.
    raw = model.decision_function(X)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    ml = (hi - raw) / (hi - lo) if hi > lo else np.zeros_like(raw)

    out = pd.DataFrame(index=df.index)
    out["ml_anomaly_score"] = ml.astype("float32")

    # Rule score: the DQ penalty normalised. dq_penalty is a weighted sum of
    # fired rules capped at 100 in validation.py.
    if "dq_penalty" in df.columns:
        out["rule_score"] = (df["dq_penalty"].fillna(0) / 100.0).clip(0, 1).astype("float32")
    else:
        out["rule_score"] = np.float32(0.0)

    # Weighted max, not mean: a hard deterministic failure must dominate.
    out["anomaly_score"] = np.maximum(
        out["rule_score"], out["ml_anomaly_score"] * 0.85
    ).clip(0, 1).astype("float32")

    out["exception_type"] = df.get(
        "exception_type", pd.Series(pd.NA, index=df.index, dtype="string"))
    # Records the rules did not flag but the forest finds odd get their own
    # category, so the reviewer can tell rule-driven from statistical findings.
    stat_only = out["exception_type"].isna() & (out["ml_anomaly_score"] > 0.75)
    out.loc[stat_only, "exception_type"] = "statistical_outlier"

    out["reason"] = df.get("reason", pd.Series(pd.NA, index=df.index, dtype="string"))
    out["detector_version"] = f"if_{CFG.config_version}"
    out["n_hard_failures"] = df.get("n_hard_failures", 0)
    out["n_soft_failures"] = df.get("n_soft_failures", 0)
    return out


def top_contributing_fields(
    df: pd.DataFrame,
    features: list[str],
    idx: pd.Index,
    k: int = 4,
) -> list[list[dict]]:
    """
    Per-record drivers of the anomaly score, expressed as robust z-scores.

    prd.md Section 4.3 requires "the top contributing fields/features driving
    the anomaly score, in a reviewer-readable format." A median/MAD z-score is
    used rather than a mean/std one because the population contains the very
    outliers we are measuring, which would inflate a standard deviation and hide
    them.
    """
    sub = df.loc[:, features]
    med = sub.median(numeric_only=True)
    mad = (sub - med).abs().median(numeric_only=True).replace(0, np.nan)
    z = ((df.loc[idx, features] - med) / mad).abs()

    out: list[list[dict]] = []
    for _, row in z.iterrows():
        top = row.dropna().sort_values(ascending=False).head(k)
        out.append([
            {"field": f,
             "robust_z": round(float(v), 2),
             "value": (None if pd.isna(df.at[row.name, f])
                       else round(float(df.at[row.name, f]), 4)),
             "population_median": (None if pd.isna(med.get(f))
                                   else round(float(med[f]), 4))}
            for f, v in top.items()
        ])
    return out


RECOMMENDATION = {
    "balance_inconsistency": "Halt use of this record for risk decisioning. Reconcile current_balance against the servicer's balance file and confirm whether an undocumented modification occurred.",
    "balance_increase": "Verify whether a capitalisation or modification event occurred. If none is documented, treat the balance as unreliable and request a servicer correction.",
    "date_conflict": "Reject the record. A reporting month before origination indicates a mis-keyed date or a mis-joined loan id; re-ingest from source.",
    "delinquency_status_mismatch": "Do not rely on the delinquency field. Obtain the servicer's payment history and re-derive status before scoring this loan.",
    "negative_value": "Reject and re-ingest. A negative balance or term is structurally impossible and indicates a parsing or feed error.",
    "stale_update": "Confirm the servicer is still reporting on this loan. An unchanged balance over six or more months on an active loan suggests a stalled feed rather than a real payment pattern.",
    "status_flapping": "Investigate servicer reporting quality for this loan. Repeated Current/Delinquent alternation usually indicates a systems issue rather than genuine borrower behaviour.",
    "missing_status": "Request the missing delinquency status. Do not impute it as current -- an unreported status is not evidence of performance.",
    "impossible_term": "Reject the term fields. Remaining term cannot exceed the original term; verify the amortisation schedule.",
    "post_termination_activity": "Investigate the duplicate lifecycle. Activity after a terminal zero-balance event indicates either a reversed termination or a loan-id collision.",
    "amortisation_anomaly": "Confirm whether a large curtailment or partial prepayment occurred. If not, the balance movement is suspect.",
    "source_conflict": "Reconcile the two servicer feeds. Prefer the most recently timestamped source, and escalate if the conflict cannot be resolved.",
    "document_gap": "Request the missing monthly records. Gaps break rolling features and understate delinquency history.",
    "statistical_outlier": "No deterministic rule fired, but this record's combination of balance, delinquency and risk attributes is unusual versus the portfolio. Review manually before relying on its prediction.",
}


def build_examples(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    features: list[str],
    n: int | None = None,
) -> pd.DataFrame:
    """
    Assemble reviewer-ready anomaly examples.

    implementation.md Phase 5 and the master specification both require at least
    20 examples with drivers and recommendations. Examples are drawn to span
    exception types rather than simply taking the global top-N, because 20 copies
    of the same failure mode is not a useful reviewer artifact.
    """
    n = n or CFG.anomaly.min_examples_in_report
    joined = df.join(scores[["anomaly_score", "ml_anomaly_score", "rule_score",
                             "exception_type", "reason"]], rsuffix="_s")
    flagged = joined[joined["anomaly_score"] > 0].copy()
    if flagged.empty:
        return pd.DataFrame()

    # Stratify: take the strongest few from every exception type, then top up
    # from the global ranking until we have n.
    per_type = max(2, n // max(flagged["exception_type"].nunique(), 1))
    picks = (flagged.sort_values("anomaly_score", ascending=False)
             .groupby("exception_type", dropna=False, group_keys=False)
             .head(per_type))
    if len(picks) < n:
        extra = (flagged.drop(index=picks.index, errors="ignore")
                 .sort_values("anomaly_score", ascending=False)
                 .head(n - len(picks)))
        picks = pd.concat([picks, extra])
    picks = picks.sort_values("anomaly_score", ascending=False).head(max(n, 20))

    drivers = top_contributing_fields(joined, features, picks.index)
    picks = picks.reset_index(drop=True)
    picks["top_drivers"] = drivers
    picks["affected_fields"] = [
        ", ".join(d["field"] for d in dv) for dv in drivers
    ]
    picks["recommendation"] = picks["exception_type"].map(
        lambda t: RECOMMENDATION.get(t, "Review manually before relying on this record.")
    )
    return picks


def run(
    sample_rows: int = 400_000,
    persist: bool = True,
) -> dict:
    """
    Full anomaly pass over the engineered feature table.

    Reads a bounded sample for fitting/scoring so the pass stays inside the
    available memory envelope, then persists scores and examples.
    """
    from ..config import FEATURES_DIR
    feat_path = FEATURES_DIR / "engineered_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"missing {feat_path}; run the feature build first")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(feat_path)
    total = pf.metadata.num_rows

    need = set(ANOMALY_FEATURES) | {
        "loan_id", "reporting_month", "current_status", "dq_score", "dq_penalty",
        "exception_type", "reason", "n_hard_failures", "n_soft_failures",
        "original_balance", "state", "credit_score_band", "servicer_name",
        "modification_flag", "is_terminated", "vintage_year",
    }
    cols = sorted(need & set(pf.schema_arrow.names))

    # Deterministic stride so the sample is reproducible and spans all months.
    frames, taken = [], 0
    stride = max(1, total // max(sample_rows, 1))
    for i, batch in enumerate(pf.iter_batches(batch_size=200_000, columns=cols)):
        b = batch.to_pandas()
        if stride > 1:
            b = b.iloc[::stride]
        frames.append(b)
        taken += len(b)
        if taken >= sample_rows:
            break
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    model, features, info = fit_detector(df)
    scores = score(df, model, features, info["medians"])
    examples = build_examples(df, scores, features)

    md = ModelMetadata(
        model_name=DETECTOR_NAME, version=next_version(DETECTOR_NAME),
        algorithm="isolation_forest", target="anomaly", task="unsupervised",
        features=features, n_train_rows=info["n_train_rows"],
        hyperparameters={"n_estimators": CFG.anomaly.n_estimators,
                         "contamination": info["contamination"],
                         "max_samples": CFG.anomaly.max_samples},
        metrics={"scored_rows": int(len(df)),
                 "flagged_rate": float((scores["anomaly_score"] > 0.5).mean()),
                 "mean_score": float(scores["anomaly_score"].mean())},
        notes=("Fitted on rule-clean records only, scored against the full "
               "population, per orchestrator.md Section 3.4."),
    )
    register(model, md)
    md_dir = model_dir(DETECTOR_NAME, md.version)
    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / "imputation_medians.json").write_text(json.dumps(info["medians"], indent=2))

    result = {
        "scored_rows": int(len(df)),
        "total_rows_available": int(total),
        "sampling_stride": stride,
        "fit": {k: v for k, v in info.items() if k != "medians"},
        "flagged_any": int((scores["anomaly_score"] > 0).sum()),
        "flagged_high": int((scores["anomaly_score"] > 0.5).sum()),
        "n_examples": int(len(examples)),
        "by_exception_type": scores["exception_type"].value_counts(dropna=False)
                             .head(20).to_dict(),
        "registry": {"model_name": md.model_name, "version": md.version},
    }

    if persist:
        out = df[["loan_id", "reporting_month"]].join(scores)
        out.to_parquet(PROCESSED_DIR / "anomaly_results.parquet",
                       compression="snappy", index=False)
        if len(examples):
            keep = [c for c in ("loan_id", "reporting_month", "anomaly_score",
                                "ml_anomaly_score", "rule_score", "exception_type",
                                "reason", "affected_fields", "recommendation",
                                "current_status", "current_balance",
                                "original_balance", "days_past_due", "state",
                                "credit_score_band", "servicer_name")
                    if c in examples.columns]
            examples[keep].to_parquet(PROCESSED_DIR / "anomaly_examples.parquet",
                                      compression="snappy", index=False)
            examples[keep + ["top_drivers"]].to_json(
                REFERENCE_DIR / "anomaly_examples.json",
                orient="records", indent=2, date_format="iso")
        (REFERENCE_DIR / "anomaly_summary.json").write_text(
            json.dumps(result, indent=2, default=str))
    return result


__all__ = [
    "ANOMALY_FEATURES", "available_features", "fit_detector", "score",
    "top_contributing_fields", "build_examples", "run", "RECOMMENDATION",
    "DETECTOR_NAME",
]

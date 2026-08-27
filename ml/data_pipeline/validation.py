"""
Deterministic validation rule engine.

Implements orchestrator.md Section 3.2 and dataset_usage.md Sections 5.4 /
11.1-11.2. Rules are defined ONCE, in a machine-readable JSON contract
(data/supporting/validation_rules.json), and reused identically by:

  * ingestion / data-quality scoring   (this module)
  * the anomaly detection layer        (ml/anomaly/)
  * the backend AnomalyService         (backend/app/services/)

Defining them once and reusing them is the explicit requirement of
dataset_usage.md Section 13.2: "ensuring rule logic is defined once and reused
everywhere rather than duplicated in code."

Two non-negotiable behaviours:

  1. NO SILENT DROPS. A record that fails a rule is tagged with an
     `exception_type` and a human-readable `reason`, and is routed onward to the
     anomaly layer. It is never deleted (orchestrator.md Section 3.2).
  2. Rule severity is separated from rule truth. `hard` failures are
     structurally impossible states; `soft` failures are suspicious but
     legitimate under some conditions. Only hard failures quarantine a row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CFG, SUPPORTING_DIR

RULES_PATH = SUPPORTING_DIR / "validation_rules.json"

HARD = "hard"
SOFT = "soft"


@dataclass(frozen=True)
class Rule:
    id: str
    exception_type: str
    severity: str
    description: str
    reason_template: str
    weight: float          # contribution to the record-level DQ penalty


# ---------------------------------------------------------------------------
# The rule set. Each rule maps to a concrete vectorised check in `apply_rules`.
# ---------------------------------------------------------------------------
RULES: tuple[Rule, ...] = (
    Rule(
        id="R001",
        exception_type="balance_inconsistency",
        severity=HARD,
        description="current_balance exceeds original_balance, which is impossible under standard amortisation without a documented modification",
        reason_template="current_balance {current_balance:,.2f} exceeds original_balance {original_balance:,.2f} with no modification_flag",
        weight=25.0,
    ),
    Rule(
        id="R002",
        exception_type="balance_increase",
        severity=SOFT,
        description="current_balance increased month-over-month without a modification flag",
        reason_template="current_balance rose by {balance_delta:,.2f} versus the prior month without a modification_flag",
        weight=15.0,
    ),
    Rule(
        id="R003",
        exception_type="date_conflict",
        severity=HARD,
        description="reporting_month precedes origination_month",
        reason_template="reporting_month {reporting_month} precedes origination_month {origination_month}",
        weight=25.0,
    ),
    Rule(
        id="R004",
        exception_type="delinquency_status_mismatch",
        severity=HARD,
        description="days_past_due is zero while current_status is Delinquent, or non-zero while status is Current",
        reason_template="current_status '{current_status}' is inconsistent with days_past_due {days_past_due:.0f}",
        weight=20.0,
    ),
    Rule(
        id="R005",
        exception_type="negative_value",
        severity=HARD,
        description="a balance or term field is negative",
        reason_template="negative value detected: current_balance={current_balance:,.2f}, remaining_term_months={remaining_term_months}",
        weight=25.0,
    ),
    Rule(
        id="R006",
        exception_type="stale_update",
        severity=SOFT,
        description="the loan's reported balance has not changed for 6 or more consecutive months while the loan is active and interest-bearing",
        reason_template="balance unchanged for {stale_months:.0f} consecutive months while status is '{current_status}'",
        weight=10.0,
    ),
    Rule(
        id="R007",
        exception_type="status_flapping",
        severity=SOFT,
        description="the loan alternated between Current and Delinquent three or more times within a rolling 12-month window",
        reason_template="status changed {n_transitions:.0f} times in the trailing 12 months, indicating unstable servicer reporting",
        weight=10.0,
    ),
    Rule(
        id="R008",
        exception_type="missing_status",
        severity=SOFT,
        description="the servicer did not report a delinquency status (DLQ_STATUS = 'XX' or blank) on an active loan",
        reason_template="delinquency status was not reported by the servicer for this month",
        weight=8.0,
    ),
    Rule(
        id="R009",
        exception_type="impossible_term",
        severity=HARD,
        description="remaining_term_months exceeds the original loan term",
        reason_template="remaining_term_months {remaining_term_months} exceeds original_term_months {original_term_months}",
        weight=20.0,
    ),
    Rule(
        id="R010",
        exception_type="post_termination_activity",
        severity=HARD,
        description="a performance record exists for a month after the loan reached a terminal zero-balance state",
        reason_template="record dated {reporting_month} occurs after termination on {zero_balance_date}",
        weight=25.0,
    ),
    Rule(
        id="R011",
        exception_type="amortisation_anomaly",
        severity=SOFT,
        description="the balance fell by more than 40% in a single month without the loan terminating, which is inconsistent with scheduled amortisation",
        reason_template="balance fell {balance_drop_pct:.1f}% in one month without a zero-balance termination",
        weight=12.0,
    ),
    Rule(
        id="R012",
        exception_type="source_conflict",
        severity=SOFT,
        description="the secondary servicer feed reports a different status or balance than the primary performance record for the same loan-month",
        reason_template="servicer_updates reports status '{conflict_status}' versus primary '{current_status}'",
        weight=18.0,
    ),
    Rule(
        id="R013",
        exception_type="document_gap",
        severity=SOFT,
        description="a month is missing from the loan's otherwise continuous monthly reporting sequence",
        reason_template="gap of {gap_months:.0f} months detected in the monthly reporting sequence",
        weight=10.0,
    ),
)

RULE_BY_ID = {r.id: r for r in RULES}
HARD_RULES = tuple(r.id for r in RULES if r.severity == HARD)
SOFT_RULES = tuple(r.id for r in RULES if r.severity == SOFT)


def write_rules_json(path: Path | None = None) -> Path:
    """
    Materialise the rule set as the machine-readable contract referenced
    throughout the documentation as `validation_rules.json`.

    dataset_usage.md Section 13 designates this a SYNTHETIC supporting file to
    be generated, so the header declares it as such rather than passing it off
    as delivered data.
    """
    path = path or RULES_PATH
    payload = {
        "_meta": {
            "file": "validation_rules.json",
            "status": "GENERATED - synthetic supporting dataset",
            "authorised_by": "dataset_usage.md Section 13.2",
            "description": (
                "Machine-readable deterministic data-quality rules applied "
                "uniformly across ingestion, data-quality scoring and anomaly "
                "detection. Defined once here and reused everywhere."
            ),
            "config_version": CFG.config_version,
            "n_rules": len(RULES),
            "severity_semantics": {
                "hard": "structurally impossible state; quarantines the record",
                "soft": "suspicious but possibly legitimate; flags for review only",
            },
        },
        "thresholds": {
            "delinquency_dlq_months": CFG.rules.delinquency_dlq_months,
            "default_dlq_months": CFG.rules.default_dlq_months,
            "stale_months_threshold": 6,
            "flapping_transitions_threshold": 3,
            "balance_drop_pct_threshold": 40.0,
        },
        "rules": [asdict(r) for r in RULES],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_rules(path: Path | None = None) -> dict:
    path = path or RULES_PATH
    if not path.exists():
        write_rules_json(path)
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def apply_rules(
    panel: pd.DataFrame,
    static: pd.DataFrame | None = None,
    servicer_updates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Evaluate every rule against the monthly panel.

    `panel` must be sorted by (loan_id, reporting_month) -- lag-based rules are
    otherwise meaningless. Returns a frame indexed like `panel` with one boolean
    column per rule id, plus `dq_penalty`, `dq_score`, `n_hard_failures`,
    `n_soft_failures`, `exception_type` and `reason`.

    All checks are vectorised; dataset_usage.md Section 17.8 forbids row loops.
    """
    df = panel
    if static is not None:
        cols = [c for c in ("loan_id", "original_balance", "origination_month",
                            "original_term_months") if c in static.columns]
        df = df.merge(static[cols], on="loan_id", how="left")

    n = len(df)
    idx = df.index
    flags = pd.DataFrame(index=idx)

    grp = df.groupby("loan_id", sort=False)
    bal = pd.to_numeric(df.get("current_balance"), errors="coerce")
    orig_bal = pd.to_numeric(df.get("original_balance", pd.Series(np.nan, index=idx)),
                             errors="coerce")
    prev_bal = grp["current_balance"].shift(1)
    mod = df.get("modification_flag", pd.Series(False, index=idx)).fillna(False).astype(bool)
    dpd = pd.to_numeric(df.get("days_past_due"), errors="coerce").fillna(0)
    status = df.get("current_status", pd.Series("Current", index=idx)).astype("string")
    terminated = df.get("is_terminated", pd.Series(False, index=idx)).fillna(False).astype(bool)

    # R001 current_balance > original_balance without a modification
    flags["R001"] = (bal.notna() & orig_bal.notna() & (bal > orig_bal * 1.0001) & ~mod)

    # R002 month-over-month balance increase without a modification
    delta = bal - prev_bal
    flags["R002"] = (delta.notna() & (delta > 1.0) & ~mod & ~terminated)

    # R003 reporting_month before origination_month
    if "origination_month" in df.columns:
        flags["R003"] = (df["reporting_month"] < df["origination_month"])
    else:
        flags["R003"] = pd.Series(False, index=idx)

    # R004 status / days_past_due disagreement
    flags["R004"] = (
        (status.eq("Delinquent") & dpd.le(0))
        | (status.eq("Current") & dpd.gt(0))
    )

    # R005 negative balance or term
    rem = pd.to_numeric(df.get("remaining_term_months"), errors="coerce")
    flags["R005"] = (bal.lt(0).fillna(False) | rem.lt(0).fillna(False))

    # R006 stale balance on an active loan
    unchanged = (bal.sub(prev_bal).abs() < 0.01) & bal.notna() & prev_bal.notna()
    # consecutive-run length of `unchanged` within each loan
    blocks = (~unchanged).groupby(df["loan_id"], sort=False).cumsum()
    run_len = unchanged.groupby([df["loan_id"], blocks]).cumsum()
    flags["R006"] = (run_len >= 6) & ~terminated
    df["_stale_months"] = run_len

    # R007 status flapping within a trailing 12-month window
    changed = status.ne(grp["current_status"].shift(1)) & status.notna()
    n_trans = (changed.fillna(False).astype("int8")
               .groupby(df["loan_id"], sort=False)
               .rolling(12, min_periods=1).sum()
               .reset_index(level=0, drop=True))
    flags["R007"] = n_trans >= 3
    df["_n_transitions"] = n_trans

    # R008 servicer did not report a delinquency status
    flags["R008"] = (
        df.get("dlq_status_unreported", pd.Series(False, index=idx))
        .fillna(False).astype(bool) & ~terminated
    )

    # R009 remaining term exceeds original term
    if "original_term_months" in df.columns:
        ot = pd.to_numeric(df["original_term_months"], errors="coerce")
        flags["R009"] = (rem.notna() & ot.notna() & (rem > ot))
    else:
        flags["R009"] = pd.Series(False, index=idx)

    # R010 activity after a terminal zero-balance event
    zbd = df.get("zero_balance_date")
    if zbd is not None:
        first_term = grp["zero_balance_date"].transform("min")
        flags["R010"] = (first_term.notna()
                         & (df["reporting_month"] > first_term))
    else:
        flags["R010"] = pd.Series(False, index=idx)

    # R011 implausibly large single-month balance drop
    drop_pct = ((prev_bal - bal) / prev_bal.replace(0, np.nan)) * 100.0
    flags["R011"] = (drop_pct > 40.0) & ~terminated
    df["_balance_drop_pct"] = drop_pct

    # R012 conflict against the secondary servicer feed
    if servicer_updates is not None and len(servicer_updates):
        su = servicer_updates.rename(columns={
            "current_status": "conflict_status",
            "days_past_due": "conflict_dpd",
        })[["loan_id", "reporting_month", "conflict_status", "conflict_dpd"]]
        df = df.merge(su, on=["loan_id", "reporting_month"], how="left")
        flags = flags.reindex(df.index)
        flags["R012"] = (df["conflict_status"].notna()
                         & df["conflict_status"].ne(df["current_status"]))
        idx = df.index
    else:
        df["conflict_status"] = pd.NA
        flags["R012"] = pd.Series(False, index=idx)

    # R013 gap in the monthly reporting sequence
    prev_month = df.groupby("loan_id", sort=False)["reporting_month"].shift(1)
    gap = ((df["reporting_month"].dt.year - prev_month.dt.year) * 12
           + (df["reporting_month"].dt.month - prev_month.dt.month))
    flags["R013"] = gap.notna() & (gap > 1)
    df["_gap_months"] = gap

    flags = flags.fillna(False).astype(bool)

    # --- scoring ---------------------------------------------------------
    penalty = pd.Series(0.0, index=flags.index)
    for rid in flags.columns:
        penalty = penalty + flags[rid].astype("float32") * RULE_BY_ID[rid].weight
    flags["dq_penalty"] = penalty
    flags["dq_score"] = (100.0 - penalty).clip(lower=0.0)
    flags["n_hard_failures"] = flags[list(HARD_RULES)].sum(axis=1).astype("int8")
    flags["n_soft_failures"] = flags[list(SOFT_RULES)].sum(axis=1).astype("int8")

    # Primary exception type = the highest-weight rule that fired.
    ordered = sorted(RULES, key=lambda r: -r.weight)
    exc = pd.Series(pd.NA, index=flags.index, dtype="string")
    for r in ordered:
        exc = exc.mask(exc.isna() & flags[r.id], r.exception_type)
    flags["exception_type"] = exc
    flags["reason"] = _build_reasons(df, flags, ordered)
    return flags


def _build_reasons(df: pd.DataFrame, flags: pd.DataFrame,
                   ordered: list[Rule]) -> pd.Series:
    """
    Render a human-readable reason for the highest-severity rule that fired.

    prd.md Section 4.3 requires reviewer-readable anomaly explanations, and
    api_spec.md's /anomalies response carries a `reason` string.
    """
    reason = pd.Series(pd.NA, index=flags.index, dtype="string")
    ctx = {
        "current_balance": pd.to_numeric(df.get("current_balance"), errors="coerce"),
        "original_balance": pd.to_numeric(df.get("original_balance"), errors="coerce"),
        "remaining_term_months": pd.to_numeric(df.get("remaining_term_months"), errors="coerce"),
        "original_term_months": pd.to_numeric(df.get("original_term_months"), errors="coerce"),
        "days_past_due": pd.to_numeric(df.get("days_past_due"), errors="coerce"),
        "current_status": df.get("current_status"),
        "stale_months": df.get("_stale_months"),
        "n_transitions": df.get("_n_transitions"),
        "balance_drop_pct": df.get("_balance_drop_pct"),
        "gap_months": df.get("_gap_months"),
        "conflict_status": df.get("conflict_status"),
    }

    for r in ordered:
        mask = flags[r.id] & reason.isna()
        if not mask.any():
            continue
        sub = df.loc[mask]
        if r.id == "R002":
            bal = ctx["current_balance"][mask]
            prev = bal - 0  # delta recomputed below for message only
            delta = (bal - df.groupby("loan_id", sort=False)["current_balance"]
                     .shift(1)[mask])
            txt = delta.map(lambda v: f"current_balance rose by {v:,.2f} versus the prior month without a modification_flag")
        elif r.id == "R003":
            txt = (sub["reporting_month"].dt.strftime("%Y-%m") + " precedes origination "
                   + sub["origination_month"].dt.strftime("%Y-%m"))
            txt = "reporting_month " + txt
        elif r.id == "R010":
            txt = ("record dated " + sub["reporting_month"].dt.strftime("%Y-%m")
                   + " occurs after the loan reached a terminal zero-balance state")
        else:
            fields = {k: v for k, v in ctx.items() if v is not None}
            rows = pd.DataFrame({k: v[mask] for k, v in fields.items()
                                 if hasattr(v, "__getitem__")})

            def _fmt(row):
                try:
                    return r.reason_template.format(**{
                        k: (0 if pd.isna(row.get(k)) else row.get(k))
                        for k in row.index
                    })
                except (KeyError, ValueError, TypeError):
                    return r.description

            txt = rows.apply(_fmt, axis=1) if len(rows) else pd.Series(dtype="string")
        reason = reason.mask(mask, pd.Series(txt, index=sub.index).astype("string"))
    return reason


def batch_summary(flags: pd.DataFrame) -> dict:
    """Batch-level data-quality summary (orchestrator.md Section 3.2)."""
    n = len(flags)
    per_rule = {
        rid: {
            "exception_type": RULE_BY_ID[rid].exception_type,
            "severity": RULE_BY_ID[rid].severity,
            "n_failed": int(flags[rid].sum()),
            "pct_failed": round(float(flags[rid].mean()) * 100, 4),
        }
        for rid in (r.id for r in RULES) if rid in flags.columns
    }
    return {
        "n_records": n,
        "batch_dq_score": round(float(flags["dq_score"].mean()), 3),
        "records_with_any_failure": int((flags["dq_penalty"] > 0).sum()),
        "records_with_hard_failure": int((flags["n_hard_failures"] > 0).sum()),
        "pct_with_hard_failure": round(float((flags["n_hard_failures"] > 0).mean()) * 100, 4),
        "per_rule": per_rule,
    }


__all__ = [
    "Rule", "RULES", "RULE_BY_ID", "HARD_RULES", "SOFT_RULES", "HARD", "SOFT",
    "RULES_PATH", "write_rules_json", "load_rules", "apply_rules",
    "batch_summary",
]

"""
Generators for the three SYNTHETIC supporting datasets.

dataset_usage.md Section 13 ("Supporting Synthetic Datasets") specifies
`servicer_updates.csv`, `validation_rules.json` and `macro_scenarios.csv`,
describing each one's purpose and how it integrates with the core panel. None
of the three is present in the delivered data pack, and Section 13 explicitly
designates them as synthetic, so they are generated here rather than invented
ad hoc or silently skipped.

Every generated file carries a header/meta block declaring:
  * that it is GENERATED, not delivered
  * which document authorises it
  * the seed used, so it is byte-reproducible

`validation_rules.json` is produced by ml/data_pipeline/validation.py, which
owns the rule definitions; this module handles the other two.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import CFG, SUPPORTING_DIR

SERVICER_UPDATES_PATH = SUPPORTING_DIR / "servicer_updates.csv"
MACRO_SCENARIOS_PATH = SUPPORTING_DIR / "macro_scenarios.csv"
SUPPORTING_META_PATH = SUPPORTING_DIR / "SUPPORTING_DATA_README.md"

SEED = 20260101  # fixed and logged; reproducibility per dataset_usage.md S16


# ---------------------------------------------------------------------------
# 13.1 servicer_updates.csv
# ---------------------------------------------------------------------------

def generate_servicer_updates(
    panel: pd.DataFrame,
    coverage: float = 0.15,
    conflict_rate: float = 0.06,
    stale_rate: float = 0.05,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Build a second, partially-overlapping servicing feed.

    Purpose per Section 13.1: source-conflict detection and data reconciliation
    -- simulating the real situation where multiple servicing systems report on
    the same loan and disagree.

    Construction:
      * `coverage` of loan-months are echoed from the primary panel, so the two
        sources overlap only partially (a full copy would make conflict
        detection trivial and unrealistic).
      * `conflict_rate` of those rows get a DIFFERENT status/days_past_due,
        producing genuine irreconcilable conflicts for rule R012.
      * `stale_rate` of rows carry an old `last_updated_at`, producing the
        stale-record condition the documentation asks the anomaly layer to catch.

    Conflicts are injected as *plausible* disagreements (a neighbouring
    delinquency bucket, not a random value), because a detector that only finds
    absurd conflicts would not transfer to real reconciliation work.
    """
    rng = np.random.default_rng(seed)
    cols = ["loan_id", "reporting_month", "current_status", "days_past_due",
            "current_balance"]
    have = [c for c in cols if c in panel.columns]
    n = len(panel)
    take = rng.random(n) < coverage
    su = panel.loc[take, have].copy().reset_index(drop=True)
    if su.empty:
        return su

    m = len(su)
    su["source_system"] = rng.choice(
        ["SERVICER_FEED_A", "SERVICER_FEED_B", "LEGACY_MSP"], size=m,
        p=[0.55, 0.32, 0.13],
    )

    # --- staleness -------------------------------------------------------
    lag_months = np.where(rng.random(m) < stale_rate,
                          rng.integers(2, 7, size=m),
                          rng.integers(0, 2, size=m))
    su["last_updated_at"] = (
        su["reporting_month"] + pd.to_timedelta(lag_months * 30, unit="D")
    )
    su["reported_lag_months"] = lag_months.astype("int16")
    su["is_stale"] = (lag_months >= 2)

    # --- conflicts -------------------------------------------------------
    conflict = rng.random(m) < conflict_rate
    su["has_conflict"] = conflict

    status_ladder = ["Current", "Delinquent", "Default", "Prepaid", "Closed"]
    orig_status = su["current_status"].astype("string").fillna("Current")
    shifted = orig_status.map(
        lambda s: status_ladder[min(status_ladder.index(s) + 1, len(status_ladder) - 1)]
        if s in status_ladder else "Current"
    )
    su["current_status"] = orig_status.where(~conflict, shifted)

    if "days_past_due" in su.columns:
        dpd = pd.to_numeric(su["days_past_due"], errors="coerce").fillna(0)
        bumped = dpd + rng.choice([30, 60, -30], size=m, p=[0.5, 0.3, 0.2])
        su["days_past_due"] = dpd.where(~conflict, bumped.clip(lower=0))

    if "current_balance" in su.columns:
        bal = pd.to_numeric(su["current_balance"], errors="coerce")
        jitter = 1.0 + rng.normal(0, 0.004, size=m)
        su["current_balance"] = np.where(conflict, bal * jitter, bal)

    su["_generated"] = "synthetic"
    return su


def write_servicer_updates(panel: pd.DataFrame, **kw) -> dict:
    su = generate_servicer_updates(panel, **kw)
    header = (
        "# GENERATED SYNTHETIC DATASET - servicer_updates.csv\n"
        "# Authorised by dataset_usage.md Section 13.1 (Supporting Synthetic Datasets).\n"
        "# NOT delivered source data. Purpose: source-conflict detection and\n"
        "# data reconciliation testing against the primary performance panel.\n"
        f"# seed={SEED} rows={len(su)} config_version={CFG.config_version}\n"
    )
    with open(SERVICER_UPDATES_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        su.to_csv(fh, index=False)
    return {
        "path": str(SERVICER_UPDATES_PATH),
        "rows": len(su),
        "conflicts": int(su["has_conflict"].sum()) if len(su) else 0,
        "stale": int(su["is_stale"].sum()) if len(su) else 0,
    }


# ---------------------------------------------------------------------------
# 13.3 macro_scenarios.csv
# ---------------------------------------------------------------------------

# Multipliers/shifts applied to model inputs and outputs for each scenario.
# Magnitudes are anchored to observable stress episodes rather than invented:
# the adverse-credit case is scaled to the 2020 COVID delinquency spike visible
# in this dataset's own population aggregates, and the high-prepayment case to
# the 2020-2021 refinance wave.
SCENARIO_ASSUMPTIONS: tuple[dict, ...] = (
    {
        "scenario_name": "base",
        "description": "Current observed conditions; no macro adjustment applied. Reference point for comparison.",
        "unemployment_shift_bps": 0,
        "hpi_shock_pct": 0.0,
        "rate_shift_bps": 0,
        "delinquency_multiplier": 1.00,
        "default_multiplier": 1.00,
        "prepayment_multiplier": 1.00,
        "dpd_stress_months": 0.0,
        "ltv_stress_pct": 0.0,
        "rationale": "Baseline: models scored on unmodified current feature distributions.",
    },
    {
        "scenario_name": "adverse_credit",
        "description": "Stressed credit environment: rising unemployment, falling house prices, elevated payment stress.",
        "unemployment_shift_bps": 300,
        "hpi_shock_pct": -15.0,
        "rate_shift_bps": 100,
        "delinquency_multiplier": 2.10,
        "default_multiplier": 2.60,
        "prepayment_multiplier": 0.65,
        "dpd_stress_months": 1.0,
        "ltv_stress_pct": 15.0,
        "rationale": "Calibrated to the 2020 stress episode observed in this dataset's own population aggregates; a -15% HPI shock raises effective LTV by ~15pp, and prepayment falls as refinancing becomes uneconomic for stressed borrowers.",
    },
    {
        "scenario_name": "high_prepayment",
        "description": "Falling-rate environment driving a refinance wave and accelerated loan closure.",
        "unemployment_shift_bps": -50,
        "hpi_shock_pct": 6.0,
        "rate_shift_bps": -150,
        "delinquency_multiplier": 0.85,
        "default_multiplier": 0.75,
        "prepayment_multiplier": 2.40,
        "dpd_stress_months": 0.0,
        "ltv_stress_pct": -6.0,
        "rationale": "Calibrated to the 2020-2021 refinance wave; a 150bp rate decline makes refinancing attractive for most seasoned loans, concentrated in high-rate/high-credit segments.",
    },
)


def write_macro_scenarios() -> dict:
    df = pd.DataFrame(list(SCENARIO_ASSUMPTIONS))
    header = (
        "# GENERATED SYNTHETIC DATASET - macro_scenarios.csv\n"
        "# Authorised by dataset_usage.md Section 13.3 (Supporting Synthetic Datasets).\n"
        "# NOT delivered source data. Purpose: supplies the quantitative stress\n"
        "# assumptions parameterising the base / adverse_credit / high_prepayment\n"
        "# scenarios described in dataset_usage.md Section 12.\n"
        f"# config_version={CFG.config_version}\n"
    )
    with open(MACRO_SCENARIOS_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return {"path": str(MACRO_SCENARIOS_PATH), "scenarios": len(df)}


def load_macro_scenarios() -> pd.DataFrame:
    if not MACRO_SCENARIOS_PATH.exists():
        write_macro_scenarios()
    return pd.read_csv(MACRO_SCENARIOS_PATH, comment="#")


def load_servicer_updates() -> pd.DataFrame:
    if not SERVICER_UPDATES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SERVICER_UPDATES_PATH, comment="#",
                     parse_dates=["reporting_month", "last_updated_at"],
                     dtype={"loan_id": "string"})
    return df


def write_readme(stats: dict) -> None:
    SUPPORTING_META_PATH.write_text(
        "# Supporting Datasets - GENERATED, NOT DELIVERED\n\n"
        "The project documentation references three supporting files that are "
        "**not present** in the delivered data pack:\n\n"
        "- `servicer_updates.csv`\n- `validation_rules.json`\n- `macro_scenarios.csv`\n\n"
        "`dataset_usage.md` Section 13 titles these \"Supporting Synthetic "
        "Datasets\" and specifies each one's purpose and integration, so they are "
        "generated programmatically rather than fabricated ad hoc or skipped. "
        "Generation is seeded and byte-reproducible.\n\n"
        "| File | Authorised by | Purpose | Rows |\n|---|---|---|---|\n"
        f"| servicer_updates.csv | S13.1 | Source-conflict + staleness detection | {stats.get('servicer_updates', {}).get('rows', 0):,} |\n"
        f"| validation_rules.json | S13.2 | Deterministic DQ rules, defined once, reused everywhere | {stats.get('validation_rules', {}).get('n_rules', 0)} rules |\n"
        f"| macro_scenarios.csv | S13.3 | Stress assumptions for the 3 scenarios | {stats.get('macro_scenarios', {}).get('scenarios', 0)} |\n\n"
        "None of these files is used to produce a predictive probability. They "
        "supply reconciliation inputs, rule definitions and scenario parameters "
        "only.\n\n"
        f"Seed: `{SEED}` | config_version: `{CFG.config_version}`\n"
    )


def generate_all(panel: pd.DataFrame | None = None) -> dict:
    """Generate every supporting file. `panel` is required for servicer_updates."""
    from .validation import write_rules_json, RULES

    stats: dict = {}
    write_rules_json()
    stats["validation_rules"] = {"n_rules": len(RULES)}
    stats["macro_scenarios"] = write_macro_scenarios()
    if panel is not None and len(panel):
        stats["servicer_updates"] = write_servicer_updates(panel)
    write_readme(stats)
    return stats


__all__ = [
    "generate_servicer_updates", "write_servicer_updates", "write_macro_scenarios",
    "load_macro_scenarios", "load_servicer_updates", "generate_all",
    "SCENARIO_ASSUMPTIONS", "SERVICER_UPDATES_PATH", "MACRO_SCENARIOS_PATH",
    "SEED",
]

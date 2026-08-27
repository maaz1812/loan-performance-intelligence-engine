"""
Feature engineering and forward-looking target construction.

Implements dataset_usage.md Section 6 (four feature families) and Section 7
(five ML targets), under the leakage discipline of Section 8.3.

THE LEAKAGE CONTRACT
--------------------
Two rules govern every line in this module, because violating either is an
explicit disqualification condition (decision.md ADR-2):

  Features  use only information available at or before month t.
            Lag/rolling features are built with shift(+k) and trailing windows.
            No feature ever reads a row dated later than t.

  Targets   use only information strictly after month t.
            Built with shift(-k) over the loan's own future rows.

`assert_no_leakage` verifies the first rule empirically by checking that no
feature column correlates perfectly with a shifted future value, and
tests/test_features.py pins the target semantics.

LABEL COMPLETENESS
------------------
A row whose forward window extends past the end of the data has an UNKNOWABLE
label and must not be trained on. Section 8.3 requires such rows be "either
excluded or clearly documented as having a truncated label horizon." Each target
therefore ships with a companion `<target>_complete` mask, and only complete
rows are used for training and evaluation. A label is complete when either:

  * the full forward window fits inside the observed data, OR
  * the loan terminated inside the window (the outcome is then known regardless
    of how much calendar data remains)

WHY PER-SHARD
-------------
Loan sets are disjoint across quarterly files and a loan's rows are contiguous
within one file (both verified empirically). Every per-loan feature and target
is therefore exactly computable inside a single shard, which keeps peak memory
to one quarter rather than the whole ~26M-row panel -- essential at ~2.5 GB free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG

PANEL_KEYS = ["loan_id", "reporting_month"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Whole months from b to a (a - b), as a float series."""
    return ((a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)).astype("float32")


def _forward_any(df: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """
    1.0 if `col` is truthy in ANY of the loan's rows t+1 .. t+horizon.

    Built from explicit negative shifts rather than a reversed rolling window:
    the semantics are unambiguous and it is impossible to accidentally include
    row t itself, which would be leakage.
    """
    g = df.groupby("loan_id", sort=False)[col]
    acc = None
    for i in range(1, horizon + 1):
        sh = g.shift(-i).astype("float32")
        acc = sh if acc is None else np.fmax(acc, sh)
    return pd.Series(acc, index=df.index)


def _forward_rows_available(df: pd.DataFrame, horizon: int) -> pd.Series:
    """How many of the next `horizon` rows exist for this loan."""
    g = df.groupby("loan_id", sort=False)["reporting_month"]
    cnt = pd.Series(0.0, index=df.index)
    for i in range(1, horizon + 1):
        cnt = cnt + g.shift(-i).notna().astype("float32")
    return cnt


# ---------------------------------------------------------------------------
# 6.1 Financial features
# ---------------------------------------------------------------------------

def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    dataset_usage.md Section 6.1. Normalises balance information so loans of
    very different original sizes are comparable model inputs.
    """
    bal = df["current_balance"].astype("float32")
    orig = df["original_balance"].astype("float32")
    safe_orig = orig.replace(0, np.nan)

    df["loan_utilization"] = (bal / safe_orig).astype("float32")
    df["balance_reduction_ratio"] = ((orig - bal) / safe_orig).astype("float32")
    df["remaining_balance_pct"] = (df["loan_utilization"] * 100).astype("float32")

    # Amortisation progress relative to schedule: how far through the term the
    # loan is versus how much principal it has actually repaid. A loan that is
    # 40% through its term but has repaid only 10% of principal is under stress.
    term = df["original_term_months"].astype("float32").replace(0, np.nan)
    age = df["loan_age_months"].astype("float32")
    df["term_elapsed_pct"] = (age / term).astype("float32")
    df["amortisation_gap"] = (
        df["term_elapsed_pct"] - df["balance_reduction_ratio"]
    ).astype("float32")

    # Rate gap: current versus origination rate. Drives refinance incentive.
    df["rate_gap"] = (
        df["current_interest_rate"].astype("float32")
        - df["interest_rate"].astype("float32")
    ).astype("float32")

    df["deferred_balance_pct"] = (
        df["non_interest_bearing_upb"].astype("float32") / safe_orig
    ).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 6.2 Temporal features -- all trailing, never forward
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    dataset_usage.md Section 6.2. Lets the model see *trajectory* rather than a
    single-month snapshot: a loan deteriorating slowly over six months is a
    different risk from one that is stable then suddenly delinquent.

    Every window here is TRAILING and includes row t (which is known at time t)
    but never row t+1.
    """
    g = df.groupby("loan_id", sort=False)
    dpd = df["days_past_due"].astype("float32")
    delinq = df["delinquency"].astype("float32")

    df["dpd_lag_1"] = g["days_past_due"].shift(1).astype("float32")
    df["dpd_lag_3"] = g["days_past_due"].shift(3).astype("float32")
    df["dpd_delta_1m"] = (dpd - df["dpd_lag_1"]).astype("float32")

    for w in CFG.rules.rolling_windows:
        roll = (df.groupby("loan_id", sort=False)["days_past_due"]
                .rolling(w, min_periods=1))
        df[f"dpd_mean_{w}m"] = (roll.mean()
                                .reset_index(level=0, drop=True).astype("float32"))
        df[f"dpd_max_{w}m"] = (roll.max()
                               .reset_index(level=0, drop=True).astype("float32"))
        droll = (df.groupby("loan_id", sort=False)["delinquency"]
                 .rolling(w, min_periods=1))
        df[f"delinquent_months_{w}m"] = (droll.sum()
                                         .reset_index(level=0, drop=True).astype("float32"))

    # Cumulative history to date (expanding, so strictly as-of-t).
    df["cum_delinquent_months"] = (
        g["delinquency"].cumsum().astype("float32") - delinq
    ).astype("float32")
    df["ever_delinquent"] = (df["cum_delinquent_months"] > 0).astype("int8")
    df["cum_missed_payments"] = (
        g["dlq_months"].cummax().fillna(0).astype("float32")
    )

    # Balance trend: is principal actually coming down?
    bal = df["current_balance"].astype("float32")
    df["balance_delta_1m"] = (bal - g["current_balance"].shift(1)).astype("float32")
    df["balance_delta_3m"] = (bal - g["current_balance"].shift(3)).astype("float32")
    prev = g["current_balance"].shift(1).replace(0, np.nan)
    df["balance_pct_change_1m"] = (df["balance_delta_1m"] / prev).astype("float32")

    # Modification / forbearance history to date.
    df["months_since_modification"] = _months_since_flag(df, "modification_flag")
    df["ever_modified"] = (
        g["modification_flag"].cummax().fillna(False).astype("int8")
    )
    df["ever_forbearance"] = (
        g["forbearance_flag"].cummax().fillna(False).astype("int8")
    )
    df["forbearance_months"] = (
        g["forbearance_flag"].cumsum().astype("float32")
    )
    return df


def _months_since_flag(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Months since the flag was last true, as of row t inclusive.

    Implemented as (current month_index) - (last month_index where the flag was
    set, carried forward). Uses only past rows.
    """
    mi = df["month_index"].astype("float32")
    marked = mi.where(df[col].fillna(False).astype(bool))
    last = marked.groupby(df["loan_id"], sort=False).ffill()
    return (mi - last).astype("float32")


# ---------------------------------------------------------------------------
# 6.3 Risk features
# ---------------------------------------------------------------------------

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """dataset_usage.md Section 6.3."""
    cs = df["credit_score"].astype("float32")
    ltv = df["ltv"].astype("float32")
    dti = df["dti"].astype("float32")
    rate = df["interest_rate"].astype("float32")

    # Composite credit risk score on a 0-100 scale where higher = riskier.
    # Each component is normalised to [0,1] then weighted; weights reflect the
    # relative predictive strength reported in the mortgage-risk literature
    # (credit score strongest, then LTV, then DTI).
    cs_risk = ((850.0 - cs.clip(300, 850)) / 550.0).fillna(0.5)
    ltv_risk = (ltv.clip(0, 125) / 125.0).fillna(0.5)
    dti_risk = (dti.clip(0, 65) / 65.0).fillna(0.5)
    df["credit_risk_score"] = (
        (0.45 * cs_risk + 0.35 * ltv_risk + 0.20 * dti_risk) * 100.0
    ).astype("float32")

    # Debt burden: DTI scaled by the rate, approximating payment stress.
    df["debt_burden"] = (dti * rate / 100.0).astype("float32")

    # Equity cushion -- the inverse of LTV, the borrower's walk-away buffer.
    df["equity_cushion"] = (100.0 - ltv).astype("float32")

    # Loan-age risk. Mortgage hazard is non-monotonic: low in the first months,
    # peaking around years 2-5, then declining. A triangular kernel centred at
    # 36 months captures that shape without needing a spline.
    age = df["loan_age_months"].astype("float32")
    df["loan_age_risk"] = (1.0 - (age - 36.0).abs() / 60.0).clip(0, 1).astype("float32")
    df["is_seasoned"] = (age >= 36).astype("int8")

    df["high_ltv_flag"] = (ltv > 80).astype("int8")
    df["subprime_flag"] = (cs < 660).astype("int8")
    df["high_dti_flag"] = (dti > 43).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 6.4 Interaction features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    dataset_usage.md Section 6.4. Compounding-risk terms. Tree models can
    approximate interactions, but supplying them explicitly shortens the trees
    needed and makes the effect directly visible in SHAP.
    """
    cs = df["credit_score"].astype("float32")
    ltv = df["ltv"].astype("float32")
    dti = df["dti"].astype("float32")
    rate = df["interest_rate"].astype("float32")

    # Low credit AND high LTV is materially riskier than either alone, so the
    # interaction is expressed as risk x risk rather than raw x raw.
    df["credit_x_ltv"] = (((850.0 - cs) / 550.0) * (ltv / 100.0)).astype("float32")
    df["dti_x_rate"] = ((dti / 100.0) * rate).astype("float32")
    df["credit_x_dti"] = (((850.0 - cs) / 550.0) * (dti / 100.0)).astype("float32")

    # Delinquency history amplified by weak equity: a borrower who has missed
    # payments and has no equity has both motive and means to walk away.
    df["dlqhist_x_ltv"] = (
        df["cum_delinquent_months"].astype("float32") * (ltv / 100.0)
    ).astype("float32")

    # Stress interaction: a poor amortisation record on a high-rate loan.
    df["amortgap_x_rate"] = (
        df["amortisation_gap"].astype("float32") * rate
    ).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Section 7 -- forward-looking targets
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """
    Build the five targets of dataset_usage.md Section 7, each with a
    completeness mask.

    `data_end` is the last reporting month present anywhere in the corpus. It is
    passed in rather than inferred per shard, because completeness is a property
    of the whole dataset, not of one quarter.
    """
    rules = CFG.rules
    df = df.sort_values(PANEL_KEYS, kind="mergesort")

    months_left = _months_between(pd.Series(data_end, index=df.index), df["reporting_month"])

    # A terminal event inside the window makes the label knowable even when the
    # calendar window runs past the data end.
    for name, horizon, source in (
        ("next_3m_delinquency_flag", rules.horizon_delinquency_short, "delinquency"),
        ("next_6m_delinquency_flag", rules.horizon_delinquency_medium, "delinquency"),
        ("next_12m_default_flag", rules.horizon_default, "default_flag"),
        ("next_12m_prepayment_flag", rules.horizon_prepayment, "prepayment_flag"),
    ):
        fwd = _forward_any(df, source, horizon)
        df[name] = fwd.fillna(0.0).astype("int8")

        terminated_in_window = _forward_any(df, "is_terminated", horizon).fillna(0.0)
        window_fits = (months_left >= horizon)
        # A positive observation is self-certifying: if the event is seen, the
        # label is known regardless of remaining calendar coverage.
        observed_positive = fwd.fillna(0.0) > 0
        df[f"{name}_complete"] = (
            window_fits | (terminated_in_window > 0) | observed_positive
        ).astype("int8")

    # ---- next_state (multi-class) ---------------------------------------
    nxt = df.groupby("loan_id", sort=False)["current_status"].shift(-1)
    # On a loan's final row there is no t+1. If the loan terminated, its state
    # is absorbing and persists; otherwise the label is genuinely unknown.
    terminal = df["is_terminated"].fillna(False).astype(bool)
    nxt = nxt.where(nxt.notna(), df["current_status"].where(terminal))
    df["next_state"] = nxt.astype("string")
    df["next_state_complete"] = (
        df["next_state"].notna() & ((months_left >= 1) | terminal)
    ).astype("int8")

    # Alternative default definitions carried through for the report.
    if "default_flag_ex_forbearance" in df.columns:
        df["next_12m_default_ex_forbearance"] = (
            _forward_any(df, "default_flag_ex_forbearance", rules.horizon_default)
            .fillna(0.0).astype("int8")
        )
    if "default_flag_credit_event" in df.columns:
        df["next_12m_default_credit_event"] = (
            _forward_any(df, "default_flag_credit_event", rules.horizon_default)
            .fillna(0.0).astype("int8")
        )

    df["months_to_data_end"] = months_left
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

FEATURE_BUILDERS = (
    add_financial_features,
    add_temporal_features,
    add_risk_features,
    add_interaction_features,
)


def build_features(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    data_end: pd.Timestamp,
    with_targets: bool = True,
) -> pd.DataFrame:
    """
    Full feature pipeline for one shard: join static attributes onto the
    monthly panel, add all four feature families, then the targets.

    The join is done first because several features (loan_utilization,
    credit_risk_score, the interactions) need origination attributes.
    """
    static_cols = [c for c in static.columns if c != "source_file"]
    df = panel.merge(static[static_cols], on="loan_id", how="left")
    df = df.sort_values(PANEL_KEYS, kind="mergesort").reset_index(drop=True)

    for fn in FEATURE_BUILDERS:
        df = fn(df)

    if with_targets:
        df = add_targets(df, data_end)

    return df


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

# Columns that legitimately describe the future or the label; excluded from the
# leakage scan because they are targets, masks, or bookkeeping.
_TARGET_LIKE = (
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
    "next_12m_default_ex_forbearance", "next_12m_default_credit_event",
    "months_to_data_end",
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    The model-input columns.

    Deliberately EXCLUDES every target, every completeness mask, and the raw
    present-tense outcome fields (current_status, delinquency, default_flag,
    prepayment_flag, is_terminated, zero_balance_code, termination_reason).

    Those present-tense fields are not "future" data, but including them would
    make the task trivially circular for the delinquency targets and would leak
    termination knowledge into the prepayment model, so they are held out and
    only their *derived history* (cum_delinquent_months, dpd_mean_3m, ...) is
    offered to the model.
    """
    banned = set(_TARGET_LIKE) | {
        c for c in df.columns if c.endswith("_complete")
    } | {
        "loan_id", "reporting_month", "current_status", "delinquency",
        "default_flag", "prepayment_flag", "default_flag_ex_forbearance",
        "default_flag_credit_event", "is_terminated", "termination_reason",
        "zero_balance_code", "zero_balance_date", "last_paid_installment",
        "origination_month", "source_file", "last_upb",
        "dq_score", "dq_penalty", "exception_type", "reason",
    }
    out = []
    for c in df.columns:
        if c in banned:
            continue
        if df[c].dtype.kind in "biufc":
            out.append(c)
    return out


def assert_no_leakage(df: pd.DataFrame, sample: int = 200_000) -> dict:
    """
    Empirical leakage check.

    For each candidate feature, verify it is not a perfect function of a FUTURE
    value of itself -- which is the fingerprint of an accidentally
    forward-looking window. Also asserts that no held-out present-tense outcome
    column reached the feature list.

    Returns a report; raises AssertionError on a hard violation.
    """
    feats = feature_columns(df)
    forbidden = [c for c in feats if c in _TARGET_LIKE or c.endswith("_complete")]
    assert not forbidden, f"target-like columns leaked into features: {forbidden}"

    present_tense = {"delinquency", "default_flag", "prepayment_flag",
                     "current_status", "is_terminated"}
    overlap = present_tense & set(feats)
    assert not overlap, f"present-tense outcome columns leaked into features: {overlap}"

    sub = df if len(df) <= sample else df.sample(sample, random_state=CFG.model.random_seed)
    sub = sub.sort_values(PANEL_KEYS, kind="mergesort")

    suspicious: list[dict] = []
    for c in feats:
        s = pd.to_numeric(sub[c], errors="coerce")
        if s.notna().sum() < 100 or s.nunique(dropna=True) < 3:
            continue
        fwd = sub.groupby("loan_id", sort=False)[c].shift(-1)
        f = pd.to_numeric(fwd, errors="coerce")
        both = s.notna() & f.notna()
        if both.sum() < 100:
            continue
        # A feature that equals its own next-month value on essentially every
        # row is carrying future information backwards.
        if float((s[both] == f[both]).mean()) > 0.999:
            suspicious.append({"feature": c, "equals_next_month_rate": 1.0})

    return {
        "n_features": len(feats),
        "features": feats,
        "suspicious": suspicious,
        "checked_rows": len(sub),
    }


__all__ = [
    "add_financial_features", "add_temporal_features", "add_risk_features",
    "add_interaction_features", "add_targets", "build_features",
    "feature_columns", "assert_no_leakage", "FEATURE_BUILDERS", "PANEL_KEYS",
]

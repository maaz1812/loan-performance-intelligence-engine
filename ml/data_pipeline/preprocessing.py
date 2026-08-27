"""
Preprocessing and the raw -> canonical schema bridge.

This module implements dataset_usage.md Section 5 (missing values, duplicates,
type conversion, invalid-record detection) and the Gap 1 schema bridge recorded
in docs/documentation_analysis.md Section 3.

The most correctness-critical functions here are `derive_delinquency` and
`derive_status`, which turn the two raw source fields DLQ_STATUS and
Zero_Bal_Code into every downstream label:

    DLQ_STATUS     -> dlq_months -> days_past_due -> delinquency -> default
    Zero_Bal_Code  -> prepayment_flag / default_flag / terminal state

Getting these wrong would silently invalidate every model in the project, so
they are pure functions over vectors, and tests/test_targets.py asserts their
behaviour against hand-computed cases.

All operations are vectorised. dataset_usage.md Section 17.8 explicitly forbids
row-by-row Python loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG
from .schema_detection import (
    CHANNEL_MAP, DLQ_UNKNOWN, MI_TYPE_MAP, OCCUPANCY_MAP, PROPERTY_MAP,
    PURPOSE_MAP, ZERO_BAL_MAP,
)

# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------

def parse_mmyyyy(s: pd.Series) -> pd.Series:
    """
    Parse Fannie Mae's MMYYYY date encoding into a month-start timestamp.

    '022018' -> 2018-02-01. Values that are blank or malformed become NaT
    rather than raising, because a bad date is a data-quality finding to be
    reported (Section 5.4), not a pipeline crash.
    """
    s = s.astype("string").str.strip()
    s = s.where(s.str.len() == 6, other=pd.NA)
    return pd.to_datetime(s, format="%m%Y", errors="coerce")


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.strip(), errors="coerce")


def clean_category(s: pd.Series, unknown: str | None = None) -> pd.Series:
    """Strip, and map empty string to an explicit Unknown label.

    dataset_usage.md Section 5.1: genuinely missing categoricals get an explicit
    "Unknown" category rather than a mode-imputed value, to avoid injecting
    false signal.
    """
    unknown = unknown or CFG.bands.unknown_label
    out = s.astype("string").str.strip()
    return out.replace({"": pd.NA}).fillna(unknown)


# ---------------------------------------------------------------------------
# Banding: continuous raw values -> documented categorical bands
# ---------------------------------------------------------------------------

def band(values: pd.Series, edges: tuple, labels: tuple) -> pd.Series:
    """
    Cut a continuous series into the documented bands.

    Continuous source values are always retained alongside the band (see
    build_static), so the api_spec.md band contract is satisfied without
    throwing away model resolution.
    """
    v = pd.to_numeric(values, errors="coerce")
    out = pd.cut(v, bins=list(edges), labels=list(labels), right=False,
                 include_lowest=True)
    return out.astype("string").fillna(CFG.bands.unknown_label)


def band_credit_score(s: pd.Series) -> pd.Series:
    return band(s, CFG.bands.credit_score_edges, CFG.bands.credit_score_labels)


def band_ltv(s: pd.Series) -> pd.Series:
    return band(s, CFG.bands.ltv_edges, CFG.bands.ltv_labels)


def band_dti(s: pd.Series) -> pd.Series:
    return band(s, CFG.bands.dti_edges, CFG.bands.dti_labels)


# ---------------------------------------------------------------------------
# Target derivation -- the two critical functions
# ---------------------------------------------------------------------------

def derive_delinquency(dlq_status: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    DLQ_STATUS -> (dlq_months, days_past_due).

    Fannie Mae encodes DLQ_STATUS as a zero-padded count of MONTHS delinquent:
      '00' = current, '01' = 30 days, '02' = 60 days, ... 'XX' = not reported.

    Returns dlq_months as a nullable float (NaN where the servicer reported
    'XX' or left it blank) and days_past_due as dlq_months * 30.

    NaN is deliberately preserved rather than filled with 0: an unreported
    status is NOT the same as a current loan, and conflating them would
    understate risk. The missingness is surfaced as a data-quality finding and
    handled explicitly downstream.
    """
    s = dlq_status.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, DLQ_UNKNOWN: pd.NA})
    months = pd.to_numeric(s, errors="coerce")
    days = months * 30.0
    return months, days


def derive_status(
    dlq_months: pd.Series,
    zero_bal_code: pd.Series,
    *,
    default_months: int | None = None,
) -> pd.DataFrame:
    """
    Derive the canonical loan state and the three event flags.

    Zero_Bal_Code is populated only on a loan's FINAL observation, when the
    balance reaches zero, so it identifies terminal states:

        '01'                  -> Prepaid  (prepaid in full or matured)
        '02','03','09','15'   -> Default  (credit-event termination:
                                           third-party sale, short sale,
                                           deed-in-lieu/REO, note sale)
        '06'                  -> Closed   (repurchased by the seller)
        '16'                  -> Closed   (reperforming loan sale)

    Non-terminal rows are classified from delinquency depth using the D180
    convention (assumption A4 in docs/documentation_analysis.md):

        dlq_months >= 6  -> Default
        dlq_months >= 1  -> Delinquent
        otherwise        -> Current

    Returns a frame with current_status, delinquency, default_flag,
    prepayment_flag, is_terminated and termination_reason.
    """
    rules = CFG.rules
    default_months = default_months if default_months is not None else rules.default_dlq_months

    zb = zero_bal_code.astype("string").str.strip().replace({"": pd.NA})
    has_zb = zb.notna()

    is_prepaid = has_zb & zb.isin(list(rules.zb_prepaid))
    is_credit_event = has_zb & zb.isin(list(rules.zb_credit_event))
    is_other_close = has_zb & ~is_prepaid & ~is_credit_event

    m = pd.to_numeric(dlq_months, errors="coerce")
    deep_dlq = m >= default_months
    any_dlq = m >= rules.delinquency_dlq_months

    status = pd.Series("Current", index=zero_bal_code.index, dtype="object")
    # Order matters: terminal states are assigned last so they win.
    status = status.mask(any_dlq, "Delinquent")
    status = status.mask(deep_dlq, "Default")
    status = status.mask(is_other_close, "Closed")
    status = status.mask(is_credit_event, "Default")
    status = status.mask(is_prepaid, "Prepaid")

    reason = pd.Series(pd.NA, index=zero_bal_code.index, dtype="string")
    reason = reason.mask(has_zb, zb.map(ZERO_BAL_MAP).astype("string"))

    return pd.DataFrame({
        "current_status": status,
        "delinquency": any_dlq.fillna(False).astype(bool),
        "default_flag": (deep_dlq.fillna(False) | is_credit_event).astype(bool),
        "prepayment_flag": is_prepaid.astype(bool),
        "is_terminated": has_zb.astype(bool),
        "termination_reason": reason,
    }, index=zero_bal_code.index)


# ---------------------------------------------------------------------------
# Light derivation for the full-population aggregate pass
# ---------------------------------------------------------------------------

def derive_light(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal derivation applied to 100% of rows for population aggregates.

    Only the fields the aggregate needs, because running the full bridge over
    ~870M rows would dominate runtime for no benefit. Returns a small frame
    aligned to df's index.

    Two deliberate optimisations, both measured:

    * `pd.to_numeric(..., errors="coerce")` already maps 'XX' and '' to NaN,
      so the strip/upper/replace chain in derive_delinquency is unnecessary
      here (454 ms -> 223 ms per 200K-row chunk).
    * The reporting month is kept as the raw MMYYYY *string* key. Grouping on
      it is identical to grouping on a parsed timestamp, and the ~96 distinct
      keys are parsed once at combine time instead of 870M times (saves a
      further 115 ms per chunk).
    """
    rules = CFG.rules
    dlq_months = pd.to_numeric(df["DLQ_STATUS"], errors="coerce")
    zb = df["Zero_Bal_Code"]
    is_prepaid = zb.isin(rules.zb_prepaid)
    is_credit_event = zb.isin(rules.zb_credit_event)
    return pd.DataFrame({
        "act_period": df["ACT_PERIOD"],
        "current_balance": pd.to_numeric(df["CURRENT_UPB"], errors="coerce"),
        "dlq_months": dlq_months,
        "is_delinquent": (dlq_months >= rules.delinquency_dlq_months).fillna(False),
        "is_d90": (dlq_months >= rules.default_dlq_months_alt).fillna(False),
        "is_default": ((dlq_months >= rules.default_dlq_months).fillna(False)
                       | is_credit_event),
        "is_prepaid": is_prepaid,
        "is_terminated": zb.ne("") & zb.notna(),
        "dlq_unknown": dlq_months.isna(),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Full canonical builders
# ---------------------------------------------------------------------------

def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw chunk -> canonical loan_monthly_performance rows.

    One row per (loan_id, reporting_month). `month_index` is NOT set here -- it
    is a per-loan sequential counter that requires the loan's complete sorted
    history, so it is assigned in dataset_builder once shards are combined.
    """
    out = pd.DataFrame(index=df.index)
    out["loan_id"] = df["LOAN_ID"].astype("string").str.strip()
    out["reporting_month"] = parse_mmyyyy(df["ACT_PERIOD"])

    out["current_balance"] = to_numeric(df["CURRENT_UPB"])
    out["current_interest_rate"] = to_numeric(df["CURR_RATE"])
    out["loan_age_months"] = to_numeric(df["LOAN_AGE"])
    out["remaining_term_months"] = to_numeric(df["REM_MONTHS"])
    out["scheduled_principal"] = to_numeric(df["TOT_SCHD_PRNCPL"])
    out["non_interest_bearing_upb"] = to_numeric(df["NON_INTEREST_BEARING_UPB"])
    out["principal_forgiveness"] = to_numeric(df["PRINCIPAL_FORGIVENESS_AMOUNT"])
    out["last_upb"] = to_numeric(df["LAST_UPB"])

    dlq_months, days = derive_delinquency(df["DLQ_STATUS"])
    out["dlq_months"] = dlq_months
    out["days_past_due"] = days

    status = derive_status(dlq_months, df["Zero_Bal_Code"])
    out = pd.concat([out, status], axis=1)

    zb = df["Zero_Bal_Code"].astype("string").str.strip().replace({"": pd.NA})
    out["zero_balance_code"] = zb
    out["zero_balance_date"] = parse_mmyyyy(df["ZB_DTE"])
    out["last_paid_installment"] = parse_mmyyyy(df["LAST_PAID_INSTALLMENT_DATE"])

    out["modification_flag"] = (
        df["MOD_FLAG"].astype("string").str.strip().eq("Y").fillna(False)
    )
    fb = df["FORBEARANCE_INDICATOR"].astype("string").str.strip()
    out["forbearance_flag"] = (~fb.isin(["", "N", "7"])).fillna(False)
    out["servicer_name"] = clean_category(df["SERVICER"])

    # ---- alternative default definitions ---------------------------------
    # The primary default_flag above is D180-or-credit-event, per assumption A4.
    # That definition is standard, but it has a documented weakness in this
    # dataset: COVID-era forbearance plans pushed large numbers of loans past
    # 180 days delinquent WITHOUT an economic default, because missed payments
    # under an approved forbearance still accrue as delinquency. Verified in the
    # data -- the D180 rate spikes sharply through 2020-2021.
    #
    # Rather than silently pick one definition, all three are materialised and
    # reported side by side:
    #
    #   default_flag                 D180 OR credit event      (primary, documented)
    #   default_flag_credit_event    terminal credit event only (unambiguous)
    #   default_flag_ex_forbearance  credit event OR (D180 AND not in forbearance)
    #
    # The forbearance flag is also carried as a model feature so the classifier
    # can separate relief-driven delinquency from genuine credit deterioration.
    rules = CFG.rules
    zb_clean = zb.fillna("")
    is_credit_event = zb_clean.isin(list(rules.zb_credit_event))
    deep_dlq = (out["dlq_months"] >= rules.default_dlq_months).fillna(False)
    out["default_flag_credit_event"] = is_credit_event.astype(bool)
    out["default_flag_ex_forbearance"] = (
        is_credit_event | (deep_dlq & ~out["forbearance_flag"])
    ).astype(bool)

    return out


def build_static(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw chunk -> canonical loan_static_attributes rows (one per loan).

    Retains BOTH the continuous source value and the documented band for
    credit score, LTV and DTI, so api_spec.md's band contract is satisfied
    while models keep full numeric resolution.
    """
    out = pd.DataFrame(index=df.index)
    out["loan_id"] = df["LOAN_ID"].astype("string").str.strip()

    out["original_balance"] = to_numeric(df["ORIG_UPB"])
    out["interest_rate"] = to_numeric(df["ORIG_RATE"])
    out["original_term_months"] = to_numeric(df["ORIG_TERM"])

    out["credit_score"] = to_numeric(df["CSCORE_B"])
    out["co_borrower_credit_score"] = to_numeric(df["CSCORE_C"])
    out["ltv"] = to_numeric(df["OLTV"])
    out["combined_ltv"] = to_numeric(df["OCLTV"])
    out["dti"] = to_numeric(df["DTI"])

    out["credit_score_band"] = band_credit_score(out["credit_score"])
    out["ltv_band"] = band_ltv(out["ltv"])
    out["dti_band"] = band_dti(out["dti"])

    out["origination_month"] = parse_mmyyyy(df["ORIG_DATE"])
    out["state"] = clean_category(df["STATE"])
    out["loan_purpose"] = clean_category(df["PURPOSE"]).map(
        lambda v: PURPOSE_MAP.get(v, CFG.bands.unknown_label)
    ).astype("string")
    out["property_type"] = clean_category(df["PROP"]).map(
        lambda v: PROPERTY_MAP.get(v, CFG.bands.unknown_label)
    ).astype("string")
    out["occupancy_type"] = clean_category(df["OCC_STAT"]).map(
        lambda v: OCCUPANCY_MAP.get(v, CFG.bands.unknown_label)
    ).astype("string")
    out["channel"] = clean_category(df["CHANNEL"]).map(
        lambda v: CHANNEL_MAP.get(v, CFG.bands.unknown_label)
    ).astype("string")
    out["mi_type"] = clean_category(df["MI_TYPE"]).map(
        lambda v: MI_TYPE_MAP.get(v, CFG.bands.unknown_label)
    ).astype("string")

    out["seller_name"] = clean_category(df["SELLER"])
    out["num_borrowers"] = to_numeric(df["NUM_BO"])
    out["num_units"] = to_numeric(df["NO_UNITS"])
    out["msa"] = clean_category(df["MSA"])
    out["zip3"] = clean_category(df["ZIP"])
    out["mi_pct"] = to_numeric(df["MI_PCT"])
    out["product_type"] = clean_category(df["PRODUCT"])
    out["first_time_buyer"] = df["FIRST_FLAG"].astype("string").str.strip().eq("Y")
    out["homeready_flag"] = (
        df["HOMEREADY_PROGRAM_INDICATOR"].astype("string").str.strip().eq("H")
    )
    out["high_balance_flag"] = (
        df["HIGH_BALANCE_LOAN_INDICATOR"].astype("string").str.strip().eq("Y")
    )

    out["vintage_year"] = out["origination_month"].dt.year
    out["vintage_quarter"] = out["origination_month"].dt.quarter
    return out


# ---------------------------------------------------------------------------
# Cleaning applied after the panel is assembled
# ---------------------------------------------------------------------------

def clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Post-assembly cleaning of the monthly panel.

    Implements dataset_usage.md Section 5:
      - structural missingness encoded explicitly, not statistically imputed
      - reporting-lag balance gaps forward-filled WITH a carried-forward flag
      - duplicates on (loan_id, reporting_month): identical extras dropped,
        differing rows quarantined for the anomaly pipeline
      - no silent drops: everything removed is returned in `quarantine`

    Returns (clean_panel, quarantine).
    """
    panel = panel.sort_values(["loan_id", "reporting_month"], kind="mergesort")
    quarantine_parts: list[pd.DataFrame] = []

    # --- rows with no usable key cannot be placed in the panel at all -------
    bad_key = panel["loan_id"].isna() | panel["reporting_month"].isna()
    if bad_key.any():
        q = panel.loc[bad_key].copy()
        q["quarantine_reason"] = "missing_primary_key"
        quarantine_parts.append(q)
        panel = panel.loc[~bad_key]

    # --- duplicates on the composite key ------------------------------------
    dup_mask = panel.duplicated(subset=["loan_id", "reporting_month"], keep=False)
    if dup_mask.any():
        dups = panel.loc[dup_mask]
        # Byte-identical extras are safe to drop; genuinely conflicting rows
        # are a data-quality signal a reviewer must see.
        full_dup = dups.duplicated(keep="first")
        conflicting = dups.loc[~full_dup].duplicated(
            subset=["loan_id", "reporting_month"], keep=False
        )
        conflict_rows = dups.loc[~full_dup].loc[conflicting]
        if len(conflict_rows):
            q = conflict_rows.copy()
            q["quarantine_reason"] = "conflicting_duplicate_loan_month"
            quarantine_parts.append(q)
        panel = panel.loc[~panel.duplicated(subset=["loan_id", "reporting_month"],
                                            keep="first")]

    # --- structural missingness (Section 5.1) -------------------------------
    # days_past_due is naturally absent for a loan that is not delinquent.
    # Encode explicitly rather than impute, and keep a flag recording that the
    # servicer did not report a status.
    panel["dlq_status_unreported"] = panel["dlq_months"].isna()
    panel["dlq_months"] = panel["dlq_months"].fillna(0.0)
    panel["days_past_due"] = panel["days_past_due"].fillna(0.0)

    # --- reporting-lag balance gaps ----------------------------------------
    # Fannie Mae systematically reports CURRENT_UPB = 0.00 on a loan's earliest
    # records, before the servicer's first full balance submission. Verified
    # empirically: in the 2019Q1 file EVERY record in the first reporting month
    # (129,126 rows) carries a zero balance. A zero balance on a NON-terminal
    # row is therefore a reporting artifact, not a paid-off loan, and must not
    # be fed to the model as a real balance of zero -- it would corrupt every
    # balance ratio and make new loans look fully amortised.
    #
    # Repair order matters: ffill carries the last known balance forward across
    # a mid-history gap; bfill then repairs LEADING zeros, which have no prior
    # value to carry. Both are flagged so the imputation is auditable and can
    # be excluded from analysis (dataset_usage.md Section 5.1 requires a
    # carried-forward marker).
    zero_nonterminal = (
        panel["current_balance"].fillna(0).le(0) & ~panel["is_terminated"]
    )
    panel["balance_imputed"] = zero_nonterminal
    panel.loc[zero_nonterminal, "current_balance"] = np.nan
    grp = panel.groupby("loan_id", sort=False)["current_balance"]
    panel["current_balance"] = grp.ffill()
    panel["current_balance"] = (
        panel.groupby("loan_id", sort=False)["current_balance"].bfill()
    )
    # Any loan whose entire history is zero-balance stays NaN rather than being
    # silently filled with a fabricated number.
    panel["balance_unrecoverable"] = panel["current_balance"].isna()

    quarantine = (
        pd.concat(quarantine_parts, ignore_index=True)
        if quarantine_parts else
        pd.DataFrame(columns=list(panel.columns) + ["quarantine_reason"])
    )
    return panel, quarantine


__all__ = [
    "parse_mmyyyy", "to_numeric", "clean_category", "band",
    "band_credit_score", "band_ltv", "band_dti",
    "derive_delinquency", "derive_status", "derive_light",
    "build_monthly", "build_static", "clean_panel",
]

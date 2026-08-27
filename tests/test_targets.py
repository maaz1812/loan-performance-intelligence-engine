"""
Tests for the target-derivation bridge.

These are the highest-risk functions in the project: DLQ_STATUS and
Zero_Bal_Code drive every label, so an error here would silently invalidate
every model. Each case below is hand-computed from the Fannie Mae code
definitions documented in ml/data_pipeline/schema_detection.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ml.config import CFG
from ml.data_pipeline.preprocessing import (
    band_credit_score, band_dti, band_ltv, derive_delinquency, derive_status,
    parse_mmyyyy,
)


# ---------------------------------------------------------------------------
# DLQ_STATUS -> months / days delinquent
# ---------------------------------------------------------------------------

def test_dlq_status_maps_months_to_days():
    s = pd.Series(["00", "01", "02", "03", "06", "12"])
    months, days = derive_delinquency(s)
    assert list(months) == [0, 1, 2, 3, 6, 12]
    assert list(days) == [0.0, 30.0, 60.0, 90.0, 180.0, 360.0]


def test_dlq_status_unknown_becomes_nan_not_zero():
    """
    'XX' means the servicer did not report. Treating it as 0 (current) would
    understate risk, so it must stay NaN and be surfaced as a DQ finding.
    """
    months, days = derive_delinquency(pd.Series(["XX", "", "00"]))
    assert months.isna().tolist() == [True, True, False]
    assert days.isna().tolist() == [True, True, False]
    assert months.iloc[2] == 0


def test_dlq_status_tolerates_whitespace_and_case():
    months, _ = derive_delinquency(pd.Series([" 01 ", "xx", "0"]))
    assert months.iloc[0] == 1
    assert pd.isna(months.iloc[1])
    assert months.iloc[2] == 0


# ---------------------------------------------------------------------------
# Status / event-flag derivation
# ---------------------------------------------------------------------------

def _status(dlq, zb):
    return derive_status(pd.Series(dlq, dtype="float64"),
                         pd.Series(zb, dtype="object"))


def test_current_loan():
    r = _status([0.0], [None])
    assert r["current_status"].iloc[0] == "Current"
    assert not r["delinquency"].iloc[0]
    assert not r["default_flag"].iloc[0]
    assert not r["prepayment_flag"].iloc[0]
    assert not r["is_terminated"].iloc[0]


def test_delinquent_at_30_days():
    r = _status([1.0], [None])
    assert r["current_status"].iloc[0] == "Delinquent"
    assert r["delinquency"].iloc[0]
    assert not r["default_flag"].iloc[0]


def test_delinquent_but_not_yet_default_at_90_days():
    """D180 is the primary definition, so 3 months delinquent is not default."""
    r = _status([3.0], [None])
    assert r["current_status"].iloc[0] == "Delinquent"
    assert not r["default_flag"].iloc[0]


def test_default_at_180_days():
    r = _status([6.0], [None])
    assert r["current_status"].iloc[0] == "Default"
    assert r["default_flag"].iloc[0]
    assert r["delinquency"].iloc[0]        # a defaulted loan is also delinquent


def test_prepaid_terminal_state():
    r = _status([0.0], ["01"])
    assert r["current_status"].iloc[0] == "Prepaid"
    assert r["prepayment_flag"].iloc[0]
    assert not r["default_flag"].iloc[0]
    assert r["is_terminated"].iloc[0]
    assert r["termination_reason"].iloc[0] == "prepaid_or_matured"


@pytest.mark.parametrize("code", ["02", "03", "09", "15"])
def test_credit_event_codes_are_default_not_prepayment(code):
    """
    A credit-event termination is a default even when the loan was not yet
    180 days delinquent on its final record. Mixing these into prepayment
    would invert the economics of two opposite borrower behaviours.
    """
    r = _status([0.0], [code])
    assert r["current_status"].iloc[0] == "Default"
    assert r["default_flag"].iloc[0]
    assert not r["prepayment_flag"].iloc[0]


@pytest.mark.parametrize("code", ["06", "16"])
def test_repurchase_and_reperforming_sale_are_closed(code):
    r = _status([0.0], [code])
    assert r["current_status"].iloc[0] == "Closed"
    assert not r["default_flag"].iloc[0]
    assert not r["prepayment_flag"].iloc[0]
    assert r["is_terminated"].iloc[0]


def test_prepayment_wins_over_delinquency_on_terminal_row():
    """A loan can be delinquent and still pay off in full; the terminal
    zero-balance code is authoritative for the final state."""
    r = _status([2.0], ["01"])
    assert r["current_status"].iloc[0] == "Prepaid"
    assert r["prepayment_flag"].iloc[0]


def test_deep_delinquency_plus_credit_event_is_default():
    r = _status([8.0], ["09"])
    assert r["current_status"].iloc[0] == "Default"
    assert r["default_flag"].iloc[0]


def test_states_are_confined_to_the_documented_five():
    """dataset_usage.md Section 7.4 fixes exactly five states."""
    r = derive_status(
        pd.Series([0.0, 1.0, 6.0, 0.0, 0.0, 0.0, None], dtype="float64"),
        pd.Series([None, None, None, "01", "09", "06", None], dtype="object"),
    )
    assert set(r["current_status"]) <= set(CFG.model.loan_states)


def test_unreported_dlq_does_not_fabricate_delinquency():
    r = _status([None], [None])
    assert r["current_status"].iloc[0] == "Current"
    assert not r["delinquency"].iloc[0]
    assert not r["default_flag"].iloc[0]


def test_alt_d90_definition_is_stricter():
    """The configurable alternative default definition must widen the label."""
    d180 = derive_status(pd.Series([3.0]), pd.Series([None], dtype="object"))
    d90 = derive_status(pd.Series([3.0]), pd.Series([None], dtype="object"),
                        default_months=CFG.rules.default_dlq_months_alt)
    assert not d180["default_flag"].iloc[0]
    assert d90["default_flag"].iloc[0]


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_parse_mmyyyy():
    s = parse_mmyyyy(pd.Series(["022018", "122025", "012018"]))
    assert s.iloc[0] == pd.Timestamp("2018-02-01")
    assert s.iloc[1] == pd.Timestamp("2025-12-01")
    assert s.iloc[2] == pd.Timestamp("2018-01-01")


def test_parse_mmyyyy_rejects_malformed():
    s = parse_mmyyyy(pd.Series(["", "2018", "1320188", "992018"]))
    assert s.isna().all()


# ---------------------------------------------------------------------------
# Banding -- must reproduce the literal examples in api_spec.md
# ---------------------------------------------------------------------------

def test_credit_band_matches_api_spec_example():
    """api_spec.md shows credit_score_band "680-719"; our finer grid must
    place 700 inside the 700-719 bucket and 690 inside 680-699."""
    assert band_credit_score(pd.Series([690.0])).iloc[0] == "680-699"
    assert band_credit_score(pd.Series([700.0])).iloc[0] == "700-719"
    assert band_credit_score(pd.Series([575.0])).iloc[0] == "<580"
    assert band_credit_score(pd.Series([800.0])).iloc[0] == "780+"


def test_ltv_band_matches_api_spec_example():
    assert band_ltv(pd.Series([85.0])).iloc[0] == "85-90"
    assert band_ltv(pd.Series([80.0])).iloc[0] == "80-85"
    assert band_ltv(pd.Series([98.0])).iloc[0] == "97+"


def test_dti_band_matches_api_spec_example():
    assert band_dti(pd.Series([40.0])).iloc[0] == "36-43"
    assert band_dti(pd.Series([19.0])).iloc[0] == "<20"


def test_bands_label_missing_as_unknown_not_a_numeric_bucket():
    for fn in (band_credit_score, band_ltv, band_dti):
        assert fn(pd.Series([None])).iloc[0] == CFG.bands.unknown_label

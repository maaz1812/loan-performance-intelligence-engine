"""
Schema detection and the raw -> canonical schema bridge.

The delivered data is the RAW Fannie Mae Single-Family Loan Performance file:
pipe-delimited, NO header row, 113 positional fields. The project
documentation (api_spec.md, database_schema.md, dataset_usage.md) instead
describes an organizer-curated schema with fields like `loan_id`,
`credit_score_band`, `current_status`, `default_flag`.

This module owns the bridge between the two. It is the single source of truth
for:

  1. FIELDS      -- all 113 positional fields, named and classified
  2. USED_FIELDS -- the subset actually parsed (column pruning for throughput)
  3. discovery   -- empirical profiling used to VERIFY the layout rather than
                    assume it

Every field carries `verified`, recording whether the positional mapping was
confirmed against observed data rather than taken on faith from the published
layout. Unverified tail fields are reported as such in
reports/data_dictionary.md instead of being silently trusted.
"""
from __future__ import annotations

import collections
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Field classification vocabulary (dataset_usage.md Section 17.3, Rule 3)
# ---------------------------------------------------------------------------
IDENTIFIER = "identifier"    # keys linking the tables
TIME = "time"                # monthly observation / date fields
ACQUISITION = "acquisition"  # origination-level, constant per loan
PERFORMANCE = "performance"  # monthly-varying servicing observations
CREDIT_EVENT = "credit_event"  # termination / loss fields
UNUSED = "unused"            # present in the layout but empty or out of scope


@dataclass(frozen=True)
class Field:
    idx: int
    name: str
    kind: str
    dtype: str
    meaning: str
    ml_usage: str
    verified: bool = True


# ---------------------------------------------------------------------------
# The 113-field layout.
#
# Names follow the published Fannie Mae SF Loan Performance layout. Every
# field marked verified=True had its positional mapping confirmed empirically
# (value ranges, formats, cardinality and cross-field consistency) during
# discovery -- see discover_layout(). Fields marked verified=False are
# consistently empty or ambiguous in the delivered vintage; they are named for
# completeness but nothing in the pipeline depends on them.
# ---------------------------------------------------------------------------
FIELDS: tuple[Field, ...] = (
    Field(0,  "POOL_ID", UNUSED, "string", "Pool identifier", "Not used - empty in this vintage", False),
    Field(1,  "LOAN_ID", IDENTIFIER, "string", "Unique loan identifier assigned by Fannie Mae", "PRIMARY KEY; join key; deterministic sampling key"),
    Field(2,  "ACT_PERIOD", TIME, "MMYYYY", "Monthly activity/reporting period", "reporting_month; drives the time-aware split"),
    Field(3,  "CHANNEL", ACQUISITION, "category", "Origination channel: R=Retail, C=Correspondent, B=Broker", "Categorical feature"),
    Field(4,  "SELLER", ACQUISITION, "string", "Entity that sold the loan to Fannie Mae", "High-cardinality categorical; segment dimension"),
    Field(5,  "SERVICER", PERFORMANCE, "string", "Current servicer of record", "Segment dimension; servicer-level aggregate features"),
    Field(6,  "MASTER_SERVICER", UNUSED, "string", "Master servicer", "Not used - empty", False),
    Field(7,  "ORIG_RATE", ACQUISITION, "float", "Interest rate at origination (%)", "Numeric feature; prepayment incentive"),
    Field(8,  "CURR_RATE", PERFORMANCE, "float", "Current interest rate (%)", "Numeric feature; rate-gap vs origination"),
    Field(9,  "ORIG_UPB", ACQUISITION, "float", "Original unpaid principal balance", "original_balance; denominator for balance ratios"),
    Field(10, "ISSUANCE_UPB", UNUSED, "float", "UPB at securitization", "Not used - empty", False),
    Field(11, "CURRENT_UPB", PERFORMANCE, "float", "Current actual unpaid principal balance", "current_balance; amortisation and stress signal"),
    Field(12, "ORIG_TERM", ACQUISITION, "int", "Original loan term in months", "Numeric feature"),
    Field(13, "ORIG_DATE", TIME, "MMYYYY", "Origination date", "origination_month; vintage segment"),
    Field(14, "FIRST_PAY", TIME, "MMYYYY", "First scheduled payment date", "Derived timing feature"),
    Field(15, "LOAN_AGE", TIME, "int", "Months since origination", "loan_age_months; survival duration clock"),
    Field(16, "REM_MONTHS", TIME, "int", "Remaining months to maturity", "remaining_term_months"),
    Field(17, "ADJ_REM_MONTHS", TIME, "int", "Adjusted remaining months to maturity", "Numeric feature"),
    Field(18, "MATR_DT", TIME, "MMYYYY", "Scheduled maturity date", "Censoring horizon for survival analysis"),
    Field(19, "OLTV", ACQUISITION, "float", "Original loan-to-value ratio (%)", "ltv_band source; core risk driver"),
    Field(20, "OCLTV", ACQUISITION, "float", "Original combined loan-to-value ratio (%)", "Numeric feature"),
    Field(21, "NUM_BO", ACQUISITION, "int", "Number of borrowers", "Numeric feature"),
    Field(22, "DTI", ACQUISITION, "float", "Debt-to-income ratio at origination (%)", "dti_band source; payment capacity"),
    Field(23, "CSCORE_B", ACQUISITION, "float", "Borrower credit score at origination", "credit_score_band source; strongest single predictor"),
    Field(24, "CSCORE_C", ACQUISITION, "float", "Co-borrower credit score at origination", "Numeric feature; ~43% populated"),
    Field(25, "FIRST_FLAG", ACQUISITION, "category", "First-time homebuyer indicator (Y/N)", "Binary feature"),
    Field(26, "PURPOSE", ACQUISITION, "category", "Loan purpose: P=Purchase, C=Cash-out refi, R=Refi (no cash-out)", "loan_purpose; categorical feature"),
    Field(27, "PROP", ACQUISITION, "category", "Property type: SF, PU, CO, CP, MH", "property_type; categorical feature"),
    Field(28, "NO_UNITS", ACQUISITION, "int", "Number of units in the property", "Numeric feature"),
    Field(29, "OCC_STAT", ACQUISITION, "category", "Occupancy: P=Principal, I=Investor, S=Second home", "occupancy_type; categorical feature"),
    Field(30, "STATE", ACQUISITION, "category", "US state / territory code", "state; geographic segment dimension"),
    Field(31, "MSA", ACQUISITION, "category", "Metropolitan statistical area code", "Geographic feature"),
    Field(32, "ZIP", ACQUISITION, "category", "First three digits of the property ZIP", "Coarse geographic feature"),
    Field(33, "MI_PCT", ACQUISITION, "float", "Mortgage insurance coverage percentage", "Numeric feature; ~29% populated"),
    Field(34, "PRODUCT", ACQUISITION, "category", "Product type (FRM / ARM)", "Categorical feature; FRM-only in this vintage"),
    Field(35, "PPMT_FLG", ACQUISITION, "category", "Prepayment penalty indicator", "Binary feature"),
    Field(36, "IO", ACQUISITION, "category", "Interest-only indicator", "Binary feature"),
    Field(37, "FIRST_PAY_IO", UNUSED, "MMYYYY", "First IO payment date", "Not used - empty", False),
    Field(38, "MNTHS_TO_AMTZ_IO", UNUSED, "int", "Months to IO amortisation", "Not used - empty", False),
    Field(39, "DLQ_STATUS", PERFORMANCE, "category", "Months delinquent: 00=current, 01=30d, 02=60d ... XX=unknown", "PRIMARY TARGET SOURCE -> days_past_due, delinquency, default"),
    Field(40, "PMT_HISTORY", PERFORMANCE, "string", "48-character string encoding payment history", "Not used directly - superseded by DLQ_STATUS panel history"),
    Field(41, "MOD_FLAG", PERFORMANCE, "category", "Loan modification indicator (Y/N)", "modification_flag; anomaly rule input"),
    Field(42, "MI_CANCEL_FLAG", UNUSED, "category", "MI cancellation indicator", "Not used - empty", False),
    Field(43, "Zero_Bal_Code", CREDIT_EVENT, "category", "Reason the balance went to zero: 01=prepaid/matured, 02/03/09/15=credit event, 06=repurchase", "PRIMARY TARGET SOURCE -> prepayment_flag, default_flag, Closed state"),
    Field(44, "ZB_DTE", CREDIT_EVENT, "MMYYYY", "Date the balance reached zero", "Event date for survival analysis"),
    Field(45, "LAST_UPB", CREDIT_EVENT, "float", "UPB immediately before the zero-balance event", "Exposure at default; loss severity input"),
    Field(46, "RPRCH_DTE", UNUSED, "MMYYYY", "Repurchase date", "Not used - sparse", False),
    Field(47, "CURR_SCHD_PRNCPL", UNUSED, "float", "Current scheduled principal", "Not used - empty", False),
    Field(48, "TOT_SCHD_PRNCPL", PERFORMANCE, "float", "Total scheduled principal paid", "Payment-behaviour feature"),
    Field(49, "UNSCHD_PRNCPL_CURR", UNUSED, "float", "Current unscheduled principal (curtailment)", "Not used - empty", False),
    Field(50, "LAST_PAID_INSTALLMENT_DATE", PERFORMANCE, "MMYYYY", "Date of the last paid installment", "Staleness / payment-recency signal"),
    Field(51, "FORECLOSURE_DATE", CREDIT_EVENT, "MMYYYY", "Foreclosure completion date", "Credit-event corroboration"),
    Field(52, "DISPOSITION_DATE", CREDIT_EVENT, "MMYYYY", "Property disposition date", "Credit-event corroboration"),
    Field(53, "FORECLOSURE_COSTS", CREDIT_EVENT, "float", "Foreclosure costs incurred", "Loss severity input"),
    Field(54, "PROPERTY_PRESERVATION_AND_REPAIR_COSTS", CREDIT_EVENT, "float", "Property preservation and repair costs", "Loss severity input"),
    Field(55, "ASSET_RECOVERY_COSTS", CREDIT_EVENT, "float", "Asset recovery costs", "Loss severity input"),
    Field(56, "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS", CREDIT_EVENT, "float", "Miscellaneous holding expenses and credits", "Loss severity input"),
    Field(57, "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", CREDIT_EVENT, "float", "Taxes while holding the property", "Loss severity input"),
    Field(58, "NET_SALES_PROCEEDS", CREDIT_EVENT, "float", "Net proceeds from the property sale", "Loss severity input"),
    Field(59, "CREDIT_ENHANCEMENT_PROCEEDS", CREDIT_EVENT, "float", "Credit enhancement proceeds", "Loss severity input"),
    Field(60, "REPURCHASES_MAKE_WHOLE_PROCEEDS", CREDIT_EVENT, "float", "Repurchase / make-whole proceeds", "Loss severity input"),
    Field(61, "OTHER_FORECLOSURE_PROCEEDS", CREDIT_EVENT, "float", "Other foreclosure proceeds", "Loss severity input"),
    Field(62, "NON_INTEREST_BEARING_UPB", PERFORMANCE, "float", "Non-interest-bearing UPB (deferred balance)", "Forbearance / deferral signal"),
    Field(63, "PRINCIPAL_FORGIVENESS_AMOUNT", PERFORMANCE, "float", "Principal forgiveness amount", "Modification-severity signal"),
    Field(64, "ORIGINAL_LIST_START_DATE", UNUSED, "MMYYYY", "Original REO listing start date", "Not used - empty", False),
    Field(65, "ORIGINAL_LIST_PRICE", UNUSED, "float", "Original REO list price", "Not used - empty", False),
    Field(66, "CURRENT_LIST_START_DATE", UNUSED, "MMYYYY", "Current REO listing start date", "Not used - empty", False),
    Field(67, "CURRENT_LIST_PRICE", UNUSED, "float", "Current REO list price", "Not used - empty", False),
    Field(68, "ISSUE_SCOREB", UNUSED, "float", "Borrower score at issuance", "Not used - empty", False),
    Field(69, "ISSUE_SCOREC", UNUSED, "float", "Co-borrower score at issuance", "Not used - empty", False),
    Field(70, "CURR_SCOREB", UNUSED, "float", "Current borrower score", "Not used - empty", False),
    Field(71, "CURR_SCOREC", UNUSED, "float", "Current co-borrower score", "Not used - empty", False),
    Field(72, "MI_TYPE", ACQUISITION, "category", "Mortgage insurance type: 1=Borrower paid, 2=Lender paid, 3=Investor paid", "Categorical feature; fill rate tracks MI_PCT"),
    Field(73, "SERV_IND", PERFORMANCE, "category", "Servicing activity indicator (Y/N)", "Binary feature"),
    Field(74, "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT", UNUSED, "float", "Current-period modification loss", "Not used - empty", False),
    Field(75, "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", UNUSED, "float", "Cumulative modification loss", "Not used - empty", False),
    Field(76, "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS", UNUSED, "float", "Current-period credit event net gain/loss", "Not used - empty", False),
    Field(77, "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", UNUSED, "float", "Cumulative credit event net gain/loss", "Not used - empty", False),
    Field(78, "HOMEREADY_PROGRAM_INDICATOR", ACQUISITION, "category", "HomeReady affordable-lending program indicator", "Categorical feature"),
    Field(79, "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", CREDIT_EVENT, "float", "Principal written off at foreclosure", "Loss severity input"),
    Field(80, "RELOCATION_MORTGAGE_INDICATOR", ACQUISITION, "category", "Relocation mortgage indicator (Y/N)", "Binary feature"),
    Field(81, "ZERO_BALANCE_CODE_CHANGE_DATE", UNUSED, "MMYYYY", "Zero balance code change date", "Not used - empty", False),
    Field(82, "LOAN_HOLDBACK_INDICATOR", UNUSED, "category", "Loan holdback indicator", "Not used - empty", False),
    Field(83, "LOAN_HOLDBACK_EFFECTIVE_DATE", UNUSED, "MMYYYY", "Loan holdback effective date", "Not used - empty", False),
    Field(84, "DELINQUENT_ACCRUED_INTEREST", UNUSED, "float", "Delinquent accrued interest", "Not used - empty", False),
    Field(85, "PROPERTY_INSPECTION_WAIVER_INDICATOR", ACQUISITION, "category", "Property inspection waiver indicator", "Categorical feature"),
    Field(86, "HIGH_BALANCE_LOAN_INDICATOR", ACQUISITION, "category", "High-balance loan indicator (Y/N)", "Binary feature"),
    Field(87, "ARM_5_YR_INDICATOR", UNUSED, "category", "ARM 5-year indicator", "Not used - FRM-only vintage", False),
    Field(88, "ARM_PRODUCT_TYPE", UNUSED, "category", "ARM product type", "Not used - empty", False),
    Field(89, "MONTHS_UNTIL_FIRST_PAYMENT_RESET", UNUSED, "int", "Months until first payment reset", "Not used - empty", False),
    Field(90, "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET", UNUSED, "int", "Months between subsequent resets", "Not used - empty", False),
    Field(91, "INTEREST_RATE_CHANGE_DATE", UNUSED, "MMYYYY", "Interest rate change date", "Not used - empty", False),
    Field(92, "PAYMENT_CHANGE_DATE", UNUSED, "MMYYYY", "Payment change date", "Not used - empty", False),
    Field(93, "ARM_INDEX", UNUSED, "category", "ARM index", "Not used - empty", False),
    Field(94, "ARM_CAP_STRUCTURE", UNUSED, "category", "ARM cap structure", "Not used - empty", False),
    Field(95, "INITIAL_INTEREST_RATE_CAP", UNUSED, "float", "Initial interest rate cap", "Not used - empty", False),
    Field(96, "PERIODIC_INTEREST_RATE_CAP", UNUSED, "float", "Periodic interest rate cap", "Not used - empty", False),
    Field(97, "LIFETIME_INTEREST_RATE_CAP", UNUSED, "float", "Lifetime interest rate cap", "Not used - empty", False),
    Field(98, "MARGIN", UNUSED, "float", "ARM margin", "Not used - empty", False),
    Field(99, "BALLOON_INDICATOR", UNUSED, "category", "Balloon indicator", "Not used - empty", False),
    Field(100, "PLAN_NUMBER", UNUSED, "category", "ARM plan number", "Not used - empty", False),
    Field(101, "FORBEARANCE_INDICATOR", PERFORMANCE, "category", "Forbearance / loss-mitigation plan indicator", "Categorical feature; COVID-era relief signal"),
    Field(102, "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR", ACQUISITION, "category", "High-LTV refinance option indicator", "Binary feature"),
    Field(103, "DEAL_NAME", UNUSED, "string", "Deal name", "Not used - empty", False),
    Field(104, "RE_PROCS_FLAG", CREDIT_EVENT, "category", "Relief refinance / repurchase processing flag", "Sparse categorical"),
    Field(105, "ADR_TYPE", PERFORMANCE, "category", "Alternative disaster relief type", "Categorical feature; disaster-relief signal"),
    Field(106, "ADR_COUNT", PERFORMANCE, "int", "Alternative disaster relief count", "Numeric feature"),
    Field(107, "ADR_UPB", PERFORMANCE, "float", "UPB under alternative disaster relief", "Numeric feature"),
    Field(108, "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", PERFORMANCE, "category", "Payment deferral modification event flag", "Categorical feature", False),
    Field(109, "INTEREST_BEARING_UPB", UNUSED, "float", "Interest bearing UPB", "Not used - empty", False),
    Field(110, "UNVERIFIED_110", UNUSED, "float", "Trailing field carrying credit-score-like values at ~2% fill; position not confirmed against the published layout", "Not used - reported as unverified", False),
    Field(111, "UNVERIFIED_111", UNUSED, "string", "Trailing field, empty in this vintage", "Not used - reported as unverified", False),
    Field(112, "UNVERIFIED_112", UNUSED, "string", "Trailing field, empty in this vintage", "Not used - reported as unverified", False),
)

N_FIELDS = len(FIELDS)
assert N_FIELDS == 113, f"layout must describe 113 fields, got {N_FIELDS}"

FIELD_BY_IDX = {f.idx: f for f in FIELDS}
FIELD_BY_NAME = {f.name: f for f in FIELDS}


# ---------------------------------------------------------------------------
# Column pruning: the subset we actually parse.
# Reading 113 string columns over ~870M rows is wasteful; we parse only what
# the canonical schema, the features and the anomaly rules need.
# ---------------------------------------------------------------------------
USED_IDX: tuple[int, ...] = (
    1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 39, 41, 43, 44, 45, 48, 50,
    62, 63, 72, 73, 78, 86, 101, 105,
)
USED_NAMES: tuple[str, ...] = tuple(FIELD_BY_IDX[i].name for i in USED_IDX)

# Fields constant per loan -> the loan_static_attributes table.
STATIC_FIELDS: tuple[str, ...] = (
    "CHANNEL", "SELLER", "ORIG_RATE", "ORIG_UPB", "ORIG_TERM", "ORIG_DATE",
    "OLTV", "OCLTV", "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG",
    "PURPOSE", "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP",
    "MI_PCT", "PRODUCT", "MI_TYPE", "HOMEREADY_PROGRAM_INDICATOR",
    "HIGH_BALANCE_LOAN_INDICATOR",
)

# Fields that vary month to month -> the loan_monthly_performance table.
MONTHLY_FIELDS: tuple[str, ...] = (
    "ACT_PERIOD", "SERVICER", "CURR_RATE", "CURRENT_UPB", "LOAN_AGE",
    "REM_MONTHS", "MATR_DT", "DLQ_STATUS", "MOD_FLAG", "Zero_Bal_Code",
    "ZB_DTE", "LAST_UPB", "TOT_SCHD_PRNCPL", "LAST_PAID_INSTALLMENT_DATE",
    "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT", "SERV_IND",
    "FORBEARANCE_INDICATOR", "ADR_TYPE",
)

NUMERIC_RAW: frozenset[str] = frozenset({
    "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "CURRENT_UPB", "ORIG_TERM",
    "LOAN_AGE", "REM_MONTHS", "OLTV", "OCLTV", "NUM_BO", "DTI", "CSCORE_B",
    "CSCORE_C", "NO_UNITS", "MI_PCT", "LAST_UPB", "TOT_SCHD_PRNCPL",
    "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT",
})

DATE_RAW: frozenset[str] = frozenset({
    "ACT_PERIOD", "ORIG_DATE", "MATR_DT", "ZB_DTE",
    "LAST_PAID_INSTALLMENT_DATE",
})


# ---------------------------------------------------------------------------
# Canonical schema: what the rest of the system (API, DB, models) consumes.
# This is the target of the bridge.
# ---------------------------------------------------------------------------
CANONICAL_STATIC: tuple[str, ...] = (
    "loan_id", "original_balance", "interest_rate", "credit_score",
    "credit_score_band", "ltv", "ltv_band", "dti", "dti_band", "state",
    "loan_purpose", "property_type", "occupancy_type", "origination_month",
    "original_term_months", "channel", "seller_name", "num_borrowers",
    "co_borrower_credit_score", "first_time_buyer", "num_units", "msa",
    "zip3", "mi_pct", "mi_type", "product_type", "homeready_flag",
    "high_balance_flag", "vintage_year", "vintage_quarter", "source_file",
)

CANONICAL_MONTHLY: tuple[str, ...] = (
    "loan_id", "reporting_month", "month_index", "loan_age_months",
    "current_balance", "current_interest_rate", "remaining_term_months",
    "days_past_due", "dlq_months", "current_status", "delinquency",
    "default_flag", "prepayment_flag", "modification_flag",
    "zero_balance_code", "zero_balance_date", "last_upb",
    "scheduled_principal", "non_interest_bearing_upb",
    "principal_forgiveness", "forbearance_flag", "servicer_name",
    "last_paid_installment", "is_terminated", "termination_reason",
)

# Raw -> canonical name map for straight renames (no transformation).
RENAME_MAP: dict[str, str] = {
    "LOAN_ID": "loan_id",
    "ORIG_UPB": "original_balance",
    "ORIG_RATE": "interest_rate",
    "CSCORE_B": "credit_score",
    "CSCORE_C": "co_borrower_credit_score",
    "OLTV": "ltv",
    "OCLTV": "combined_ltv",
    "DTI": "dti",
    "STATE": "state",
    "ORIG_TERM": "original_term_months",
    "CHANNEL": "channel",
    "SELLER": "seller_name",
    "NUM_BO": "num_borrowers",
    "NO_UNITS": "num_units",
    "MSA": "msa",
    "ZIP": "zip3",
    "MI_PCT": "mi_pct",
    "MI_TYPE": "mi_type",
    "PRODUCT": "product_type",
    "CURR_RATE": "current_interest_rate",
    "CURRENT_UPB": "current_balance",
    "LOAN_AGE": "loan_age_months",
    "REM_MONTHS": "remaining_term_months",
    "SERVICER": "servicer_name",
    "LAST_UPB": "last_upb",
    "TOT_SCHD_PRNCPL": "scheduled_principal",
    "NON_INTEREST_BEARING_UPB": "non_interest_bearing_upb",
    "PRINCIPAL_FORGIVENESS_AMOUNT": "principal_forgiveness",
}

# Coded value dictionaries, used for decoding and for the data dictionary.
PURPOSE_MAP = {"P": "purchase", "C": "cash_out_refinance", "R": "refinance", "U": "Unknown"}
PROPERTY_MAP = {"SF": "single_family", "PU": "planned_urban_development",
                "CO": "condominium", "CP": "cooperative", "MH": "manufactured_home"}
OCCUPANCY_MAP = {"P": "principal", "I": "investor", "S": "second_home", "U": "Unknown"}
CHANNEL_MAP = {"R": "retail", "C": "correspondent", "B": "broker"}
MI_TYPE_MAP = {"1": "borrower_paid", "2": "lender_paid", "3": "investor_paid"}

# DLQ_STATUS -> months delinquent. 'XX' means the servicer did not report.
DLQ_UNKNOWN = "XX"

ZERO_BAL_MAP = {
    "01": "prepaid_or_matured",
    "02": "third_party_sale",
    "03": "short_sale",
    "06": "repurchased",
    "09": "deed_in_lieu_reo",
    "15": "note_sale",
    "16": "reperforming_loan_sale",
}


# ---------------------------------------------------------------------------
# Empirical discovery -- verification, not assumption
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    idx: int
    name: str
    n_distinct: int
    fill_rate: float
    examples: list[str]
    min_len: int
    max_len: int


def discover_layout(
    zip_path: Path,
    n_rows: int = 40_000,
    stride: int = 1,
) -> tuple[int, list[ColumnProfile]]:
    """
    Profile every positional field in a raw ZIP member without extracting it.

    Returns (observed_field_count, per-column profiles). Used to VERIFY that
    the delivered file matches the 113-field contract before the pipeline
    trusts any positional mapping, and to generate the observed-value evidence
    printed in reports/data_dictionary.md.

    A `stride` > 1 samples every Nth line, which avoids the head-of-file bias
    where the first rows all belong to one loan's earliest months.
    """
    zf = zipfile.ZipFile(zip_path)
    member = zf.namelist()[0]

    counters: list[collections.Counter] = []
    lengths: list[list[int]] = []
    n_seen = 0
    observed_widths: collections.Counter = collections.Counter()

    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        for i, line in enumerate(text):
            if n_seen >= n_rows:
                break
            if stride > 1 and (i % stride):
                continue
            parts = line.rstrip("\n").split("|")
            observed_widths[len(parts)] += 1
            if not counters:
                counters = [collections.Counter() for _ in range(len(parts))]
                lengths = [[] for _ in range(len(parts))]
            for j, val in enumerate(parts):
                if j >= len(counters):
                    break
                counters[j][val] += 1
                if val:
                    lengths[j].append(len(val))
            n_seen += 1

    width = observed_widths.most_common(1)[0][0] if observed_widths else 0
    profiles: list[ColumnProfile] = []
    for j in range(width):
        c = counters[j]
        nonempty = sum(n for v, n in c.items() if v != "")
        ex = [v for v, _ in c.most_common(6) if v != ""][:3]
        ln = lengths[j]
        profiles.append(ColumnProfile(
            idx=j,
            name=FIELD_BY_IDX[j].name if j in FIELD_BY_IDX else f"POS_{j}",
            n_distinct=len([v for v in c if v != ""]),
            fill_rate=(nonempty / n_seen) if n_seen else 0.0,
            examples=ex,
            min_len=min(ln) if ln else 0,
            max_len=max(ln) if ln else 0,
        ))
    return width, profiles


def verify_contract(zip_path: Path, n_rows: int = 20_000) -> dict:
    """
    Assert the delivered file honours the schema contract.

    Raises ValueError on a field-count mismatch -- a silent width change would
    corrupt every positional mapping downstream, so this fails loudly rather
    than proceeding.
    """
    width, profiles = discover_layout(zip_path, n_rows=n_rows, stride=7)
    if width != N_FIELDS:
        raise ValueError(
            f"{zip_path.name}: expected {N_FIELDS} pipe-delimited fields, "
            f"observed {width}. The schema contract "
            f"({SCHEMA_CONTRACT_VERSION_HINT}) does not match this file."
        )
    loan_id = profiles[1]
    act_period = profiles[2]
    problems: list[str] = []
    if loan_id.fill_rate < 0.999:
        problems.append(f"LOAN_ID fill rate {loan_id.fill_rate:.4f} < 0.999")
    if not (loan_id.min_len >= 8):
        problems.append(f"LOAN_ID min length {loan_id.min_len} unexpectedly short")
    if act_period.min_len != 6 or act_period.max_len != 6:
        problems.append("ACT_PERIOD is not uniformly 6 characters (MMYYYY)")
    return {
        "file": zip_path.name,
        "field_count": width,
        "contract_ok": not problems,
        "problems": problems,
        "profiles": profiles,
    }


SCHEMA_CONTRACT_VERSION_HINT = "fnma_sf_llp_113col_v1"


__all__ = [
    "Field", "FIELDS", "N_FIELDS", "FIELD_BY_IDX", "FIELD_BY_NAME",
    "USED_IDX", "USED_NAMES", "STATIC_FIELDS", "MONTHLY_FIELDS",
    "NUMERIC_RAW", "DATE_RAW", "CANONICAL_STATIC", "CANONICAL_MONTHLY",
    "RENAME_MAP", "PURPOSE_MAP", "PROPERTY_MAP", "OCCUPANCY_MAP",
    "CHANNEL_MAP", "MI_TYPE_MAP", "ZERO_BAL_MAP", "DLQ_UNKNOWN",
    "IDENTIFIER", "TIME", "ACQUISITION", "PERFORMANCE", "CREDIT_EVENT",
    "UNUSED", "ColumnProfile", "discover_layout", "verify_contract",
]

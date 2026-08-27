"""
Central configuration for the Loan Performance Intelligence Engine (LPIE).

Per dataset_usage.md Section 16 (Reproducibility Guidelines), every pipeline
parameter lives here in a versioned config rather than being hard-coded at the
call site, so a run can be reconstructed from (code version, config version,
data version).

Nothing in this module depends on wall-clock time or on a random number
generator without an explicit seed -- reproducibility is a hard requirement
(prd.md Section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Versions -- bumped deliberately, logged with every artifact
# ---------------------------------------------------------------------------
CONFIG_VERSION = "1.0.0"
FEATURE_SET_VERSION = "fs_v1"
SCHEMA_CONTRACT_VERSION = "fnma_sf_llp_113col_v1"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"        # intentionally empty; see README
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = DATA_DIR / "splits"
SUPPORTING_DIR = DATA_DIR / "supporting"
REFERENCE_DIR = DATA_DIR / "reference"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
LLM_LOG_DIR = LOGS_DIR / "llm"
DATABASE_DIR = PROJECT_ROOT / "database"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

for _d in (
    DATA_DIR, RAW_DIR, EXTRACTED_DIR, PROCESSED_DIR, FEATURES_DIR, SPLITS_DIR,
    SUPPORTING_DIR, REFERENCE_DIR, REPORTS_DIR, DOCS_DIR, MODELS_DIR,
    LOGS_DIR, LLM_LOG_DIR, DATABASE_DIR, SUBMISSION_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IngestConfig:
    """
    Controls the streaming ingestion pass over data/raw/*.zip.

    Why streaming and not extraction: the 16 ZIPs total ~15 GB compressed but
    ~200 GB uncompressed (2020Q4.csv alone is 26.5 GB) against 56 GB of free
    disk. Extraction is physically impossible, so every ZIP member is read
    through zipfile.open() straight into a chunked pandas reader. This also
    makes the dataset_usage.md Section 3.1 immutability guarantee structural
    rather than procedural: raw bytes are opened read-only and never rewritten.
    """
    # Chunk size is memory-bound, not throughput-bound. Measured free RAM on
    # this workstation is ~2.5 GB of 16.8 GB, so a 200K-row chunk of 45 string
    # columns (~250 MB resident) times 3 workers keeps the whole pass inside
    # ~1.5 GB. Raise both if more headroom is available.
    chunk_size: int = 200_000

    # Deterministic loan-level sample rate for the modelling panel.
    # Loans are sampled (never rows) so each selected loan keeps its COMPLETE
    # trajectory -- required for forward-looking labels, rolling features and
    # survival analysis, and it simultaneously delivers the loan-level
    # containment demanded by dataset_usage.md Section 8.3.
    sample_rate: float = 0.03

    # Deterministic hash salt. Fixed => the same loans are selected on every
    # run, on every machine. This is what makes the sample reproducible.
    sample_salt: str = "lpie-v1"

    # Full-population aggregates are computed from 100% of rows regardless of
    # the sample, so portfolio reporting describes the entire universe.
    compute_population_aggregates: bool = True

    n_workers: int = 3          # memory-bound, not core-bound; see chunk_size
    progress_every: int = 20    # log every N chunks


# ---------------------------------------------------------------------------
# Business rule thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleConfig:
    """
    Domain thresholds that define delinquency, default and prepayment.

    prd.md Section 4.2 requires "a defined forward horizon" and "a defined
    default state" without fixing the numbers, so they are set here explicitly
    and reported in the model card.
    """
    # Delinquency: 30+ days past due. DLQ_STATUS is expressed in *months*
    # delinquent, so 1 month == 30 days.
    delinquency_dlq_months: int = 1

    # Default: 180+ days delinquent (the D180 industry convention) OR a
    # credit-event zero-balance termination. Assumption A4 in
    # docs/documentation_analysis.md; a D90 variant is also reported.
    default_dlq_months: int = 6
    default_dlq_months_alt: int = 3

    # Zero_Bal_Code groupings (Fannie Mae published code list).
    zb_prepaid: tuple = ("01",)                       # prepaid or matured
    zb_credit_event: tuple = ("02", "03", "09", "15")  # 3rd-party/short sale, REO, note sale
    zb_repurchase: tuple = ("06",)

    # Forward label horizons, in months.
    horizon_delinquency_short: int = 3
    horizon_delinquency_medium: int = 6
    horizon_default: int = 12
    horizon_prepayment: int = 12

    # Rolling feature windows, in months.
    rolling_windows: tuple = (3, 6, 12)


# ---------------------------------------------------------------------------
# Banding -- the schema bridge from raw continuous fields to the documented
# categorical bands used by api_spec.md and database_schema.md.
# Cut-points are chosen to reproduce the literal examples in api_spec.md
# ("680-719", "80-90", "36-43"). The continuous source value is ALWAYS retained
# alongside the band, so models get full resolution while the API contract holds.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandConfig:
    credit_score_edges: tuple = (0, 580, 620, 660, 680, 700, 720, 740, 760, 780, 851)
    credit_score_labels: tuple = (
        "<580", "580-619", "620-659", "660-679", "680-699",
        "700-719", "720-739", "740-759", "760-779", "780+",
    )

    ltv_edges: tuple = (0, 60, 70, 80, 85, 90, 95, 97, 1000)
    ltv_labels: tuple = ("<60", "60-70", "70-80", "80-85", "85-90", "90-95", "95-97", "97+")

    dti_edges: tuple = (0, 20, 28, 36, 43, 50, 1000)
    dti_labels: tuple = ("<20", "20-28", "28-36", "36-43", "43-50", "50+")

    unknown_label: str = "Unknown"


# ---------------------------------------------------------------------------
# Time-aware split (decision.md ADR-2, dataset_usage.md Section 8)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """
    Calendar-time split. NEVER a random row split -- that is an explicit
    disqualification condition (decision.md ADR-2).

    Boundaries follow the master specification: train 2018-2020,
    validate 2021, test the latest fully-labelled period. The test window is
    resolved from the data at runtime (see resolve_test_window) rather than
    hard-coded, because the forward label horizon determines how late a month
    can still carry a complete label.
    """
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"

    # Contiguous secondary holdout reported alongside the primary test window.
    holdout_2022_start: str = "2022-01-01"
    holdout_2022_end: str = "2022-12-31"

    # Rows whose forward-label window would cross the split boundary are
    # dropped rather than trained on partially-known outcomes
    # (dataset_usage.md Section 8.3).
    truncate_labels_at_boundary: bool = True


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 42
    n_jobs: int = 6

    # Row cap per training matrix. The sampled panel is large; tree ensembles
    # converge well before consuming every row, and a fixed seeded cap keeps
    # training reproducible and bounded. Sampling here is at ROW level within
    # an already loan-contained split, so it cannot introduce loan leakage.
    max_train_rows: int = 3_000_000
    max_shap_rows: int = 20_000

    calibration_method: str = "isotonic"

    risk_thresholds: tuple = (0.02, 0.10, 0.25)   # low | medium | high | critical

    targets: tuple = (
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
    )
    multiclass_target: str = "next_state"

    loan_states: tuple = ("Current", "Delinquent", "Default", "Prepaid", "Closed")


@dataclass(frozen=True)
class AnomalyConfig:
    contamination: float = 0.02
    n_estimators: int = 200
    max_samples: int = 100_000
    random_seed: int = 42
    min_examples_in_report: int = 20


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Stress assumptions for the three required scenarios. Persisted to
    data/supporting/macro_scenarios.csv (declared synthetic per
    dataset_usage.md Section 13.3).
    """
    names: tuple = ("base", "adverse_credit", "high_prepayment")
    segment_types: tuple = ("vintage", "credit_band", "state", "servicer")
    top_n_segments: int = 12


@dataclass(frozen=True)
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    bands: BandConfig = field(default_factory=BandConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    config_version: str = CONFIG_VERSION
    feature_set_version: str = FEATURE_SET_VERSION
    schema_version: str = SCHEMA_CONTRACT_VERSION


CFG = Config()


def raw_zips() -> list[Path]:
    """Sorted list of the raw quarterly ZIP archives."""
    return sorted(RAW_DIR.glob("*.zip"))


__all__ = [
    "CFG", "Config", "IngestConfig", "RuleConfig", "BandConfig", "SplitConfig",
    "ModelConfig", "AnomalyConfig", "ScenarioConfig",
    "CONFIG_VERSION", "FEATURE_SET_VERSION", "SCHEMA_CONTRACT_VERSION",
    "PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "EXTRACTED_DIR", "PROCESSED_DIR",
    "FEATURES_DIR", "SPLITS_DIR", "SUPPORTING_DIR", "REFERENCE_DIR",
    "REPORTS_DIR", "DOCS_DIR", "MODELS_DIR", "LOGS_DIR", "LLM_LOG_DIR",
    "DATABASE_DIR", "SUBMISSION_DIR", "raw_zips",
]

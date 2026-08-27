# Database Schema — Loan Performance Intelligence Engine

## 1. Database Overview

### Why PostgreSQL

PostgreSQL is the system of record for this platform for the following reasons:

- **Relational integrity for financial data.** Loans, monthly performance records, predictions, anomalies, and LLM logs are all tightly related entities with strict foreign-key relationships. PostgreSQL's constraint enforcement (FK, CHECK, UNIQUE) prevents orphaned or inconsistent records, which is critical in a lending/audit context.
- **Native JSONB support.** Model explanations (SHAP values), scenario assumptions, and LLM prompt/response payloads are semi-structured. PostgreSQL's `JSONB` type stores this data natively while remaining queryable and indexable (GIN indexes), avoiding the need for a separate document store for these use cases.
- **Strong support for time-series/panel data.** The `loan_performance` table is a monthly panel dataset (loan × month). PostgreSQL's native partitioning (range partitioning by `month`) scales well to hundreds of millions of rows while keeping query planning efficient.
- **Numeric precision.** Financial fields (balance, interest rate, probabilities) require the `NUMERIC`/`DECIMAL` type for exact precision — critical to avoid floating-point drift in monetary and probability calculations.
- **Mature extension ecosystem.** `pg_partman` for partition management, `pg_stat_statements` for query performance monitoring, `pgvector` (optional) if RAG-style embedding search over the data dictionary is added later, and `TimescaleDB` as an optional extension for the performance panel if very high ingestion rates are needed.
- **ACID compliance.** Every prediction, anomaly flag, and reviewer decision must be auditable and consistent — a strict requirement in credit-risk workflows. PostgreSQL guarantees this without the eventual-consistency trade-offs of NoSQL alternatives.
- **Ecosystem fit with FastAPI/Python.** SQLAlchemy 2.0 + `asyncpg`/`psycopg` gives first-class async ORM support, and Alembic handles versioned schema migrations cleanly.

---

## 2. Schema Design

### 2.1 `users`

Stores platform users: reviewers, data scientists, and admins.

| Column | Type | Constraints |
|---|---|---|
| id | UUID | PRIMARY KEY, DEFAULT gen_random_uuid() |
| name | VARCHAR(150) | NOT NULL |
| email | VARCHAR(255) | NOT NULL, UNIQUE |
| password_hash | TEXT | NOT NULL |
| role | VARCHAR(30) | NOT NULL, CHECK (role IN ('admin','reviewer','data_scientist','viewer')) |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() |

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role VARCHAR(30) NOT NULL CHECK (role IN ('admin','reviewer','data_scientist','viewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
```

---

### 2.2 `loans`

Static, origination-level attributes. One row per loan (the "dimension" table).

| Column | Type | Constraints |
|---|---|---|
| loan_id | VARCHAR(40) | PRIMARY KEY |
| original_balance | NUMERIC(14,2) | NOT NULL, CHECK (original_balance > 0) |
| current_balance | NUMERIC(14,2) | CHECK (current_balance >= 0) |
| interest_rate | NUMERIC(6,4) | NOT NULL, CHECK (interest_rate >= 0) |
| credit_score_band | VARCHAR(20) | NOT NULL |
| ltv_band | VARCHAR(20) | NOT NULL |
| dti_band | VARCHAR(20) | NOT NULL |
| state | CHAR(2) | NOT NULL |
| loan_purpose | VARCHAR(30) | NOT NULL |
| property_type | VARCHAR(30) | |
| occupancy_type | VARCHAR(30) | |
| origination_month | DATE | NOT NULL |
| remaining_term_months | INTEGER | CHECK (remaining_term_months >= 0) |
| servicer_name | VARCHAR(100) | |
| source_system | VARCHAR(50) | |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() |
| updated_at | TIMESTAMPTZ | NOT NULL, DEFAULT now() |

```sql
CREATE TABLE loans (
    loan_id VARCHAR(40) PRIMARY KEY,
    original_balance NUMERIC(14,2) NOT NULL CHECK (original_balance > 0),
    current_balance NUMERIC(14,2) CHECK (current_balance >= 0),
    interest_rate NUMERIC(6,4) NOT NULL CHECK (interest_rate >= 0),
    credit_score_band VARCHAR(20) NOT NULL,
    ltv_band VARCHAR(20) NOT NULL,
    dti_band VARCHAR(20) NOT NULL,
    state CHAR(2) NOT NULL,
    loan_purpose VARCHAR(30) NOT NULL,
    property_type VARCHAR(30),
    occupancy_type VARCHAR(30),
    origination_month DATE NOT NULL,
    remaining_term_months INTEGER CHECK (remaining_term_months >= 0),
    servicer_name VARCHAR(100),
    source_system VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_loans_state ON loans(state);
CREATE INDEX idx_loans_credit_band ON loans(credit_score_band);
CREATE INDEX idx_loans_origination_month ON loans(origination_month);
```

---

### 2.3 `loan_performance`

Monthly panel data — one row per loan per month. Partitioned by `reporting_month` (range partitioning) for scale (250K–1M+ rows growing monthly).

| Column | Type | Constraints |
|---|---|---|
| id | BIGSERIAL | part of composite PK |
| loan_id | VARCHAR(40) | NOT NULL, FK → loans(loan_id) |
| month_index | INTEGER | NOT NULL |
| reporting_month | DATE | NOT NULL |
| loan_age_months | INTEGER | |
| days_past_due | INTEGER | DEFAULT 0, CHECK (days_past_due >= 0) |
| current_status | VARCHAR(30) | NOT NULL |
| delinquency | BOOLEAN | NOT NULL DEFAULT false |
| default_flag | BOOLEAN | NOT NULL DEFAULT false |
| prepayment_flag | BOOLEAN | NOT NULL DEFAULT false |
| modification_flag | BOOLEAN | DEFAULT false |
| loss_severity_band | VARCHAR(20) | |
| document_status | VARCHAR(30) | |
| last_updated_at | TIMESTAMPTZ | |
| source_system | VARCHAR(50) | |

```sql
CREATE TABLE loan_performance (
    id BIGSERIAL,
    loan_id VARCHAR(40) NOT NULL REFERENCES loans(loan_id),
    month_index INTEGER NOT NULL,
    reporting_month DATE NOT NULL,
    loan_age_months INTEGER,
    days_past_due INTEGER DEFAULT 0 CHECK (days_past_due >= 0),
    current_status VARCHAR(30) NOT NULL,
    delinquency BOOLEAN NOT NULL DEFAULT false,
    default_flag BOOLEAN NOT NULL DEFAULT false,
    prepayment_flag BOOLEAN NOT NULL DEFAULT false,
    modification_flag BOOLEAN DEFAULT false,
    loss_severity_band VARCHAR(20),
    document_status VARCHAR(30),
    last_updated_at TIMESTAMPTZ,
    source_system VARCHAR(50),
    PRIMARY KEY (loan_id, reporting_month)
) PARTITION BY RANGE (reporting_month);

-- Example monthly partitions (created dynamically via pg_partman or migration job)
CREATE TABLE loan_performance_2026_01 PARTITION OF loan_performance
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE INDEX idx_perf_loan_id ON loan_performance(loan_id);
CREATE INDEX idx_perf_month ON loan_performance(reporting_month);
CREATE INDEX idx_perf_status ON loan_performance(current_status);
```

---

### 2.4 `features`

Engineered ML feature vectors per loan per snapshot month. Stored as JSONB for schema flexibility as feature sets evolve, plus a version tag for reproducibility.

```sql
CREATE TABLE features (
    id BIGSERIAL PRIMARY KEY,
    loan_id VARCHAR(40) NOT NULL REFERENCES loans(loan_id),
    reporting_month DATE NOT NULL,
    feature_set_version VARCHAR(20) NOT NULL,
    feature_vector JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (loan_id, reporting_month, feature_set_version)
);
CREATE INDEX idx_features_loan_month ON features(loan_id, reporting_month);
CREATE INDEX idx_features_gin ON features USING GIN (feature_vector);
```

---

### 2.5 `predictions`

Model outputs per loan per scoring run.

```sql
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    loan_id VARCHAR(40) NOT NULL REFERENCES loans(loan_id),
    reporting_month DATE NOT NULL,
    model_version VARCHAR(30) NOT NULL,
    default_probability NUMERIC(6,5) CHECK (default_probability BETWEEN 0 AND 1),
    delinquency_probability NUMERIC(6,5) CHECK (delinquency_probability BETWEEN 0 AND 1),
    prepayment_probability NUMERIC(6,5) CHECK (prepayment_probability BETWEEN 0 AND 1),
    next_state VARCHAR(30),
    confidence NUMERIC(6,5) CHECK (confidence BETWEEN 0 AND 1),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (loan_id, reporting_month, model_version)
);
CREATE INDEX idx_predictions_loan ON predictions(loan_id);
CREATE INDEX idx_predictions_month_model ON predictions(reporting_month, model_version);
```

---

### 2.6 `anomaly_results`

```sql
CREATE TABLE anomaly_results (
    id BIGSERIAL PRIMARY KEY,
    loan_id VARCHAR(40) NOT NULL REFERENCES loans(loan_id),
    reporting_month DATE NOT NULL,
    anomaly_score NUMERIC(6,5) NOT NULL CHECK (anomaly_score BETWEEN 0 AND 1),
    exception_type VARCHAR(50),
    reason TEXT,
    detector_version VARCHAR(30) NOT NULL,
    reviewed BOOLEAN NOT NULL DEFAULT false,
    reviewed_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_anomaly_loan ON anomaly_results(loan_id);
CREATE INDEX idx_anomaly_score ON anomaly_results(anomaly_score DESC);
CREATE INDEX idx_anomaly_type ON anomaly_results(exception_type);
```

---

### 2.7 `explanations`

Stores SHAP values and feature importance per prediction, linked via `prediction_id`.

```sql
CREATE TABLE explanations (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    loan_id VARCHAR(40) NOT NULL REFERENCES loans(loan_id),
    shap_values JSONB NOT NULL,
    global_feature_importance JSONB,
    top_drivers JSONB,
    model_version VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_explanations_prediction ON explanations(prediction_id);
CREATE INDEX idx_explanations_loan ON explanations(loan_id);
CREATE INDEX idx_explanations_gin ON explanations USING GIN (shap_values);
```

---

### 2.8 `scenarios`

Stores simulation/stress-test outputs (base, adverse-credit, high-prepayment).

```sql
CREATE TABLE scenarios (
    id BIGSERIAL PRIMARY KEY,
    scenario_name VARCHAR(50) NOT NULL CHECK (scenario_name IN ('base','adverse_credit','high_prepayment')),
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    segment_type VARCHAR(30),
    segment_value VARCHAR(50),
    projected_delinquency_rate NUMERIC(6,5),
    projected_default_rate NUMERIC(6,5),
    projected_prepayment_rate NUMERIC(6,5),
    assumptions JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_scenarios_run ON scenarios(run_id);
CREATE INDEX idx_scenarios_name ON scenarios(scenario_name);
```

---

### 2.9 `llm_logs`

Governance log for every LLM call — required for the "governed AI copilot" requirement.

```sql
CREATE TABLE llm_logs (
    id BIGSERIAL PRIMARY KEY,
    loan_id VARCHAR(40) REFERENCES loans(loan_id),
    user_id UUID REFERENCES users(id),
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    grounding_sources JSONB,
    approval_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (approval_status IN ('pending','approved','rejected','corrected')),
    reviewer_notes TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_llm_logs_loan ON llm_logs(loan_id);
CREATE INDEX idx_llm_logs_status ON llm_logs(approval_status);
CREATE INDEX idx_llm_logs_timestamp ON llm_logs(timestamp DESC);
```

---

## 3. Entity-Relationship Diagram

```
users(id) ────────────────────────────┐
                                       │ reviewed_by / user_id
loans(loan_id) 1───* loan_performance  │
    │                                  │
    │ 1───* features                   │
    │ 1───* predictions ──1───* explanations
    │ 1───* anomaly_results ───────────┘
    │ 1───* llm_logs
    │
scenarios (segment-level, not FK-bound to a single loan)
```

Textual relationship summary:

- `users` 1 — * `anomaly_results` (reviewed_by)
- `users` 1 — * `llm_logs` (user_id)
- `loans` 1 — * `loan_performance`
- `loans` 1 — * `features`
- `loans` 1 — * `predictions`
- `loans` 1 — * `anomaly_results`
- `loans` 1 — * `llm_logs`
- `predictions` 1 — * `explanations`
- `scenarios` is a standalone aggregate/simulation-output table, not directly FK-linked to individual loans (segment-level by design), but `assumptions.run_id` correlates to a scenario execution.

## 4. Indexing Strategy

- **Foreign key columns** (`loan_id` everywhere) are indexed to support fast joins across the star-schema-like structure (loans as the dimension, performance/predictions/anomalies as facts).
- **JSONB GIN indexes** on `features.feature_vector`, `explanations.shap_values` support ad-hoc querying (e.g., "find loans where feature X > threshold") without needing a separate feature store initially.
- **Composite indexes** on `(reporting_month, model_version)` in `predictions` support fast retrieval of "latest scoring run" queries.
- **Partitioning** on `loan_performance(reporting_month)` keeps query plans efficient at scale and enables cheap partition-level archival/drop of old months.
- **Descending index** on `anomaly_results.anomaly_score` supports "top N anomalies" dashboard queries directly.

## 5. Constraints & Data Integrity

- All probability fields are constrained to `[0, 1]` via `CHECK`.
- All monetary fields are `NUMERIC`, never `FLOAT`, to avoid rounding errors in balances.
- `ON DELETE CASCADE` is used only for `explanations` (child of `predictions`) since explanations have no independent lifecycle.
- `loans.loan_id` is the natural key and primary key across the schema (no surrogate ID needed since it's globally unique and stable from source systems).
- Enum-like fields (`role`, `scenario_name`, `approval_status`) use `CHECK` constraints rather than native `ENUM` types, to allow easier schema evolution without `ALTER TYPE` migrations.

## 6. Normalization

The schema is normalized to **3NF**:
- `loans` holds only origination-level (static) attributes — no repeating monthly data.
- `loan_performance` holds only time-varying attributes, referencing `loans` by foreign key — eliminating redundant storage of static attributes per month.
- `features`, `predictions`, `explanations`, `anomaly_results` are each single-purpose fact tables, avoiding mixed-grain tables.
- Deliberate **denormalization** is limited to JSONB payloads (`feature_vector`, `shap_values`, `assumptions`) where the schema is inherently semi-structured and evolves faster than a fully normalized relational structure would tolerate — a standard trade-off in ML feature/metadata storage.

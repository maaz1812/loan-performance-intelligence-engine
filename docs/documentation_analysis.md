# Documentation Analysis — Loan Performance Intelligence Engine (LPIE)

**Project:** Loan Performance Intelligence Engine
**Track:** Intain Campus FinTech Challenge 2026 — AI Track
**Document purpose:** Phase 0 deliverable. Consolidated reading of all nine source specification documents, the resulting architecture understanding, the implementation roadmap, inter-module dependencies, and the technology decisions that govern the build.

**Status:** Complete. All nine source documents read in full before any implementation code was written.

---

## 0. Reading Manifest

| # | Source document | Size | Role in the build |
|---|---|---|---|
| 1 | `prd.md` | 12.3 KB | Product requirements — the *what* and *for whom* |
| 2 | `systemdesign.md` | 9.7 KB | Layered architecture and data flow — the *shape* |
| 3 | `backend.md` | 14.4 KB | Backend layering, services, security — the *API core* |
| 4 | `frontend.md` | 9.9 KB | Pages, components, state, visualization — the *UI* |
| 5 | `database_schema.md` | 14.9 KB | Nine-table PostgreSQL schema — the *system of record* |
| 6 | `api_spec.md` | 8.1 KB | Seven endpoint contracts — the *frontend/backend treaty* |
| 7 | `orchestrator.md` | 11.4 KB | Seven-stage ML lifecycle DAG — the *offline pipeline* |
| 8 | `decision.md` | 12.6 KB | Nine ADRs — the *why*, and the constraints I may not violate |
| 9 | `dataset_usage (1).md` | 49.7 KB | Dataset lifecycle + explicit Claude Code instruction set — the *data contract* |

> **Filename note:** the master prompt refers to `dataset_usage.md`; the file on disk is `dataset_usage (1).md`. Same document, treated as authoritative. No other document is missing.

---

## 1. Summary of Every Source Document

### 1.1 `prd.md` — Product Requirements

**Problem.** Loan portfolios emit large volumes of monthly performance data from multiple source systems (origination platforms, servicers, document repositories). That data is incomplete, inconsistently formatted, and prone to *silent* quality failure — stale updates, conflicting source records, broken cross-field relationships. Institutions cannot tell whether incoming data is trustworthy until after it has corrupted a decision.

**Four compounding business problems:** (1) data quality is invisible until it causes damage; (2) manual loan-by-loan review does not scale to hundreds of thousands of loans refreshed monthly; (3) black-box scores cannot be defended to auditors or credit committees; (4) legacy workflows are backward-looking reporting rather than forward-looking prediction.

**Users and their core need:**

| User | Core need |
|---|---|
| Loan Analysts | Triage flagged loans fast; understand *why* a loan was flagged |
| Risk Managers | Portfolio risk concentration, scenario outcomes, trend shifts |
| Data Scientists / ML Engineers | Build, validate, calibrate, monitor models |
| Financial Institutions | Auditable, governed, reproducible intelligence for compliance |

**Five product objectives:** analyze portfolios, predict risk, detect unreliable records, generate explanations, assist reviewers.

**Functional requirements**, in five blocks:
- **4.1 Data Intelligence** — versioned batch upload; schema validation with per-row reason codes and quarantine (never silent drop); per-field missingness at record and batch level plus *change detection versus a prior batch*; statistical outlier detection **and** logical relationship breaks; record-level and batch-level data-quality scoring.
- **4.2 ML Prediction** — four independently trained outcomes: default (12-month forward), delinquency (3- and 6-month), prepayment (12-month forward), next-state (multi-class). Mandatory time-aware split, evaluated on ROC-AUC, PR-AUC, F1, recall-at-fixed-precision, Brier, macro-F1.
- **4.3 Anomaly Detection** — unsupervised/semi-supervised scoring *layered on top of* deterministic rules; exception-type classification; per-record driver explanation.
- **4.4 Scenario Simulation** — base, adverse-credit, high-prepayment; each must produce portfolio **and segment-level** projections (vintage, credit band, state, servicer) plus plain-language driver explanation.
- **4.5 Explainable AI** — global feature importance per model; local SHAP per loan; reviewer-facing view combining prediction, confidence/uncertainty, and top drivers in non-data-scientist language.
- **4.6 LLM Reviewer Copilot** — an explicit allow/deny list. **Allowed:** summarize a loan's risk profile from structured model output, draft reviewer notes grounded in retrieved definitions, explain data-dictionary fields, summarize scenarios. **Not allowed:** make or imply a final credit decision, generate or substitute for a probability/risk score, produce ungrounded claims. Every interaction logged; every output labeled **recommendation**, never decision.

**Explicit non-objective (§3):** "LPIE does not use an LLM as the source of predictive risk scores." This is the single hardest architectural constraint in the project and is reinforced by `systemdesign.md` §2, `decision.md` ADR-1, and `orchestrator.md` §6.4.

**Non-functional requirements:** RBAC across four roles; encryption in transit and at rest; scale 250K → 1M+ monthly loan-month records *without redesign*; idempotent and resumable pipelines; every prediction/flag/LLM output traceable to a model version + data snapshot; bit-exact reproducibility given the same snapshot and model version.

**User stories** are grouped by the four roles and are the acceptance criteria for the frontend: prioritized flag queue, plain-language explanation, accept/reject/annotate LLM notes, data-quality visibility on a record, concentration dashboard, scenario comparison, calibration metrics, LLM audit trail, reproducible retraining, drift comparison, auto-populated model card, role management, job monitoring, model versioning/rollback.

---

### 1.2 `systemdesign.md` — System Design

Six layers: Frontend → Backend API → Orchestrator → ML Pipeline → Database, with an LLM Layer *alongside* rather than inside the prediction path.

**The load-bearing sentence (§2):** "the LLM Layer is reachable only through the Orchestrator, and only ever *reads* ML outputs and reference data — it has no path to write predictions, override scores, or bypass the ML Pipeline. This enforces the ML-first, LLM-assists requirement **structurally, not just by convention**."

I read this as a hard instruction about *code structure*, not documentation prose: the LLM module must have no import path to, and no write capability over, the prediction tables. I implement this as a physical separation (LLM module receives already-computed model output as read-only input and can only write to `llm_logs`) and assert it with a test.

- **Frontend** — dashboard, charts, reports; pure API consumer, no business logic, no direct DB/model access.
- **Backend API** — REST endpoints, request validation, RBAC, translating UI actions into orchestrator commands. Explicitly: "the backend never performs model inference or LLM calls directly."
- **Orchestrator** — the only component permitted to invoke both ML Pipeline and LLM Layer; enforces execution order (predictions before explanations before LLM summaries); writes a run-level audit record. §4's sequence diagram fixes the exact order.
- **ML Layer** — six sub-modules: Data Intelligence, Prediction, Survival/Transition, Anomaly, Scenario, Explainability.
- **Database Layer** — relational store, loan data versioned by ingestion snapshot, predictions tagged with producing model version, full LLM prompt/response audit trail.
- **LLM Layer** — RAG over data dictionary + validation rules; summaries; reviewer assistance; every output labeled and stored with grounding context.
- **Data Flow (§8)** — ingestion → validation → feature engineering → model inference → explanation generation → LLM copilot. "The LLM Copilot only engages after step 5."
- **Deployment (§9)** — gateway, stateless backend, orchestrator dispatching queued jobs to an ML worker pool, LLM service, relational DB, object storage for snapshots and model artifacts, monitoring.

---

### 1.3 `backend.md` — Backend Architecture

FastAPI on Python 3.11+, six-layer separation: Routes → Controllers → Services → Repositories → Database + Model Store.

- **Routes** (`app/api/routes/`) — seven thin routers (auth, loan, prediction, anomaly, scenario, explanation, llm). "Routes never contain business logic."
- **Controllers** — orchestrate one use case each; the boundary between web and domain.
- **Services** — seven services (`AuthService`, `LoanService`, `PredictionService`, `AnomalyService`, `ScenarioService`, `ExplainabilityService`, `LLMService`), deliberately **framework-agnostic (no FastAPI imports)** so batch jobs and the orchestrator can reuse them.
- **Repositories** — nine repositories; all SQL/ORM access encapsulated; return domain objects, not ORM rows. "No service talks to the database directly."
- **Database layer** — SQLAlchemy 2.0 async + asyncpg, Alembic migrations, pool_size=20/max_overflow=10, request-scoped `AsyncSession`.
- **ML service layer (§2.6)** — the key performance constraint: "**model artifacts are never loaded per-request**." A `ModelRegistry` in-process cache keyed by `model_version` loads at startup; SHAP explainers cached alongside their model; feature vectors read from the `features` table rather than recomputed synchronously.

**Cross-cutting (§3):** JWT auth (15-min access, 7-day refresh, bcrypt); `require_role` RBAC dependency; per-service business logic detail; parameterized queries only ("no raw SQL string concatenation is permitted"); Redis caching with explicit pub/sub invalidation; structlog JSON logging with a propagated correlation ID; centralized exception handlers mapping domain errors to a consistent envelope `{error, detail, request_id}`; TLS, rate limiting on `/predict`, `/simulate`, `/review-summary`, secrets from a manager, field-level PII control.

**§3.7 and §3.12 restate the LLM guardrail at the service layer:** LLM responses "are never used to directly mutate `predictions` or `loans` — they are advisory-only and routed through the `llm_logs` approval workflow, enforced at the service layer, not just the UI."

---

### 1.4 `frontend.md` — Frontend Architecture

React 18 + TypeScript + Tailwind, Vite.

**Seven pages:** Dashboard `/`, Loan Analysis `/loans/:loanId`, Risk Prediction `/predictions`, Anomaly Detection `/anomalies`, Scenario Simulator `/scenarios`, Reports `/reports`, AI Reviewer `/ai-reviewer`.

> The master prompt lists six pages (Portfolio Overview, Loan Analysis, Risk Prediction, Anomaly Detection, Scenario Simulator, AI Reviewer). `frontend.md` adds **Reports**. I build all seven — the prompt's set is a subset, and Reports is also required to surface the Phase 16 deliverables.

**Component tree** under `frontend/src/`: `pages/`, `components/{charts,tables,forms,cards,layout}/`, `hooks/`, `api/`, `store/`, `types/`, `utils/`.

**State rule of thumb:** "if it comes from the API, it lives in React Query cache; if it's purely UI/interaction state, it lives in a Zustand store. Avoid duplicating server data into Zustand."

**Four visualization families:** Risk Charts (trend + segment bars), SHAP Charts (global horizontal bar of mean |SHAP|, local waterfall/force), Scenario Graphs (grouped comparison with segment toggle), Loan Timelines (monthly status transitions with exceptions overlaid). All chart components take a standardized `{data, loading, error}` prop shape.

**User flow (§3)** is the demo script: login → dashboard → loan → prediction → anomaly triage → scenario → AI reviewer note with Accept/Edit/Reject → reports.

**Cards** carry a governance requirement: LLM note cards must be "visually distinct — e.g., a bordered 'AI Suggestion' style — to reinforce that it's a recommendation, not a decision."

Error handling, code splitting, virtualized tables, memoization, debounced filters, and chart data-point capping are specified in §5–6.

---

### 1.5 `database_schema.md` — Database Schema

PostgreSQL rationale: relational integrity for financial data, native JSONB for semi-structured SHAP/assumptions/LLM payloads, range partitioning for the monthly panel, `NUMERIC` for exact monetary and probability precision, mature extensions, ACID, and FastAPI/SQLAlchemy ecosystem fit.

**Nine tables:** `users`, `loans`, `loan_performance` (partitioned by `reporting_month`, PK `(loan_id, reporting_month)`), `features` (JSONB vector + `feature_set_version`, unique on `(loan_id, reporting_month, feature_set_version)`), `predictions` (unique on `(loan_id, reporting_month, model_version)`), `anomaly_results`, `explanations` (FK to `predictions`, `ON DELETE CASCADE`), `scenarios` (segment-level, deliberately not FK-bound to a loan), `llm_logs` (approval_status ∈ pending/approved/rejected/corrected).

**Integrity rules I must honour:** probabilities `CHECK BETWEEN 0 AND 1`; money as `NUMERIC` never `FLOAT`; `CASCADE` only on `explanations`; `loans.loan_id` is the natural PK throughout; enum-like fields use `CHECK` not native `ENUM`; schema normalized to 3NF with denormalization confined to JSONB payloads.

Indexing strategy: FK columns, JSONB GIN, composite `(reporting_month, model_version)`, partitioning, and a **descending** index on `anomaly_results.anomaly_score` for top-N dashboard queries.

---

### 1.6 `api_spec.md` — API Specification

Seven endpoints, JWT bearer except `/login`:

| Endpoint | Purpose | Notable contract detail |
|---|---|---|
| `POST /login` | Auth | Returns access + refresh + `expires_in: 900` + user{id,name,role} |
| `GET /loans` | List/filter/paginate | Filters state, credit_score_band, loan_purpose; page_size max 200 |
| `GET /loan/{id}` | Single loan detail | Static attributes + nested `latest_performance` |
| `POST /predict` | Score a loan | Returns four probabilities + next_state + risk_level + confidence + model_version + scored_at |
| `GET /anomalies` | Reviewer triage | Filters min_score, exception_type, reviewed; sortable by score |
| `POST /simulate` | Scenario run | Returns run_id, three projected rates, top_drivers |
| `GET /explanation/{loan_id}` | Explainability | `top_drivers[]` with shap_value + direction, plus `global_feature_importance[]` |
| `POST /review-summary` | LLM copilot | Returns `is_recommendation: true`, `grounding_sources[]`, `model_name`, `log_id`, `approval_status` |

Standard error envelope `{error, detail, request_id}` and a 10-row global status-code table (400/401/403/404/409/422/429/500/502/503). Notably `409 features_not_available` and `503 model_not_ready` — both are real states my `PredictionService` must be able to reach.

**Contract-level governance:** `is_recommendation: true` is a non-removable field on the LLM response, and `grounding_sources` must cite the specific prediction/anomaly/dictionary rows used.

---

### 1.7 `orchestrator.md` — ML Workflow Orchestrator

Governs the **offline** lifecycle — everything before a model is servable. Seven stages, each "an independent, idempotent task with clearly defined inputs/outputs, so any stage can be re-run in isolation."

1. **Data Ingestion** — schema-on-read validation before staging; reconcile `servicer_updates` and **flag conflicts for the anomaly pipeline rather than silently overwriting**.
2. **Validation** — deterministic `validation_rules.json` checks; record-level DQ score + batch summary; hard failures **quarantined, not dropped**, and routed to the anomaly detector as candidate exceptions.
3. **Feature Pipeline** — rolling delinquency counts, payment-trend deltas, band encodings, time-since-origination, servicer aggregates; every feature set versioned; train/test drift (PSI or KS) computed here.
4. **Training** — time-aware split where training strictly precedes validation in `reporting_month` **and the same `loan_id` is never split across train and validation**; GBM classifiers for the four targets; survival/discrete-time transition model; anomaly detector trained on validated clean records but scored against the full population; baselines trained alongside.
5. **Evaluation** — the six metrics, calibration curves + post-hoc calibration, comparison against baseline *and* current production with a minimum uplift gate, FP/FN analysis.
6. **Deployment** — register artifact + SHAP explainer + calibration transform; promote staging→production; auto-generate the model card.
7. **Monitoring** — prediction-distribution drift, DQ score trend, anomaly-rate trend, alerting that triggers retraining review "rather than silent degradation."

**Scheduling:** Airflow DAG (Prefect acceptable alternative); monthly full pipeline, daily incremental validation/anomaly; retraining triggered not purely calendar-based; backfill supported because every stage reads/writes versioned dated tables.

**MLflow** for experiment tracking and a per-target model registry with staging→production→archived lifecycle, each version linked to its exact `feature_set_version` and data snapshot.

**§6 Agentic Workflow** is itself a deliverable: AI coding assistant usage, an experiment runner that may propose but never promote, prompt logging into an **AI Development Log** kept separate from the runtime `llm_logs` table, and **three hard human approval gates** — model promotion, LLM summary approval, anomaly review. "No anomaly is auto-resolved by the pipeline."

**§7 Idempotency:** re-running a task for the same `reporting_month`/`feature_set_version` upserts rather than duplicates; failed tasks halt only downstream dependents.

---

### 1.8 `decision.md` — Architecture Decision Records

Nine ADRs. These are binding constraints, and two of them describe *disqualification* conditions for the challenge.

| ADR | Decision | Why it binds me |
|---|---|---|
| 1 | ML models, not LLM, produce predictions | "The challenge explicitly disqualifies solutions that only send records to an LLM API for classification." |
| 2 | Time-aware validation | "This directly satisfies the challenge's disqualification condition against unjustified random splits that leak the same loan across train and validation." |
| 3 | XGBoost/LightGBM primary | Strong on tabular finance, native class-weighting, SHAP-compatible, fast retraining |
| 4 | PostgreSQL | FK traceability, JSONB, proven at 250K–1M+ rows/batch |
| 5 | FastAPI | Async for long jobs, Pydantic validation, same-language ML integration, auto OpenAPI |
| 6 | SHAP | Theoretically grounded, TreeSHAP fits ADR-3, additive attributions feed the LLM layer |
| 7 | RAG | Grounding satisfies the governance requirement; keeps the LLM from needing to "know" risk facts |
| 8 | Separate frontend/backend | Independent evolution, centralized authorization, independent scaling |
| 9 | Orchestrated pipelines | Enforces stage order, centralizes run audit, enables resumable partial-failure recovery |

Each ADR also states trade-offs I should acknowledge rather than hide — e.g. ADR-2 notes time-aware splits "yield less training data per fold and can be more sensitive to macro regime shifts," and ADR-6 notes "SHAP explains model behavior, not causal real-world relationships."

---

### 1.9 `dataset_usage (1).md` — Dataset Usage

The longest and most operationally prescriptive document. It doubles as an explicit agent instruction set (§17).

- **§1** Dataset is Fannie Mae Single-Family Loan Performance, chosen because it uniquely combines origination attributes with monthly performance history — required to move "from a static risk snapshot to a dynamic, time-aware performance model." §1.6 fixes the authority direction: "machine learning models produce the predictions; the LLM copilot only summarizes."
- **§2** Four data layers: `raw/` (immutable), `processed/`, `supporting/`, features. "Keeping `raw/` strictly immutable … ensures the pipeline is auditable."
- **§3** Chunked `read_csv` with explicit dtype hints; schema validation; type conversion; files failing validation **quarantined, not silently dropped**.
- **§4** Schema in five functional groups: identification (`loan_id`, `month_index`), financial (`original_balance`, `current_balance`, `interest_rate`), borrower/credit (`credit_score_band`, `ltv_band`, `dti_band`), time (`origination_month`, `reporting_month`, `loan_age_months`, `remaining_term_months`), performance (`current_status`, `days_past_due`, `default_flag`, `prepayment_flag`).
- **§5** Preprocessing: missingness classified as **structural vs reporting-lag vs genuinely missing**, each handled differently (structural encoded explicitly, lag forward-filled *with a carried-forward flag*, categorical → explicit `"Unknown"`, numeric → median-by-segment only when confirmed non-informative). Duplicates on `(loan_id, reporting_month)`: byte-identical extras dropped, differing rows routed to the anomaly pipeline. Invalid-record checks enumerated.
- **§6** Feature engineering in four families: financial (balance reduction ratio, utilization, remaining %), temporal (prior delinquency, rolling `days_past_due` mean/max over 3/6/12, cumulative missed payments), risk (composite credit risk score, debt burden, loan-age risk), interaction (credit × LTV, DTI × rate).
- **§7** Targets, all constructed **forward-only**: `next_3m_delinquency_flag`, `next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, and multi-class `next_state` over {Current, Delinquent, Default, Prepaid, Closed}.
- **§8** Why random splitting is wrong, in two parts — **same-loan leakage** (correlated monthly rows let the model memorize a trajectory) and **temporal leakage** (forward-looking targets mean a 2021-03 training row's label consumes 2021-04→2022-03 data). Mitigations: calendar-time split, loan-level containment, **forward-label horizon respected at the boundary** (rows whose label window crosses the cutoff excluded or documented), and no post-cutoff information in any rolling feature.
- **§9–12** Pipeline usage, per-model dataset usage, anomaly detection layering (rules + Isolation Forest + optional autoencoder), and scenario simulation (re-score the population under adjusted assumptions; portfolio **and** segment aggregation; comparison table annotated with top drivers).
- **§13** The three supporting files are explicitly labeled **synthetic**: `servicer_updates.csv` (second partially-overlapping source for conflict detection), `validation_rules.json` (machine-readable deterministic rules), `macro_scenarios.csv` (stress assumptions).
- **§14–16** Storage design by layer, four honest limitations (synthetic preprocessing differences, banded rather than granular borrower data, bias risk via geography/credit history, historical market dependency), and reproducibility guidelines (version every artifact, snapshot per run, config files not hard-coded constants, fixed logged seeds, experiment tracking).
- **§17 Claude Code Instructions** — the operational contract for this build:
  - **Rule 1** Never open raw files manually; never `pd.read_csv(whole_file)`; never modify `data/raw/`.
  - **Rule 2** Chunked processing, per chunk: read → validate → clean → transform → append → next.
  - **Rule 3** Discovery workflow: scan files, record name/size/columns/dtypes/missingness, classify columns into acquisition / performance / identifier / time, generate `data_dictionary.md` enumerating **every column actually discovered**, not a sample.
  - **Rule 4** The five required scripts, the three required output parquet datasets, the `loan_id` join logic, the five data-quality validations, the performance do/don't list, and snappy-compressed Parquet conversion.

---

## 2. Final Architecture Understanding

### 2.1 The one invariant that shapes everything

Three documents state the same constraint in three different vocabularies:

- `prd.md` §3: "LPIE does not use an LLM as the source of predictive risk scores."
- `systemdesign.md` §2: the LLM layer "has no path to write predictions … structurally, not just by convention."
- `decision.md` ADR-1: LLM-only classification is a **disqualification** condition.

Therefore the system is designed so that the LLM *cannot* influence a number even if prompted to. Concretely, in my implementation:

- All probabilities originate from serialized scikit-learn / XGBoost / LightGBM artifacts.
- The LLM module accepts an already-computed, read-only context object and returns prose.
- The LLM module's only write target is `llm_logs`.
- Every LLM output carries `is_recommendation: true` and a `grounding_sources` list.
- A test asserts the LLM module cannot import or write the prediction path.

### 2.2 Layer map, with the authority direction marked

```
                        ┌────────────────────────────┐
                        │  Frontend (React + TS)     │  pure API consumer
                        └─────────────┬──────────────┘
                                      │ REST /api/v1, JWT
                        ┌─────────────▼──────────────┐
                        │  Backend API (FastAPI)     │  routes → controllers →
                        │  thin, stateless, RBAC     │  services → repositories
                        └─────────────┬──────────────┘
                                      │ delegates; never infers directly
                        ┌─────────────▼──────────────┐
                        │      Orchestrator          │  ONLY caller of ML + LLM
                        │  enforces stage order      │  writes run audit record
                        └──────┬──────────────┬──────┘
                               │              │
              ┌────────────────▼───┐    ┌─────▼─────────────────┐
              │   ML Pipeline      │    │  LLM Layer (RAG)      │
              │  ─ data intel      │    │  reads ML output      │
              │  ─ prediction      │    │  writes ONLY llm_logs │
              │  ─ survival        │    │  no prediction path   │
              │  ─ anomaly         │    └─────┬─────────────────┘
              │  ─ scenario        │          │
              │  ─ explainability  │          │
              └────────┬───────────┘          │
                       │  writes              │  writes llm_logs only
                ┌──────▼──────────────────────▼──────┐
                │      Database (PostgreSQL)         │
                │  loans · loan_performance ·        │
                │  features · predictions ·          │
                │  explanations · anomaly_results ·  │
                │  scenarios · llm_logs · users      │
                └────────────────────────────────────┘
```

### 2.3 Execution order is a correctness property, not a preference

From `systemdesign.md` §4 and §8, and `orchestrator.md` §2, the order is fixed and each arrow is a precondition:

```
ingestion → validation → feature engineering → model inference
          → explanation generation → LLM copilot summary
```

"Predictions must exist before a reviewer summary can be generated." The orchestrator enforces this; stages are individually idempotent and resumable so a single stage can be re-run for one `reporting_month` without disturbing the rest.

---

## 3. Reality Check: Four Material Gaps Between the Documentation and the Delivered Data

Per master-prompt execution rule 4 ("If information is missing: document assumptions. Do not silently invent"), I verified the delivered data against the documentation *before* designing the pipeline. Four gaps require documented decisions. Each is recorded here and carried into `reports/data_dictionary.md` and `docs/progress_log.md`.

### Gap 1 — Schema mismatch: curated schema vs. raw Fannie Mae layout

**Documentation assumes** an organizer-curated, pre-banded CSV: `loan_id`, `credit_score_band`, `ltv_band`, `dti_band`, `current_status`, `default_flag`, `prepayment_flag`, `month_index`, with headers.

**Delivered data is** the *raw* Fannie Mae Single-Family Loan Performance file: **pipe-delimited, no header row, 113 positional fields**, with source names like `LOAN_ID`, `ACT_PERIOD`, `CURRENT_UPB`, `DLQ_STATUS`, `Zero_Bal_Code`, `CSCORE_B`, `OLTV`, `DTI`, `ORIG_UPB`. Verified empirically:

```
|100001040173|022018|R|Quicken Loans, Llc|Quicken Loans Inc.||4.250|4.250|453000.00||0.00|360|...
 ^POOL_ID     ^LOAN_ID ^ACT_PERIOD ^CHANNEL ^SELLER  ^SERVICER   ^ORIG_RATE ^CURR_RATE ^ORIG_UPB ^CURRENT_UPB
```

**Decision.** Build an explicit **schema-bridge layer** that maps the 113 raw positional fields onto the canonical LPIE schema the rest of the documentation depends on. Continuous source fields are banded into the documented categorical bands (`CSCORE_B` → `credit_score_band`, `OLTV` → `ltv_band`, `DTI` → `dti_band`), and `DLQ_STATUS` + `Zero_Bal_Code` are derived into `current_status` / `days_past_due` / `default_flag` / `prepayment_flag`. Every mapping and every band cut is documented field-by-field in `reports/data_dictionary.md`. Deriving targets from `DLQ_STATUS` and `Zero_Bal_Code` is the single highest-risk correctness step in the project and is unit-tested.

*Consequence:* the delivered data is **richer** than documented — real continuous credit scores, LTV, DTI, plus servicer, channel, MSA, forbearance and modification flags. I retain both the continuous value and the documented band, so band-based API contracts hold while models get full resolution.

### Gap 2 — Disk capacity: extraction is physically impossible

| Measure | Value |
|---|---|
| Compressed ZIPs on disk | **15 GB** (16 files) |
| Uncompressed total (from ZIP central directory) | **≈ 200 GB** |
| Largest single file | `2020Q4.csv` at **26.5 GB** |
| Free disk | **56 GB** |

The master prompt's Phase 1 structure includes `data/extracted/`, and Phase 2 lists an "Extraction" step. **Extracting even a third of the corpus would exhaust the disk.**

**Decision.** Stream every ZIP member directly through `zipfile.ZipFile.open()` into chunked `pandas.read_csv`. Nothing is ever extracted to disk. `data/extracted/` is retained as a documented, intentionally-empty staging directory with a `README` explaining why. This *strengthens* the immutability guarantee of `dataset_usage.md` §3.1 — raw files are opened read-only and byte-for-byte untouched, verified by checksum. The ZIPs are additionally `chmod 444`.

### Gap 3 — Volume: the panel is ~870 M loan-month rows

Measured read throughput is ~200 K rows/s with column pruning; at ~230 bytes/row the corpus is **≈ 870 million loan-month records**. A full scan is ~70 minutes single-threaded and is therefore feasible, but persisting and training on all 870 M rows is not appropriate on a 12-core workstation — and is not what the PRD asks for. `prd.md` NFR states the target operating scale is **"250K to 1M+ monthly loan-month records."**

**Decision — a three-output strategy that keeps full-population honesty while making modeling tractable:**

1. **`population_monthly_aggregates.parquet`** — computed from **100 % of all 870 M rows**, every file, every month. Portfolio-level monthly statistics. This means all portfolio reporting and profiling describes the *entire* universe, not a sample.
2. **`loan_static_attributes.parquet` + `loan_monthly_performance.parquet`** — the full monthly history of a **deterministic hash-based sample of `loan_id`s** (`sample_rate` in config). Deterministic hashing (not RNG) means the sample is reproducible and a loan is either wholly in or wholly out — which simultaneously satisfies the **loan-level containment** requirement of `dataset_usage.md` §8.3.
3. **Sample representativeness validation** — the sample's distributions are compared against the full-population aggregates and reported, so the sample is *demonstrated* representative rather than assumed.

Sampling loans (never rows) is the statistically correct unit here: it preserves each loan's complete trajectory, which is required for forward-looking labels, rolling features, and survival analysis.

### Gap 4 — Three supporting files do not exist; and the split window differs

**Supporting files.** `servicer_updates.csv`, `validation_rules.json`, and `macro_scenarios.csv` are referenced throughout but are not in the delivery. `dataset_usage.md` §13 titles them "**Supporting Synthetic Datasets**" and specifies each one's purpose and integration. **Decision:** generate all three programmatically, seeded and reproducible, into `data/supporting/`, with a header in each declaring it synthetic and generated. This is explicitly sanctioned by §13, not invention.

**Split window.** The docs' illustrative split is Train 2018–2020 / Validation 2021 / Test 2022, and notes "exact boundaries depend on the delivered data's `<DATE_RANGE>`." The master prompt says Train 2018–2020, Validation 2021, Test "latest available period."

Empirically, the ZIP names are **acquisition** quarters, but each file carries monthly performance forward to the data vintage. Verified `ACT_PERIOD` range: **2018-01 → 2025-12** (2018Q1 file alone spans 99 distinct months). So the panel is ~96 months, far more than the 2018–2021 the filenames suggest.

**Decision.** Honour the specified primary split — **Train ≤ 2020-12, Validation 2021, Test = latest fully-labelled period** — and additionally report a contiguous 2022 holdout for continuity. The 12-month forward label horizon means the last fully-labelled observation month is 12 months before the data end, which I compute from the data rather than hard-code. The train→test regime gap (COVID, then the 2022+ rate shock) is a genuine stress test of generalization and is reported as such, per ADR-2's noted trade-off.

---

## 4. Implementation Roadmap

Mapping the master prompt's 17 phases onto `implementation.md`'s 8 engineering phases and the challenge's required tasks.

| Phase | Deliverable | Key outputs |
|---|---|---|
| **0** | Documentation analysis | `docs/documentation_analysis.md` *(this file)* |
| **1** | Project structure | `data/{raw,extracted,processed,features,splits,supporting}`, `backend/`, `frontend/`, `ml/`, `database/`, `pipelines/`, `reports/`, `docs/`, `tests/`, `docker/` |
| **2** | Ingestion pipeline | `ml/data_pipeline/{extract,loader,schema_detection,validation,preprocessing,feature_engineering,dataset_builder}.py` — streaming, chunked, memory-safe |
| **3** | Dataset discovery | `reports/data_dictionary.md` — all 113 raw fields, classified, with band cuts and ML usage |
| **4** | Data quality & profiling | `reports/data_intelligence_report.md` — missingness, 5 validation families, distributions, correlations, drift |
| **5** | Data processing | `loan_static_attributes.parquet`, `loan_monthly_performance.parquet`, `engineered_features.parquet`, `model_dataset.parquet`, `population_monthly_aggregates.parquet` |
| **6** | ML development | Baselines (LogReg, RandomForest) + advanced (XGBoost, LightGBM) for default / delinquency-3m / delinquency-6m / prepayment-12m / next-state |
| **7** | Time-aware validation | `reports/model_evaluation.md` — ROC-AUC, PR-AUC, F1, recall@precision, Brier, calibration; leakage-prevention argument |
| **8** | Survival / transition | `reports/survival_analysis.md` — hazard curves, survival probability, empirical + modeled transition matrix |
| **9** | Anomaly detection | `reports/anomaly_report.md` — rules + Isolation Forest, ≥20 worked examples with drivers and recommendations |
| **10** | Scenario simulation | `reports/scenario_report.md` — base / adverse-credit / high-prepayment × segments |
| **11** | Explainable AI | `reports/explainability_report.md` — global + local SHAP, FP/FN analysis, uncertainty |
| **12** | LLM reviewer copilot | `ml/llm/` RAG over dictionary + rules + model output; `logs/llm/`; `docs/llm_failure_cases.md` |
| **13** | Backend | FastAPI implementing all 7 `api_spec.md` endpoints, layered per `backend.md` |
| **14** | Frontend | 7 pages per `frontend.md` |
| **15** | Orchestration | `pipelines/` DAG implementing the 7 orchestrator stages, idempotent + resumable |
| **16** | Final deliverables | `submission/` — README, model_card, 3 reports, AI_development_log, demo_flow, submission.csv |
| **17** | Testing & review | `tests/`, plus the 12-point verification checklist |

---

## 5. Dependencies Between Modules

### 5.1 Build-order DAG

```
Phase 0 docs analysis
   └─> Phase 1 structure
        └─> Phase 2 ingestion ── Phase 3 discovery (data_dictionary)
             └─> Phase 4 profiling + validation rules
                  └─> Phase 5 processed parquet layers
                       ├─> Phase 6 model training ──> Phase 7 evaluation
                       │        ├─> Phase 8 survival / transition
                       │        ├─> Phase 11 explainability (needs trained models)
                       │        └─> Phase 10 scenarios (needs models + macro_scenarios)
                       └─> Phase 9 anomaly detection (needs features + validation flags)
                            └─> Phase 12 LLM copilot (needs predictions + SHAP + anomalies + dictionary)
                                 └─> Phase 13 backend (serves everything above)
                                      └─> Phase 14 frontend (consumes backend)
                                           └─> Phase 15 orchestration (sequences 2→12)
                                                └─> Phase 16 deliverables ──> Phase 17 review
```

### 5.2 Hard preconditions

| Module | Cannot start until | Why |
|---|---|---|
| Feature engineering | Monthly panel is sorted per `(loan_id, reporting_month)` | Rolling/lag features need ordered history; unordered input silently produces leakage |
| Target construction | Full per-loan future window is available | Forward labels scan up to 12 months ahead |
| Model training | Time split + loan containment applied | ADR-2 disqualification risk |
| Explainability | Trained model + its exact training feature matrix | SHAP needs the model's own feature space and column order |
| Anomaly detection | Validation flags + engineered numerics | Rules supply `exception_type`; IF supplies the score |
| Scenario simulation | Trained models + `macro_scenarios` | Scenarios re-score with the *same* production model |
| LLM copilot | Predictions + explanations + anomalies + data dictionary | RAG has nothing to ground on otherwise; enforces "LLM only after step 5" |
| Backend `/predict` | Serialized model + feature row available | Otherwise must return `503 model_not_ready` / `409 features_not_available` |
| Frontend | Backend OpenAPI contract | Typed client is generated from it |

### 5.3 Deliberate decoupling

Per `orchestrator.md` §7, anomaly detection and the supervised training path are **independent branches** off the shared validated feature set — a failed anomaly run must not block delinquency/default training. I keep them as separate pipeline stages with no cross-imports.

---

## 6. Technology Decisions

### 6.1 Adopted directly from the documentation

| Layer | Choice | Source |
|---|---|---|
| Prediction models | XGBoost + LightGBM, with LogisticRegression + RandomForest baselines | ADR-3; `implementation.md` Phase 4 |
| Validation strategy | Time-aware split, loan-level containment | ADR-2; `dataset_usage.md` §8 |
| Explainability | SHAP (TreeSHAP) | ADR-6 |
| Survival | `lifelines` / discrete-time transition | `implementation.md` Phase 5 |
| Anomaly | IsolationForest + deterministic rules | `dataset_usage.md` §11.2 |
| Backend | FastAPI + Pydantic v2 | ADR-5 |
| Database | PostgreSQL schema, 9 tables | ADR-4; `database_schema.md` |
| Frontend | React 18 + TypeScript + Tailwind + React Query + Zustand | `frontend.md`; `implementation.md` §1 |
| LLM | RAG, provider-agnostic, recommendation-only | ADR-7; `prd.md` §4.6 |
| Storage | Parquet + snappy via PyArrow | `dataset_usage.md` §14, §17.9 |

### 6.2 Deviations forced by the environment — each justified

| # | Documented | Implemented | Justification |
|---|---|---|---|
| 1 | Extract ZIPs to `data/extracted/` | Stream directly from ZIP; never extract | 200 GB uncompressed vs 56 GB free (Gap 2). Strengthens raw immutability. |
| 2 | PostgreSQL as system of record | **SQLite** with the *same* 9-table schema, DDL, constraints and indices; PostgreSQL DDL retained in `database/schema.sql` | No PostgreSQL server in this environment. Schema, constraints and repository interfaces are identical, so the swap is a connection-string change. JSONB → `TEXT` holding JSON; range partitioning → index on `reporting_month`. Documented in the model card. |
| 3 | MLflow Model Registry | Local versioned artifact registry with the same `staging → production → archived` lifecycle and a JSON manifest linking model version ↔ `feature_set_version` ↔ data snapshot | No MLflow server. Reproducibility and traceability requirements (`orchestrator.md` §5) are met by the manifest; the `ModelRegistry` interface in `backend.md` §2.6 is preserved so MLflow can be dropped in behind it. |
| 4 | Airflow DAG | Python orchestrator with the same 7 stages, explicit dependencies, idempotent upserts, resumable per-stage execution, run-level audit records | `orchestrator.md` §4 accepts Prefect as an alternative and states "the DAG shape does not change." I preserve the DAG shape and semantics without requiring a scheduler daemon. |
| 5 | Redis cache + Celery | In-process TTL cache; synchronous batch endpoints with a job-status table | No Redis. Caching is an optimization (`backend.md` §3.9), not a contract; the job-status polling contract from `frontend.md` §4 is preserved. |
| 6 | Live Anthropic API for the copilot | Provider-agnostic `LLMClient` with a real `anthropic` adapter **and** a deterministic grounded-template fallback used when no API key is present | Guarantees the system is demonstrable and reproducible offline. Both paths write identical `llm_logs` records and identical `grounding_sources`; the fallback is labeled as such in the log, so no output is ever passed off as model-generated when it wasn't. |

Every deviation preserves the documented **interface** so the documented technology can be substituted without touching call sites.

### 6.3 Constraints I will not trade away

These are the disqualification-adjacent requirements. They are non-negotiable regardless of time pressure:

1. **No LLM-generated predictions.** All probabilities come from serialized ML artifacts.
2. **No random train/validation split.** Time-aware, with loan-level containment, and forward-label horizons truncated at the boundary.
3. **Raw data immutable.** Streamed read-only, checksummed, `chmod 444`.
4. **No silent drops.** Invalid records are quarantined with a reason code and routed to the anomaly layer.
5. **Every LLM output labeled a recommendation** and logged with its grounding sources.
6. **Reproducibility.** Fixed seeds, versioned configs, deterministic sampling, no wall-clock dependence in the pipeline.

---

## 7. Assumptions Register

Carried forward and re-stated in `reports/data_dictionary.md` and the model card.

| # | Assumption | Basis | Risk if wrong |
|---|---|---|---|
| A1 | The 113 positional fields follow the published Fannie Mae SF Loan Performance layout | Empirically verified by value-pattern profiling of every position | Mis-mapped features; mitigated by profiling + unit tests on derived fields |
| A2 | `DLQ_STATUS` is months-delinquent as a 2-char code (`00` current, `01`=30d, … `XX` unknown) | Observed value set `{00,01,…,18,XX}` with the expected monotone frequency decay | Wrong delinquency labels — the highest-impact risk; unit-tested |
| A3 | `Zero_Bal_Code` `01` = prepaid/matured; `02,03,09,15` = credit-event termination; `06` = repurchase | Fannie Mae published code list; observed `01` dominant | Confused prepayment vs default targets; validated against balance/date behaviour |
| A4 | Default = 180+ days delinquent **or** a credit-event zero-balance code | Industry-standard D180 definition; `prd.md` says "a defined default state" without fixing it | Alternative D90 definition shifts base rate; both reported, D180 primary |
| A5 | Band cut-points for credit score / LTV / DTI | Chosen to match the API examples in `api_spec.md` (`"680-719"`, `"80-90"`, `"36-43"`) | Cosmetic; continuous values retained alongside bands |
| A6 | A deterministic loan-level sample is a valid modeling population | `prd.md` NFR target scale 250K–1M+ loan-months; representativeness validated against 100 % population aggregates | Sampling bias; explicitly measured and reported |
| A7 | The three supporting files are to be synthesized | `dataset_usage.md` §13 titles them "Supporting Synthetic Datasets" | None — sanctioned by the document; each file declares itself synthetic |
| A8 | `month_index` is a derived per-loan sequential counter | `dataset_usage.md` §4.1 defines it as such; absent from raw data | None — derived deterministically from sorted `reporting_month` |

---

## 8. Phase 0 Exit Criteria

| Criterion | Status |
|---|---|
| All 9 source documents read completely | Met |
| Product, architecture, backend, frontend, database, API, ML-workflow and decision requirements extracted | Met |
| Final architecture understanding documented | Met — §2 |
| Documentation-vs-data gaps identified and resolved with documented decisions | Met — §3, four gaps |
| Implementation roadmap defined | Met — §4 |
| Inter-module dependencies mapped | Met — §5 |
| Technology decisions recorded, including justified deviations | Met — §6 |
| Assumptions registered rather than silently invented | Met — §7 |

**Phase 0 complete.** Proceeding to Phase 1 (project structure) and Phase 2 (ingestion pipeline).

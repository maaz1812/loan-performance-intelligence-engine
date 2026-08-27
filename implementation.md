# Loan Performance Intelligence Engine — Implementation Guide

**Audience:** Engineering, Data Science, and MLOps teams
**Scope:** End-to-end build guide covering architecture, folder structure, phased development plan, and deployment
**Related doc:** `frontend.md`

---

## 1. Technology Stack

### Frontend
| Layer | Choice | Why |
|---|---|---|
| UI Library | React 18 (Vite) | Fast dev server, component model fits dashboard + form-heavy UI |
| Language | TypeScript | Type safety across API contracts (predictions, SHAP payloads, scenario outputs) |
| Styling | Tailwind CSS | Rapid, consistent styling for data-dense dashboards |
| Charts | Recharts / Plotly.js | Risk curves, SHAP bar charts, scenario projections |
| State | React Query + Zustand | Server-cache (React Query) + light client state (Zustand) |

### Backend
| Layer | Choice | Why |
|---|---|---|
| API Framework | FastAPI (Python 3.11+) | Async, automatic OpenAPI schema, Pydantic validation — matches ML team's Python stack |
| Server | Uvicorn + Gunicorn workers | Production ASGI serving |
| Task Queue | Celery + Redis (optional) | Long-running batch scoring, scenario simulation jobs |
| Auth | JWT (OAuth2 password/bearer flow) | Reviewer login, role-based access (analyst vs. admin) |

### Database
| Layer | Choice | Why |
|---|---|---|
| Primary store | PostgreSQL 15+ | Loan-level relational data, monthly performance panel, servicer updates |
| Extensions | TimescaleDB (optional) | Efficient time-series queries over `loan_monthly_performance` |
| Object storage | S3 / MinIO | Trained model artifacts, SHAP explainer objects, reports, data-quality exports |
| Cache | Redis | Cached predictions, session tokens, scenario job status |

### Machine Learning
| Layer | Choice | Why |
|---|---|---|
| Core ML | scikit-learn | Preprocessing pipelines, baseline models, metrics, calibration |
| Gradient boosting | XGBoost, LightGBM | Delinquency / default / prepayment / next-state classifiers |
| Survival modeling | `lifelines`, `scikit-survival` | Time-to-event / hazard / transition modeling |
| Explainability | SHAP | Global + local explanations, feature importance, driver analysis |
| Anomaly detection | `IsolationForest`, `PyOD` (ECOD, AutoEncoder) | Record-level anomaly scores |
| Experiment tracking | MLflow | Model registry, metric comparison across phases |

### LLM Layer
| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain | Prompt templates, retrieval chains, output parsing |
| Retrieval (RAG) | FAISS / pgvector | Ground answers in `data_dictionary.md` and `validation_rules.json` |
| LLM API | Provider-agnostic (Claude API via `anthropic` SDK, or equivalent) | Reviewer note generation, scenario summaries, natural-language Q&A |
| Guardrails | Structured output parsing + rule-based validators | Prevent LLM output from silently becoming a decision |

### Deployment
| Layer | Choice | Why |
|---|---|---|
| Containerization | Docker + Docker Compose | Reproducible local + CI environments |
| Orchestration | Kubernetes (optional at scale) or single-VM Compose | Depends on judging/demo scale vs. production scale |
| Cloud | AWS / Azure / GCP (any) | Managed Postgres, object storage, container hosting |
| CI/CD | GitHub Actions | Lint, test, build image, push, deploy |
| Monitoring | Prometheus + Grafana | API latency, model-serving health, drift alerts |

---

## 2. Project Structure

```
project/
├── frontend/                 # React + TypeScript SPA (see frontend.md)
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/                  # FastAPI application layer
│   ├── app/
│   │   ├── api/              # Route handlers (routers per domain: loans, predictions, anomalies, scenarios, reviewer)
│   │   ├── core/             # Config, security, settings
│   │   ├── schemas/          # Pydantic request/response models
│   │   ├── services/         # Business logic bridging API <-> ML/DB layers
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
│
├── ml/                        # All data science & modeling code
│   ├── ingestion/             # Load & validate raw CSVs into normalized structures
│   ├── profiling/             # Data intelligence: distributions, missingness, drift
│   ├── features/              # Feature engineering pipelines
│   ├── training/               # Model training scripts (classification, survival, anomaly)
│   ├── explainability/         # SHAP global/local explanation generation
│   ├── scenarios/              # Scenario/stress simulation logic
│   ├── evaluation/             # Metrics, calibration, error analysis
│   └── llm/                    # LangChain chains, RAG index builders, prompt templates
│
├── database/
│   ├── migrations/             # Alembic migration scripts
│   ├── schema.sql               # DDL for core tables (loans, monthly_performance, servicer_updates, predictions)
│   └── seed/                    # Seed / sample data for local dev
│
├── models/                     # Serialized model artifacts (gitignored; pulled from object storage in prod)
│   ├── classification/          # XGBoost/LightGBM model files + calibration objects
│   ├── survival/                # Hazard/transition model artifacts
│   ├── anomaly/                 # IsolationForest/PyOD artifacts
│   └── explainers/              # Saved SHAP explainer objects
│
├── reports/                     # Generated deliverables
│   ├── data_intelligence_report.md
│   ├── explainability_report.md
│   ├── scenario_report.md
│   ├── model_card.md
│   └── ai_development_log.md
│
├── docs/                         # This documentation
│   ├── implementation.md
│   └── frontend.md
│
├── docker-compose.yml
└── README.md
```

### Folder Explanations

- **`frontend/`** — The React/TypeScript single-page app reviewers and analysts use to explore loans, predictions, anomalies, scenarios, and LLM-generated notes. Fully detailed in `frontend.md`.
- **`backend/`** — FastAPI service exposing REST endpoints that the frontend consumes. Thin API layer; delegates modeling logic to `ml/` via `services/`.
- **`ml/`** — The core data-science codebase. Organized to mirror the 8 required tasks from the problem statement (profiling → features → training → anomaly → explainability → scenarios → LLM copilot). Runs both as notebooks (for reproducibility/demo) and as importable Python modules (for the API to call at inference time).
- **`database/`** — Schema definitions and migrations for PostgreSQL. Keeps loan static attributes, monthly performance panel, servicer updates, and prediction outputs in normalized relational tables.
- **`models/`** — Persisted, versioned model binaries and explainer objects. Not committed to git in production; stored in S3/MinIO and pulled at deploy/startup time. Local copies used during development.
- **`reports/`** — Auto-generated Markdown/HTML/PDF deliverables required by the judging criteria (data intelligence, explainability, scenario, model card, AI development log).
- **`docs/`** — Engineering documentation, including this file and the frontend architecture doc.

---

## 3. Development Phases

Each phase below maps directly to the challenge's **Required Tasks 1–7** plus deployment, expressed as an engineering build sequence.

### Phase 1 — Dataset Ingestion

**Goal:** Reliably load the organizer-provided CSVs into a normalized, queryable form and establish a reproducible ingestion pipeline.

**Implementation:**
- Build `ml/ingestion/loaders.py` to read `loan_monthly_performance_train.csv`, `loan_monthly_performance_test.csv`, `loan_static_attributes.csv`, `servicer_updates.csv`, `macro_scenarios.csv`.
- Validate schema against `data_dictionary.md` (column names, types, allowed value sets for banded fields like `credit_score_band`, `ltv_band`).
- Apply `validation_rules.json` as a first-pass deterministic rule engine (balance consistency, date validity, delinquency consistency, closed/prepaid logic, document gaps).
- Load validated records into PostgreSQL (`loans`, `loan_monthly_performance`, `servicer_updates` tables) via SQLAlchemy/Alembic.
- Log every rejected/flagged record with a reason code for later reconciliation.

**Input:** Raw CSV files, `data_dictionary.md`, `validation_rules.json`
**Output:** Normalized PostgreSQL tables + an ingestion log (`reports/ingestion_log.csv`) listing rule violations
**Libraries:** `pandas`, `pydantic`, `SQLAlchemy`, `alembic`, `great_expectations` (optional, for schema contracts)

---

### Phase 2 — Data Profiling

**Goal:** Produce the "data intelligence" layer required before any modeling — distributions, missingness, outliers, relationships, drift.

**Implementation:**
- `ml/profiling/distributions.py`: per-column summary stats, histograms for numeric fields, category frequency for banded/categorical fields.
- `ml/profiling/missingness.py`: missing-value heatmaps, missing-pattern clustering (e.g., `missingno`), flags for fields missing beyond a threshold.
- `ml/profiling/outliers.py`: IQR/robust z-score outlier flags on numeric fields (balance, rate), date-relationship checks (e.g., `reporting_month` before `origination_month`).
- `ml/profiling/relationships.py`: correlation matrix (numeric), Cramér's V (categorical), association-rule mining on exception co-occurrence.
- `ml/profiling/drift.py`: population stability index (PSI) or KS-statistic comparing train vs. test distributions per feature.
- Aggregate into record-level and batch-level **data-quality scores** (0–100), stored back to `loan_monthly_performance.dq_score`.
- Auto-generate `reports/data_intelligence_report.md`.

**Input:** Normalized DB tables from Phase 1
**Output:** Profiling report, per-record DQ scores, drift summary table
**Libraries:** `pandas`, `numpy`, `scipy`, `missingno`, `ydata-profiling` (optional), `mlxtend` (association rules)

---

### Phase 3 — Feature Engineering

**Goal:** Convert raw panel data into a leakage-safe, model-ready feature matrix for classification, survival, and anomaly tasks.

**Implementation:**
- `ml/features/temporal.py`: loan age buckets, rolling delinquency counts (3/6/12-month windows), payment-status transition history, time-since-last-modification.
- `ml/features/static.py`: one-hot/target/ordinal encoding for `credit_score_band`, `ltv_band`, `dti_band`, `state`, `loan_purpose`, `property_type`.
- `ml/features/reconciliation.py`: cross-source features from `servicer_updates.csv` — conflict flags, staleness indicators, "last-updated-by-source" recency.
- Strict **as-of-date construction**: every feature for month *t* uses only information available at or before month *t* to prevent target leakage.
- Persist engineered features to a `feature_store` table/parquet layer, versioned by `feature_set_version`.

**Input:** Profiled data from Phase 2
**Output:** Versioned feature matrices (train/test) partitioned by `reporting_month`
**Libraries:** `pandas`, `scikit-learn` (`ColumnTransformer`, `Pipeline`), `category_encoders`, `pyarrow`

---

### Phase 4 — ML Model Training

**Goal:** Train non-LLM supervised models for the four target outcomes, using a time-aware split.

**Implementation:**
- `ml/training/split.py`: time-aware split — train on months `t0..t_k`, validate on `t_k+1..t_k+3`, test on held-out latest months (never a random row-level split, to avoid same-loan leakage across folds).
- `ml/training/baseline.py`: logistic regression / simple gradient boosting baselines for each target (`next_3m_delinquency_flag`, `next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`).
- `ml/training/advanced.py`: XGBoost/LightGBM models with class-imbalance handling (`scale_pos_weight`, focal loss, or SMOTE-NC for mixed types) and hyperparameter search (Optuna).
- `ml/training/calibration.py`: probability calibration via Platt scaling or isotonic regression; report Brier score pre/post calibration.
- `ml/evaluation/metrics.py`: ROC-AUC, PR-AUC, F1, recall-at-fixed-precision, Brier score, macro-F1 — logged to MLflow per model/version.
- Save trained models + preprocessing pipeline to `models/classification/`.

**Input:** Feature matrices from Phase 3
**Output:** Trained + calibrated models, metrics report, MLflow experiment run
**Libraries:** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `optuna`, `mlflow`

---

### Phase 5 — Anomaly Detection (+ Survival/Transition Modeling)

**Goal:** Detect unreliable/suspicious records and model time-to-event behavior for delinquency/default/prepayment.

**Implementation:**
- `ml/training/survival.py`: Cox proportional hazards or discrete-time transition model (`lifelines`/`scikit-survival`) for time-to-default/prepayment; produce cumulative incidence curves per segment.
- `ml/training/anomaly.py`: IsolationForest / PyOD ECOD for record-level anomaly scores; a secondary classifier trained on labeled `exception_required`/`exception_type` where available, combining rule-based flags (Phase 1) with ML anomaly scores.
- Produce at least 20 reviewer-ready anomaly examples with driver explanations (feeds Phase 6).
- Compare survival model against a naive baseline (e.g., static hazard rate).

**Input:** Feature matrices, ingestion validation flags
**Output:** Anomaly scores, exception-type predictions, survival curves, transition matrices
**Libraries:** `lifelines`, `scikit-survival`, `pyod`, `scikit-learn`

---

### Phase 6 — Explainability

**Goal:** Make every model output interpretable to a human reviewer.

**Implementation:**
- `ml/explainability/global_importance.py`: SHAP summary plots (feature importance) per model.
- `ml/explainability/local_explanations.py`: SHAP force/waterfall values for individual loans, cached for on-demand API retrieval.
- `ml/explainability/error_analysis.py`: false-positive/false-negative slice analysis by segment (state, credit band, vintage).
- `ml/explainability/uncertainty.py`: prediction confidence bands (e.g., quantile or ensemble-variance based).
- Auto-generate `reports/explainability_report.md` and a `model_card.md` (objective, data, features, validation method, metrics, limitations, leakage controls, failure modes).

**Input:** Trained models (Phase 4/5), feature matrices
**Output:** SHAP artifacts (`models/explainers/`), explainability report, model card
**Libraries:** `shap`, `pandas`, `matplotlib`/`plotly`

---

### Phase 7 — LLM Integration

**Goal:** Add a governed LLM copilot layer that explains, summarizes, and retrieves — without making decisions.

**Implementation:**
- `ml/llm/rag_index.py`: build a vector index (FAISS/pgvector) over `data_dictionary.md` and `validation_rules.json` for grounded retrieval.
- `ml/llm/chains.py`: LangChain chains for (a) reviewer note generation from a loan's predictions/anomaly score/SHAP drivers, (b) natural-language scenario summaries, (c) data-dictionary Q&A.
- Every LLM call is grounded — prompts include retrieved context and structured model outputs, never asked to "predict" risk on its own.
- `ml/llm/logging.py`: persist prompt, model name, timestamp, and output to a `llm_audit_log` table.
- Explicitly label all LLM output as **"recommendation, not decision"** in both the API response schema and the UI.
- Maintain a curated set of examples where LLM output was wrong/vague/overconfident, with the corrected version, for the required deliverable.

**Input:** Model predictions, SHAP outputs, `data_dictionary.md`, `validation_rules.json`
**Output:** Reviewer notes, scenario summaries, prompt/audit log, rejected-output examples
**Libraries:** `langchain`, `faiss-cpu` or `pgvector`, `anthropic` (or chosen LLM SDK)

---

### Phase 8 — Deployment

**Goal:** Ship the system as a reproducible, monitorable, containerized application.

**Implementation:**
- Containerize `frontend`, `backend`, and a `ml-worker` (batch scoring/training jobs) as separate services in `docker-compose.yml`.
- Backend loads model artifacts from object storage at startup (or on a scheduled refresh job).
- Expose `/predict`, `/anomalies`, `/scenarios`, `/explain`, `/reviewer/notes` endpoints via FastAPI, documented via auto-generated OpenAPI schema.
- CI/CD pipeline (GitHub Actions): lint → unit tests → build images → push to registry → deploy.
- Monitoring: Prometheus scrapes API + model-serving metrics; Grafana dashboards for latency, error rate, prediction-drift alerts (PSI recompute on incoming batches).

**Input:** All artifacts from Phases 1–7
**Output:** Running, monitored, containerized application; `submission.csv` generation endpoint/job
**Libraries/Tools:** Docker, Docker Compose, GitHub Actions, Prometheus, Grafana, MLflow Model Registry

---

## 4. Deployment

### Dockerization
- Separate `Dockerfile` per service: `frontend/Dockerfile` (multi-stage: build with Vite, serve with Nginx), `backend/Dockerfile` (Python slim base, Gunicorn+Uvicorn workers), `ml/Dockerfile` (heavier image with ML libraries, used for training/batch jobs, not the live API).
- `docker-compose.yml` wires together: `frontend`, `backend`, `postgres`, `redis`, `ml-worker`, and optionally `mlflow`.
- Use multi-stage builds to keep the backend image lean (exclude training-only dependencies like `optuna`/`ydata-profiling` from the serving image).

### CI/CD
- **On PR:** lint (`ruff`, `eslint`), type-check (`mypy`, `tsc`), unit tests (`pytest`, `vitest`).
- **On merge to main:** build Docker images, tag with commit SHA, push to container registry (ECR/GCR/ACR).
- **On release tag:** deploy to staging → run smoke tests against `/health` and a sample `/predict` call → promote to production.
- Model artifacts are versioned separately from application code (MLflow Model Registry or S3 versioned buckets) so a model rollback doesn't require a code redeploy.

### Cloud Architecture
- **Compute:** Backend + ML-worker as containers on ECS/Cloud Run/AKS (or a single VM with Compose for hackathon/demo scale).
- **Database:** Managed PostgreSQL (RDS/Cloud SQL/Azure Database for PostgreSQL).
- **Storage:** S3/MinIO for model artifacts, reports, and generated `submission.csv`.
- **Networking:** Frontend served via CDN (CloudFront/Cloud CDN) + Nginx; backend behind an API gateway/load balancer with HTTPS termination.
- **Secrets:** LLM API keys and DB credentials in a secrets manager (AWS Secrets Manager / Vault), injected as environment variables at deploy time — never committed to the repo.

### Monitoring
- **API health:** request latency, error rate, throughput (Prometheus + FastAPI middleware/`prometheus-fastapi-instrumentator`).
- **Model health:** prediction-score distribution drift (PSI) computed on a rolling window of live predictions, compared against the training distribution; alert if drift exceeds threshold.
- **Business KPIs:** anomaly rate over time, exception volume by servicer/state, reviewer override rate on LLM notes.
- Dashboards in Grafana; alerting via Slack/PagerDuty webhook on threshold breach.

### Logging
- Structured JSON logging (`structlog`) across backend and ml-worker, correlated by request ID.
- All LLM calls logged to `llm_audit_log` (prompt, model, timestamp, output, linked loan_id) — required for the "governed AI copilot" judging criterion.
- All ingestion rule violations logged with reason codes for traceability (Phase 1).

### Model Updates
- Retraining pipeline runs as a scheduled `ml-worker` job (or manually triggered) — re-runs Phases 2–6 on refreshed data.
- New model versions registered in MLflow; promoted to "production" stage only after passing a validation gate (metrics ≥ current production model on held-out set).
- Backend picks up new production model version via a config flag / registry lookup — no code change required for a model swap.
- Maintain at least one prior model version for instant rollback.

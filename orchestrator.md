# ML Workflow Orchestrator — Loan Performance Intelligence Engine

## 1. Purpose

The orchestrator governs the offline ML lifecycle — everything that happens **before** a model is available for the FastAPI backend to serve via `/predict`, `/anomalies`, `/simulate`, and `/explanation`. It is deliberately kept separate from the request-serving backend: training and batch scoring are resource-intensive, scheduled, and stateful workflows, while the API layer must stay fast and stateless.

---

## 2. Pipeline Architecture

```
┌────────────────┐
│ Data Ingestion  │  Load loan_monthly_performance, loan_static_attributes,
│                 │  servicer_updates from source files / upstream systems
└───────┬─────────┘
        ▼
┌────────────────┐
│  Validation     │  Run validation_rules.json checks: balance consistency,
│                 │  date validity, delinquency logic, doc-status gaps
└───────┬─────────┘
        ▼
┌────────────────┐
│ Feature Pipeline│  Profiling, imputation, encoding, time-aware feature
│                 │  windows; writes to `features` table (versioned)
└───────┬─────────┘
        ▼
┌────────────────┐
│    Training     │  Time-aware split; train delinquency/default/prepayment/
│                 │  next-state models + survival/hazard model + anomaly model
└───────┬─────────┘
        ▼
┌────────────────┐
│   Evaluation    │  ROC-AUC, PR-AUC, F1, recall@precision, Brier score,
│                 │  calibration curves, baseline comparison, drift checks
└───────┬─────────┘
        ▼
┌────────────────┐
│   Deployment    │  Register model in MLflow Model Registry; promote to
│                 │  "production" stage; backend ModelRegistry cache refresh
└───────┬─────────┘
        ▼
┌────────────────┐
│   Monitoring    │  Drift monitoring, prediction distribution checks,
│                 │  data-quality score tracking, alerting
└────────────────┘
```

Each stage is an independent, idempotent task with clearly defined inputs/outputs (files or DB tables), so any stage can be re-run in isolation for debugging or backfill without re-running the whole DAG.

---

## 3. Stage Detail

### 3.1 Data Ingestion
- Reads `loan_monthly_performance_*.csv`, `loan_static_attributes.csv`, `servicer_updates.csv` from a landing zone (S3/blob storage or SFTP drop).
- Performs schema-on-read validation (column presence, types) before loading into staging tables.
- Reconciles `servicer_updates.csv` against primary records — flags conflicts for the anomaly pipeline rather than silently overwriting.

### 3.2 Validation
- Executes deterministic checks defined in `validation_rules.json` (balance consistency, date-order validity, delinquency-status consistency, closed/prepaid status logic, missing-document detection).
- Produces a **record-level data-quality score** and a **batch-level summary report**, both persisted for the data-intelligence report deliverable.
- Records failing hard validation (e.g., negative balance) are quarantined, not silently dropped, and routed to the anomaly detector as candidate exceptions.

### 3.3 Feature Pipeline
- Computes engineered features: rolling delinquency counts, payment-trend deltas, credit/LTV/DTI band encodings, time-since-origination features, servicer-level aggregates.
- Every feature set is **versioned** (`feature_set_version`) and written to the `features` table so that a given model version can always be traced back to the exact feature definition used to train it — critical for reproducibility and for the model card's "leakage controls" section.
- Train/test drift is computed here (population stability index or KS-test per feature) and fed into the data-intelligence report.

### 3.4 Training
- Uses a **time-aware split**: training data strictly precedes validation/test data in `reporting_month`, and the same `loan_id` is never split across train and validation to avoid leakage — directly addressing the "Low-Score/Disqualification" rule on random splits leaking loans.
- Trains:
  - Gradient-boosted classifiers for `next_3m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`.
  - A survival/hazard or discrete-time transition model for time-to-event analysis.
  - An anomaly detector (isolation forest / autoencoder) trained on validated "clean" records, scored against the full population.
- Baseline models (e.g., logistic regression) are trained alongside improved models (gradient boosting) for the required baseline-vs-improved comparison.

### 3.5 Evaluation
- Computes ROC-AUC, PR-AUC, F1, recall-at-fixed-precision, Brier score, and macro-F1 for classification targets.
- Computes calibration curves and applies post-hoc calibration (Platt/isotonic) where needed.
- Compares each candidate model against its baseline and against the currently deployed production model — a model is only promoted if it meets a configured minimum uplift threshold.
- Generates false-positive/false-negative analysis and confidence/uncertainty summaries feeding the explainability report.

### 3.6 Deployment
- On passing evaluation gates, the model artifact (plus its SHAP explainer and calibration transform) is logged to **MLflow Model Registry** and promoted from `staging` to `production`.
- The FastAPI backend's `ModelRegistry` picks up the new production model either via a registry webhook or a periodic poll, without requiring an API redeploy.
- A **model card** is auto-generated at this stage (objective, data, features, model type, validation method, metrics, limitations, leakage controls, known failure modes) and stored alongside the artifact.

### 3.7 Monitoring
- Post-deployment, scheduled jobs track prediction-distribution drift against training-time distributions, data-quality score trends, and anomaly-rate trends.
- Alerts fire (Slack/email/webhook) when drift exceeds threshold or when the batch-level data-quality score drops below a configured floor — triggering a re-training review rather than silent degradation.

---

## 4. Pipeline Scheduling

- **Orchestration tool:** Airflow (primary recommendation for this workload) is used to define the pipeline above as a DAG, with each stage as a task and explicit upstream/downstream dependencies matching the diagram in Section 2.
- **Alternative:** Prefect is a viable lighter-weight alternative for teams preferring Python-native flow definitions over Airflow's operator model; both are supported patterns — the DAG shape does not change.
- **Schedule:**
  - Full ingestion → feature → scoring pipeline runs **monthly**, aligned to the loan-performance reporting cycle.
  - Validation and anomaly detection can additionally run **daily** on incremental servicer updates, independent of the monthly full retrain.
  - Retraining is **triggered**, not purely calendar-based: a scheduled monthly retrain is the default, but the monitoring stage can also trigger an out-of-cycle retrain job when drift/data-quality alerts fire.
- **Backfill support:** because each stage reads/writes versioned, dated tables, historical months can be re-run independently (e.g., to regenerate features after a bugfix) without disturbing the current production model.

---

## 5. Experiment Tracking & Model Versioning

- **MLflow** is used for:
  - Experiment tracking: every training run logs parameters, metrics, and artifacts (model binary, SHAP explainer, calibration curve, evaluation report).
  - Model Registry: versioned models per target (`delinquency_model`, `default_model`, `prepayment_model`, `next_state_model`, `survival_model`, `anomaly_model`), each with `staging` → `production` → `archived` lifecycle stages.
  - Reproducibility: each registered model version links back to the exact `feature_set_version` and training data snapshot used, satisfying the model card's leakage-control and reproducibility requirements.
- Alternative/complementary tool: **Weights & Biases** can be used for richer experiment visualization during active development, with final production artifacts still registered in MLflow for serving consistency.

---

## 6. Agentic Workflow (AI-Assisted Development)

This project explicitly requires evidence of agentic/AI-assisted development, tracked separately from the ML pipeline itself.

### 6.1 AI Coding Assistant
- An AI coding assistant (e.g., Claude Code) is used for scaffolding pipeline code, writing repository/service boilerplate, and drafting test cases.
- All AI-generated code passes through the same PR review process as human-written code — no direct-to-main commits from agentic tools.

### 6.2 Experiment Runner
- An agentic experiment runner can be used to propose hyperparameter sweeps or feature-set variants, launch training runs via the orchestrator's API, and summarize results back to the developer for selection — but **model promotion to production always requires human sign-off**, never an autonomous agent action.

### 6.3 Prompt Logging
- Every prompt sent to an AI coding assistant or experiment-runner agent is logged (tool used, prompt text, accepted/rejected output, timestamp) in the **AI Development Log**, separate from the runtime `llm_logs` table (which governs the reviewer-facing copilot in production, not development-time tooling).
- The AI Development Log records: AI tools used, representative prompts, accepted vs. rejected outputs, human review process, approximate AI-generated code share, and lessons learned — per the challenge's required deliverable.

### 6.4 Human Approval Gates
Human approval is a hard gate at three points in the system, consistent with the "LLM copilots need governance" benchmark theme:
1. **Model promotion** (staging → production) requires a human reviewer to approve the evaluation report.
2. **LLM reviewer summaries** (`/review-summary`) are always labeled `is_recommendation: true` and require a `reviewer` role user to set `approval_status` to `approved`/`rejected`/`corrected` before being treated as final in any downstream reporting.
3. **Anomaly review** requires a human to mark `anomaly_results.reviewed = true`; no anomaly is auto-resolved by the pipeline.

---

## 7. Failure Handling & Idempotency

- Every orchestrator task is idempotent: re-running a task for the same `reporting_month`/`feature_set_version` overwrites (upserts) rather than duplicates rows, safe for retries after transient failures.
- Failed tasks halt only their downstream dependents in the DAG — e.g., a failed anomaly-detection run does not block the delinquency/default training path, since they read from the same validated feature set but are otherwise independent branches.
- All task failures emit structured alerts including the DAG run ID, task name, and stack trace, correlated with the same logging/tracing conventions used in the serving backend (see `backend.md §3.10`).

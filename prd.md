# Product Requirement Document (PRD)
## Loan Performance Intelligence Engine

---

## 1. Product Overview

### Product Name
**Loan Performance Intelligence Engine (LPIE)**

### Vision
To build an ML-first analytics platform that transforms messy, high-volume loan-level data into reliable, explainable, and forward-looking intelligence — enabling financial institutions to detect risk earlier, act with transparency, and reduce reliance on manual, spreadsheet-driven review. LPIE treats machine learning as the analytical core of the system and uses large language models strictly as a communication and workflow layer on top of that core, never as a substitute for statistically validated prediction.

### Problem Statement
Loan portfolios generate large volumes of monthly performance data across multiple source systems (origination platforms, servicers, document repositories). This data is frequently incomplete, inconsistently formatted, and prone to silent quality issues — stale updates, conflicting source records, and broken cross-field relationships. Institutions need a way to:

- Continuously assess the *reliability* of incoming loan data before it is used for decisioning.
- Predict delinquency, default, prepayment, and next-state transitions ahead of time, not after the fact.
- Detect anomalous or exception-worthy records without relying purely on brittle rule sets.
- Simulate how a portfolio behaves under adverse or shifting macro conditions.
- Give risk reviewers explanations they can trust and audit, rather than opaque model outputs or unverified LLM narratives.

### Target Users
| User | Core Need |
|---|---|
| **Loan Analysts** | Review flagged loans quickly, understand *why* a loan was flagged, action exceptions |
| **Risk Managers** | Portfolio-level visibility into risk concentration, scenario outcomes, and trend shifts |
| **Data Scientists / ML Engineers** | Build, validate, calibrate, and monitor prediction and anomaly models |
| **Financial Institutions (org-level)** | Auditable, governed, reproducible risk intelligence for compliance and reporting |

---

## 2. Business Problem

Financial institutions managing large loan portfolios face four compounding problems:

1. **Data quality is invisible until it causes damage.** Loan records arrive from multiple servicers and source systems with missing fields, stale timestamps, and conflicting updates. Without systematic profiling, bad data silently corrupts downstream risk decisions.
2. **Manual risk analysis does not scale.** Analysts reviewing loans one-by-one, or relying on static rule engines, cannot keep pace with portfolios spanning hundreds of thousands of loans updated monthly.
3. **Existing risk tooling lacks transparency.** Black-box scores without feature-level explanation make it difficult for risk managers to defend decisions to auditors, regulators, or credit committees.
4. **Predictive and scenario analytics are underused.** Most legacy workflows are backward-looking (reporting what happened) rather than forward-looking (what is likely to happen, and under what macro conditions).

LPIE addresses all four by combining rigorous data intelligence, multi-outcome predictive modeling, anomaly detection, scenario simulation, and a governed explanation layer in one system.

---

## 3. Product Objectives

The platform should:

1. **Analyze loan portfolios** — profile distributions, missingness, outliers, and drift across static and monthly performance data.
2. **Predict risk** — produce calibrated, time-aware probabilities for delinquency, default, prepayment, and next-state transitions.
3. **Detect unreliable records** — score records and batches for anomalies and data-quality exceptions, independent of and complementary to predictive scoring.
4. **Generate explanations** — provide global and loan-level explanations for every prediction and anomaly flag.
5. **Assist reviewers** — surface LLM-generated, retrieval-grounded summaries and reviewer notes that accelerate — but never replace — human judgment.

**Explicit non-objective:** LPIE does not use an LLM as the source of predictive risk scores. All probability and risk outputs must originate from trained, validated ML models.

---

## 4. Functional Requirements

### 4.1 Data Intelligence

- **Dataset upload**: Support batch upload of `loan_monthly_performance`, `loan_static_attributes`, `servicer_updates`, and `macro_scenarios` files (CSV), with versioned ingestion so a given run is always traceable to a specific data snapshot.
- **Schema validation**: Validate incoming files against an expected schema (field names, types, allowed value sets for banded fields like credit-score band, LTV band); reject or quarantine non-conforming rows with a reason code.
- **Missing value detection**: Compute per-field missingness rates at both record and batch level, and flag fields whose missingness pattern changed materially versus a prior batch.
- **Outlier detection**: Detect statistically anomalous numeric values (e.g., interest rate, balance) and logically invalid relationships (e.g., current balance greater than original balance, negative remaining term, delinquency dates preceding origination).
- **Data quality scoring**: Produce a record-level data-quality score and a batch-level aggregate score, combining missingness, outlier flags, cross-field consistency checks, and source-conflict signals from `servicer_updates`.

### 4.2 ML Prediction

The system must support multi-outcome prediction, each independently trained and evaluated:

- **Default prediction** — probability of default within a defined forward horizon (e.g., next 12 months).
- **Delinquency prediction** — probability of delinquency within short (3-month) and medium (6-month) horizons.
- **Prepayment prediction** — probability of prepayment within a forward horizon.
- **Next-state prediction** — multi-class prediction of the loan's likely status transition (e.g., current → 30-days-past-due → default/prepaid/closed).

All models must be trained with a **time-aware split** (no loan appearing in both train and validation windows within the same evaluation period) and evaluated using ROC-AUC, PR-AUC, F1, recall at fixed precision, Brier score, and macro-F1 where applicable.

### 4.3 Anomaly Detection

- **Suspicious record detection**: Unsupervised/semi-supervised scoring (e.g., isolation-based or reconstruction-based methods) layered on top of deterministic validation rules, to catch anomalies that static rules miss.
- **Exception classification**: Classify flagged records into exception types (e.g., balance inconsistency, stale update, document gap, delinquency-status mismatch).
- **Anomaly explanations**: For every flagged record, surface the top contributing fields/features driving the anomaly score, in a reviewer-readable format.

### 4.4 Scenario Simulation

- **Base scenario**: Project portfolio performance under current macro and behavioral assumptions.
- **Adverse credit scenario**: Apply stressed credit assumptions (elevated default/delinquency propensity) and project outcomes.
- **High prepayment scenario**: Apply accelerated prepayment assumptions and project outcomes.
- Each scenario must produce portfolio-level and **segment-level** projections (by vintage, credit-score band, state, servicer) along with a plain-language explanation of the top drivers of the projected shift.

### 4.5 Explainable AI

- **Feature importance**: Global feature importance per model (delinquency, default, prepayment, next-state, anomaly).
- **SHAP explanations**: Local, per-loan SHAP-based (or equivalent) explanations showing which features pushed a specific prediction up or down.
- **Loan-level explanations**: A reviewer-facing view combining the prediction, its confidence/uncertainty, and its top drivers, in language a non-data-scientist can interpret.

### 4.6 LLM Reviewer Copilot

The LLM layer operates strictly as an assistive, governed layer on top of ML outputs.

**Allowed:**
- Generating natural-language summaries of a loan's risk profile from structured model outputs.
- Drafting reviewer notes grounded in retrieved data-dictionary definitions and validation-rule context.
- Retrieving and explaining data-dictionary field definitions on request.
- Summarizing scenario outputs in plain language.

**Not allowed:**
- Making or implying a final financial/credit decision.
- Generating or substituting for a predictive probability or risk score.
- Producing ungrounded narrative claims about a loan's risk that are not traceable to model output or retrieved reference data.

Every LLM Copilot interaction must be logged (prompt, model, timestamp, retrieved context, output) and every output must be labeled as a **recommendation**, never a decision.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Security** | Role-based access control (analyst / risk manager / ML engineer / admin); encryption at rest and in transit; no raw PII/loan data exposed to the LLM layer without redaction/aggregation controls |
| **Scalability** | Batch ingestion and scoring must handle portfolios from 250K to 1M+ monthly loan-month records without redesign |
| **Reliability** | Ingestion, scoring, and explanation pipelines must be idempotent and resumable; failed batches must not silently corrupt downstream scores |
| **Auditability** | Every prediction, anomaly flag, and LLM output must be traceable to a specific model version, data snapshot, and (for LLM output) prompt/response log |
| **Reproducibility** | Given the same data snapshot and model version, all pipeline outputs (predictions, scores, explanations) must be exactly reproducible |

---

## 6. User Stories

### Loan Analyst
- As a loan analyst, I want to see a prioritized queue of flagged loans so that I can focus my review time on the highest-risk or most anomalous records first.
- As a loan analyst, I want a plain-language explanation for why a loan was flagged so that I don't have to interpret raw SHAP values myself.
- As a loan analyst, I want to accept, reject, or annotate an LLM-generated reviewer note so that my judgment is captured alongside the system's recommendation.
- As a loan analyst, I want to see the specific data-quality issues on a record (e.g., stale servicer update) so that I can decide whether the underlying prediction is trustworthy.

### Risk Manager
- As a risk manager, I want a portfolio-level dashboard showing risk concentration by vintage, credit band, and geography so that I can identify emerging concentration risk.
- As a risk manager, I want to run and compare base, adverse-credit, and high-prepayment scenarios so that I can understand portfolio resilience under stress.
- As a risk manager, I want model-level performance and calibration metrics so that I can assess whether a model is still trustworthy for decisioning.
- As a risk manager, I want an audit trail of every LLM-generated recommendation so that I can demonstrate governance to regulators.

### ML Engineer / Data Scientist
- As an ML engineer, I want a reproducible, time-aware training pipeline so that I can retrain models on new data snapshots without leakage risk.
- As an ML engineer, I want automated data drift comparisons between train and test/production data so that I can detect when a model needs retraining.
- As an ML engineer, I want calibration diagnostics per model so that predicted probabilities remain meaningful for downstream decisioning.
- As an ML engineer, I want a model card auto-populated with metrics, features, and known limitations so that model documentation stays current with minimal manual effort.

### Administrator
- As an administrator, I want to manage role-based access so that only authorized users can view sensitive loan data or approve model deployments.
- As an administrator, I want to configure and monitor scheduled ingestion and scoring jobs so that the platform stays current without manual intervention.
- As an administrator, I want visibility into system health (pipeline failures, data-quality score trends, LLM usage logs) so that I can respond to operational issues quickly.
- As an administrator, I want to manage model versioning and rollback so that a faulty model deployment can be reverted safely.

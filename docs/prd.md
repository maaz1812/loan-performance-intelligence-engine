# Product Requirements Document (PRD)
## Loan Performance Intelligence Engine

### 1. Product Overview
An ML-first system for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and grounded LLM-assisted review. 

### 2. Business Problem
Financial institutions struggle with massive, messy loan-level data. Legacy workflows are backward-looking and rely on manual review to identify risky loans. There is a need for a forward-looking predictive engine that can flag defaults, prepayments, and anomalies while explaining *why* to human reviewers.

### 3. Product Goals
* Ingest and profile up to 1M rows of loan panel data.
* Predict 3M Delinquency, 12M Default, and 12M Prepayment using time-aware ML models.
* Simulate macro-economic stress scenarios.
* Provide an LLM Copilot grounded entirely on data dictionaries to assist risk analysts.

### 4. Functional Requirements
* **Data Intelligence:** Programmatic profiling for missingness, outliers, and drift.
* **Predictive Modeling:** XGBoost/LightGBM for risk outcomes. MUST use time-aware calendar splitting.
* **Anomaly Detection:** Identify statistical outliers and flag for review.
* **Scenario Simulation:** Base, Adverse Credit, and High Prepayment transitions.
* **Explainability:** Global and local SHAP/feature importance.
* **LLM Copilot:** Natural language summarization. MUST NOT predict risk. MUST NOT hallucinate rules.

### 5. Non-Functional Requirements
* Fast API response (< 2 seconds per loan prediction).
* Memory-efficient processing (prevent OOM on large datasets).
* Fully reproducible pipelines.

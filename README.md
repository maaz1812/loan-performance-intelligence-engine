# ?? Loan Performance Intelligence Engine (LPIE)

> **Intain Campus FinTech Challenge 2026 — AI Track Submission**
>
> An ML-first platform that transforms high-volume loan-level data into reliable, explainable, and forward-looking risk intelligence.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://reactjs.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## ?? Table of Contents

1. [Project Vision](#-project-vision)
2. [Live Demo & Quick Start](#-live-demo--quick-start)
3. [System Architecture](#-system-architecture)
4. [ML Pipeline Architecture](#-ml-pipeline-architecture)
5. [Orchestrator Pipeline](#-orchestrator-pipeline)
6. [LLM Reviewer Copilot (RAG)](#-llm-reviewer-copilot-rag)
7. [Data Flow](#-data-flow)
8. [Deployment Architecture](#-deployment-architecture)
9. [Model Performance](#-model-performance)
10. [API Reference](#-api-reference)
11. [Tech Stack](#-tech-stack)
12. [Project Structure](#-project-structure)
13. [Key Architecture Decisions](#-key-architecture-decisions)
14. [Hackathon Scoring Alignment](#-hackathon-scoring-alignment)
15. [Deliverables](#-deliverables)
16. [Clone & Run Locally](#-clone--run-locally)

---

## ?? Project Vision

LPIE treats **machine learning as the analytical core** and uses large language models strictly as a **communication and workflow layer** on top of that core — never as a substitute for statistically validated prediction.

> **Explicit non-objective:** LPIE does not use an LLM as the source of predictive risk scores. All probability and risk outputs originate from trained, validated ML models.

### Business Problems Solved

| Problem | LPIE Solution |
|---|---|
| Data quality is invisible until it causes damage | Automated profiling, validation scoring, cross-field checks |
| Manual risk analysis does not scale at 250K–1M+ loans | Batch ML inference with SHAP explainability per loan |
| Black-box scores fail regulatory audits | Every prediction traceable to model version + data snapshot |
| Legacy workflows are backward-looking | Forward-looking 3M/6M/12M delinquency and default predictions |
| No stress-testing capability | Scenario simulation (base / adverse-credit / high-prepayment) |

---

## ? Live Demo & Quick Start

**GitHub Repository:** [github.com/maaz1812/loan-performance-intelligence-engine](https://github.com/maaz1812/loan-performance-intelligence-engine)

**Clone:**
```bash
git clone https://github.com/maaz1812/loan-performance-intelligence-engine.git
cd loan-performance-intelligence-engine
```

### Test Cases for the Dashboard

| Loan ID | Expected Result |
|---|---|
| `106897301046` | ?? Anomaly — 89.6% Delinquency, 66.1% Default |
| `100948053631` | ?? Anomaly — 56.6% Default, routed for Review |
| `101725753571` | ? Standard — near-zero Default, high Prepayment risk |
| `100010079393` | ? Standard — low-risk Auto-Approve profile |

---

## ??? System Architecture

LPIE follows a strict layered architecture. The **LLM Layer** is reachable only through the Orchestrator and only ever *reads* ML outputs — it has no path to write predictions, override scores, or bypass the ML Pipeline.

```mermaid
flowchart TD
    A["?? Browser / Frontend (React + TypeScript)"] --> B["? Backend API Layer (FastAPI)"]
    B --> C["?? Orchestrator"]
    C --> D["?? ML Pipeline Layer"]
    D --> E["??? PostgreSQL Database"]
    C --> F["?? LLM Layer (RAG Copilot)"]
    F --> E
    E --> B
    B --> A
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| **Frontend** | Dashboard, charts, loan-level review, scenario comparison, SHAP visualizations |
| **Backend API** | REST endpoints, request validation, role-based auth, delegates all ML/LLM to Orchestrator |
| **Orchestrator** | Sequences all pipeline stages, writes audit records, enforces execution order |
| **ML Pipeline** | Training, batch inference, feature engineering, SHAP, anomaly scoring |
| **Database** | Loans, predictions, SHAP explanations, scenario results, LLM prompt/response audit log |
| **LLM Layer** | RAG over data_dictionary.md + validation_rules.json, reviewer notes, NL summaries |

---

## ?? ML Pipeline Architecture

```mermaid
sequenceDiagram
    participant BE as Backend API
    participant OR as Orchestrator
    participant ML as ML Pipeline
    participant DB as Database
    participant LLM as LLM Layer

    BE->>OR: Trigger pipeline run (data snapshot id)
    OR->>ML: Run profiling + feature engineering
    ML->>DB: Write data-quality scores
    OR->>ML: Run prediction + anomaly + scenario models
    ML->>DB: Write predictions, anomaly scores, scenario outputs
    OR->>ML: Run explainability (SHAP, feature importance)
    ML->>DB: Write explanations
    OR->>LLM: Request grounded reviewer summary (reads DB context)
    LLM->>DB: Read predictions, explanations, data dictionary
    LLM->>DB: Write reviewer note + prompt/response log
    OR->>BE: Run complete, results available
```

### Models Trained

| Model | Algorithm | Target Variable |
|---|---|---|
| Delinquency 3M | XGBoost + LightGBM | `next_3m_delinquency_flag` |
| Delinquency 6M | XGBoost + LightGBM | `next_6m_delinquency_flag` |
| Default 12M | XGBoost + LightGBM | `next_12m_default_flag` |
| Prepayment 12M | XGBoost + LightGBM | `next_12m_prepayment_flag` |
| Next State | LightGBM Multiclass | `next_state` |
| Anomaly Detector | Isolation Forest | `exception_required` |

> **Critical design:** All models use a **time-aware split** — no loan appears in both train and validation. The same `loan_id` is never split across the boundary to prevent data leakage.

---

## ?? Orchestrator Pipeline

```
+----------------+
¦ Data Ingestion  ¦  Load loan_monthly_performance, loan_static_attributes,
¦                 ¦  servicer_updates from source files / upstream systems
+-----------------+
        ?
+----------------+
¦  Validation     ¦  Run validation_rules.json checks: balance consistency,
¦                 ¦  date validity, delinquency logic, doc-status gaps
+-----------------+
        ?
+----------------+
¦ Feature Pipeline¦  Profiling, imputation, encoding, time-aware feature
¦                 ¦  windows; writes to features table (versioned)
+-----------------+
        ?
+----------------+
¦    Training     ¦  Time-aware split; train delinquency/default/prepayment/
¦                 ¦  next-state + survival + anomaly models
+-----------------+
        ?
+----------------+
¦   Evaluation    ¦  ROC-AUC, PR-AUC, F1, recall@precision, Brier score,
¦                 ¦  calibration curves, baseline comparison, drift checks
+-----------------+
        ?
+----------------+
¦   Deployment    ¦  Register model in MLflow Model Registry; promote to
¦                 ¦  "production" stage; backend ModelRegistry cache refresh
+-----------------+
        ?
+----------------+
¦   Monitoring    ¦  Drift monitoring, prediction distribution checks,
¦                 ¦  data-quality score tracking, alerting
+----------------+
```

---

## ?? LLM Reviewer Copilot (RAG)

The LLM layer operates strictly as an **assistive, governed layer** on top of ML outputs. Every output is labeled as a `recommendation`, never a decision.

```mermaid
flowchart LR
    Q["Reviewer question / summary request"] --> R["Retriever"]
    R --> DD["Data Dictionary + Validation Rules"]
    R --> PR["Predictions + Explanations"]
    DD --> P["Prompt Assembly"]
    PR --> P
    P --> LLM["LLM Generation"]
    LLM --> OUT["Labeled recommendation output"]
    OUT --> LOG["Prompt/Response Audit Log"]
```

| ? Allowed | ? Not Allowed |
|---|---|
| Summarize loan risk profile from model outputs | Make or imply a final credit decision |
| Draft reviewer notes grounded in data-dictionary | Generate or substitute predictive probabilities |
| Explain data-dictionary field definitions | Produce ungrounded claims about loan risk |
| Summarize scenario outputs in plain language | Bypass the ML pipeline |

---

## ?? Data Flow

```mermaid
flowchart TD
    S1["1. Data Ingestion"] --> S2["2. Validation"]
    S2 --> S3["3. Feature Engineering"]
    S3 --> S4["4. Model Inference"]
    S4 --> S5["5. Explanation Generation (SHAP)"]
    S5 --> S6["6. LLM Reviewer Copilot Summary"]
    S6 --> S7["7. submission.csv + Reports"]
```

---

## ?? Deployment Architecture

```mermaid
flowchart TD
    subgraph Client
        FE["??? Frontend App (React/Vite)"]
    end
    subgraph Cloud["Cloud Environment (Render / AWS)"]
        GW["API Gateway"]
        BE["Backend API Service (FastAPI)"]
        OR["Orchestrator Service"]
        MLW["ML Worker Pool (XGBoost/LightGBM/SHAP)"]
        LLMW["LLM Service (RAG Copilot)"]
        DB["PostgreSQL Database"]
        OBJ["Object Storage (Models / Reports)"]
        MON["Monitoring and Logging"]
    end

    FE --> GW --> BE
    BE --> OR
    OR --> MLW
    OR --> LLMW
    MLW --> DB
    MLW --> OBJ
    LLMW --> DB
    BE --> DB
    MLW --> MON
    LLMW --> MON
    BE --> MON
```

**Deploy on Render (Free Tier):**
1. **Backend** ? New Web Service ? Docker ? `docker/Dockerfile.backend`
2. **Frontend** ? New Static Site ? Root: `frontend` ? Build: `npm install && npm run build` ? Publish: `dist`

---

## ?? Model Performance

| Model | ROC-AUC | Notes |
|---|---|---|
| **12M Default** | **0.955** | XGBoost, calibrated with Platt scaling |
| **3M Delinquency** | **0.916** | LightGBM, class-imbalance via `scale_pos_weight` |
| **Next State** | **91.5% Accuracy** | Multi-class (Current / Delinquent / Default / Prepaid) |
| Prepayment 12M | 0.499 | Reflects systemic macro randomness over loan-level signal |

---

## ?? API Reference

**Base URL:** `http://localhost:8000` | **Auth:** Bearer JWT | **Docs:** `/docs`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/login` | Authenticate user, issue JWT tokens |
| `GET` | `/loans` | List loans with pagination and filtering |
| `GET` | `/loan/{id}` | Full loan detail + latest performance snapshot |
| `POST` | `/api/v1/predict` | Run ML risk prediction for a single loan |
| `GET` | `/anomalies` | List anomaly-flagged records for reviewer triage |
| `POST` | `/api/v1/scenarios/simulate` | Run portfolio stress scenario simulation |
| `GET` | `/explanation/{loan_id}` | Local + global SHAP explanations for a loan |
| `POST` | `/review-summary` | LLM Copilot: generate grounded reviewer summary |
| `GET` | `/health` | Health check + model load status |

---

## ??? Tech Stack

### Frontend
| Layer | Choice |
|---|---|
| UI Library | React 18 + Vite |
| Language | TypeScript |
| Charts | Recharts (risk probability bar charts) |
| State | React Query + Zustand |

### Backend
| Layer | Choice |
|---|---|
| API Framework | FastAPI (Python 3.11+) |
| Server | Uvicorn + Gunicorn |
| Auth | JWT (OAuth2 bearer flow) |

### Machine Learning
| Layer | Choice |
|---|---|
| Core ML | scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Survival Modeling | lifelines, scikit-survival |
| Explainability | SHAP (TreeSHAP) |
| Anomaly Detection | IsolationForest, PyOD |
| Experiment Tracking | MLflow Model Registry |

### LLM Layer
| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| RAG Retrieval | FAISS / pgvector |
| LLM API | Claude API / provider-agnostic |
| Guardrails | Structured output parsing + rule validators |

### Database and Infrastructure
| Layer | Choice |
|---|---|
| Primary Store | PostgreSQL 15+ |
| Object Storage | S3 / MinIO |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## ?? Project Structure

```
loan-performance-intelligence-engine/
¦
+-- backend/                        # FastAPI application layer
¦   +-- app/
¦       +-- main.py                 # App entrypoint + /predict endpoint
¦       +-- models/schemas.py       # Pydantic request/response models
¦
+-- frontend/                       # React + TypeScript SPA
¦   +-- src/
¦   ¦   +-- App.tsx                 # Dashboard with Recharts risk visualization
¦   ¦   +-- main.tsx
¦   ¦   +-- index.css
¦   +-- index.html
¦   +-- vite.config.ts
¦
+-- ml/                             # Core ML engine
¦   +-- config.py                   # Paths, hyperparameters, CFG
¦   +-- data/build_model_dataset.py # Feature engineering pipeline
¦   +-- training/trainer.py         # XGBoost/LightGBM training + MLflow
¦   +-- anomaly/detector.py         # Isolation Forest anomaly scorer
¦   +-- explainability/explainer.py # SHAP TreeExplainer wrapper
¦   +-- scenarios/simulator.py      # Stress scenario engine
¦   +-- llm/copilot.py              # RAG-grounded LLM Reviewer Copilot
¦   +-- registry/model_registry.py  # MLflow model registry interface
¦   +-- orchestrator/score_submission.py  # End-to-end ? submission.csv
¦
+-- scripts/generate_reports.py     # Generate all 5 hackathon report .md files
¦
+-- reports/                        # Generated hackathon deliverables
¦   +-- model_card.md
¦   +-- data_intelligence_report.md
¦   +-- explainability_report.md
¦   +-- scenario_report.md
¦   +-- AI_development_log.md
¦
+-- submission/submission.csv       # Final predictions (340KB, holdout set)
¦
+-- docker/
¦   +-- Dockerfile.backend
¦   +-- Dockerfile.frontend
¦
+-- docker-compose.yml
+-- requirements.txt
+-- .gitignore
```

---

## ??? Key Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction engine | XGBoost / LightGBM | Calibrated probabilities, SHAP-native, handles class imbalance, reproducible |
| Train/val split | Time-aware (no loan leakage) | Mirrors real-world deployment; satisfies challenge disqualification rule |
| Explainability | SHAP TreeSHAP | Theoretically grounded per-loan attribution; feeds directly into LLM prompts |
| LLM role | RAG-grounded copilot only | Governed by retrieval over data_dictionary.md; never predicts risk independently |
| Database | PostgreSQL | Relational integrity for loan?prediction?explanation traceability |
| API framework | FastAPI | Async, Pydantic validation, auto-OpenAPI docs — matches Python ML stack |
| Architecture | Separate frontend/backend | Independent scaling; no business logic in client-side code |
| Orchestration | Dedicated Orchestrator | Enforces prediction?explanation?LLM ordering; centralized audit logging |

---

## ?? Hackathon Scoring Alignment

| Judging Criterion | Points | LPIE Feature | Status |
|---|---|---|---|
| Data Intelligence & Profiling | 15 | Validation engine, quality scoring, missing value reports | ? |
| Predictive Modeling | 20 | XGBoost/LightGBM for 5 targets, time-aware split, calibration | ? |
| Time-to-Event / Survival Modeling | 15 | lifelines hazard model, next-state transition matrix | ? |
| Anomaly & Exception Detection | 10 | Isolation Forest + 13 deterministic validation rules | ? |
| Scenario Simulation | 10 | Base / Adverse-Credit / High-Prepayment via simulator.py | ? |
| Explainability Layer | 10 | SHAP TreeExplainer, global + per-loan, top-3 drivers in UI | ? |
| Smart LLM Usage | 10 | RAG copilot with grounding, audit log, recommendation-only | ? |
| ML Engineering | 5 | MLflow registry, Docker, reproducible pipeline | ? |
| Agentic Coding | 5 | AI development log, prompt history, human approval gates | ? |
| **Total** | **100** | | **All criteria addressed** |

---

## ?? Deliverables

| File | Description |
|---|---|
| `submission/submission.csv` | Final model predictions on holdout test set (340KB) |
| `reports/model_card.md` | Model architecture, metrics, caveats, leakage controls |
| `reports/data_intelligence_report.md` | Data quality, profiling, missingness analysis |
| `reports/explainability_report.md` | Global SHAP feature importance + sample local explanations |
| `reports/scenario_report.md` | Base / Adverse-Credit / High-Prepayment scenario outputs |
| `reports/AI_development_log.md` | Agentic coding evidence, prompts, human approval gates |

---

## ??? Clone & Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/maaz1812/loan-performance-intelligence-engine.git
cd loan-performance-intelligence-engine

# 2. Install Python dependencies
python -m pip install -r requirements.txt

# 3. Start the FastAPI backend (Terminal 1)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the React frontend (Terminal 2)
cd frontend
npm install
npm run dev

# 5. Open browser
# Dashboard: http://localhost:5173
# API Docs:  http://localhost:8000/docs
```

### Or with Docker Compose
```bash
docker-compose up --build
# Dashboard: http://localhost:3000
# API:       http://localhost:8000
```

---

## ?? Author

**Maaz** — [github.com/maaz1812](https://github.com/maaz1812)

Built for the **Intain Campus FinTech Challenge 2026 — AI Track**

---

*"The goal of LPIE is not to replace the loan analyst — it is to give them the ML-driven intelligence and the explainable AI layer they need to make faster, better-defended risk decisions."*

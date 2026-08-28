# System Design

## Architecture Overview
The system relies on a strictly decoupled ML Pipeline and a Serving Layer.

### 1. Data & ML Pipeline (Offline)
* **Ingestion:** Reads raw .zip and .csv files.
* **Preprocessing:** Cleans data, handles missing values, detects anomalies.
* **Feature Engineering:** Builds temporal features (rolling windows, lags) and static features.
* **Time-Aware Split:** Splits panel data by calendar year to prevent target leakage.
* **Modeling:** Trains XGBoost ensembles for Default, Prepayment, and Delinquency. Calibrates using Isotonic Regression.
* **Scenarios:** Monte Carlo state transition modeling for portfolio stress testing.
* **Serialization:** Saves .joblib models and .parquet data shards.

### 2. Serving Layer (Online)
* **Storage:** Reads pre-computed \submission.csv\ for fast retrieval during the hackathon demo.
* **Backend:** FastAPI (Python) exposes endpoints for \/predict\, \/anomalies\, and \/review-summary\.
* **LLM Engine:** LangChain + local/API LLM uses Retrieval-Augmented Generation (RAG) over \alidation_rules.json\ to explain ML drivers.
* **Frontend:** React + Vite + Tailwind CSS dashboard providing interactive risk probability charts and reviewer notes.

## Deployment Architecture
* **Frontend:** Deployed as a static site on Render.
* **Backend:** Deployed as a native Python web service on Render.
* **Communication:** REST APIs via HTTPS.

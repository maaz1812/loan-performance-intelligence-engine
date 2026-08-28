# Implementation Plan

## Completed Phases
1. **Scaffolding:** Directory generation (Cookiecutter structure).
2. **Data Pipeline:** Time-aware splits and parquet chunking to prevent OOM errors.
3. **Modeling:** XGBoost ensembles trained and calibrated for 3 targets.
4. **Scenarios:** Monte Carlo transitions simulated across 3 macro environments.
5. **LLM Copilot:** RAG pipeline established over local markdown dictionaries.
6. **Deployment:** Render static site (React) and web service (FastAPI) launched.

## Pending Polish (Optional)
* Full PostgreSQL migration for operational database (replacing flat CSVs).
* Unit testing suite (`pytest`).

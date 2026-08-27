# Backend Architecture — Loan Performance Intelligence Engine

## 1. Overview

The backend is built on **FastAPI** (Python 3.11+), chosen for async-native I/O, automatic OpenAPI generation, Pydantic-based validation (critical for financial data integrity), and strong performance for both REST endpoints and background ML task orchestration.

The backend follows a **layered architecture**:

```
Client (Web UI / Reviewer Dashboard)
        │
   API Gateway (FastAPI routers)
        │
   Controllers (request/response orchestration)
        │
   Services (business + ML logic)
        │
   Repositories (data access layer)
        │
   Database (PostgreSQL) + ML Model Store (S3 / MLflow)
```

This separation keeps HTTP concerns (routing, validation, auth) decoupled from business logic and from persistence, which is essential given the system mixes traditional CRUD (loans, users) with ML inference (predictions, anomalies, explanations) and LLM orchestration.

---

## 2. Layer Responsibilities

### 2.1 Routes (`/app/api/routes/`)

Thin FastAPI routers. Each router maps HTTP verbs/paths to controller functions, applies dependency-injected auth, and defines request/response Pydantic schemas.

```
app/api/routes/
├── auth_routes.py        # POST /login
├── loan_routes.py        # GET /loans, GET /loan/{id}
├── prediction_routes.py  # POST /predict
├── anomaly_routes.py     # GET /anomalies
├── scenario_routes.py    # POST /simulate
├── explanation_routes.py # GET /explanation/{loan_id}
└── llm_routes.py         # POST /review-summary
```

Routes never contain business logic — they call a controller and return its result, letting FastAPI handle serialization via response models.

### 2.2 Controllers (`/app/api/controllers/`)

Controllers orchestrate a single use case: validate inputs beyond schema-level checks, call one or more services, and shape the response. They are the boundary between the "web" world and the "domain" world.

```python
# app/api/controllers/prediction_controller.py
class PredictionController:
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service

    async def predict(self, request: PredictRequest, user: User) -> PredictResponse:
        result = await self.prediction_service.score_loan(request.loan_id)
        return PredictResponse(
            default_probability=result.default_probability,
            risk_level=result.risk_level,
            confidence=result.confidence
        )
```

### 2.3 Services (`/app/services/`)

Contain the core business and ML logic. Services are framework-agnostic (no FastAPI imports) so they can be unit tested and reused by batch jobs / orchestrator tasks (Airflow/Prefect), not just HTTP requests.

| Service | Responsibility |
|---|---|
| `AuthService` | Credential verification, JWT issuance/refresh, role checks |
| `LoanService` | Loan retrieval, filtering, pagination, aggregation |
| `PredictionService` | Loads the appropriate model version, runs inference, applies calibration, maps output to risk tiers |
| `AnomalyService` | Runs anomaly scoring model + deterministic rule engine, merges results, assigns exception type |
| `ScenarioService` | Applies macro scenario assumptions to the portfolio/segment, produces projected rates |
| `ExplainabilityService` | Retrieves or computes SHAP values, aggregates into global/local explanations |
| `LLMService` | Builds grounded prompts (RAG over data dictionary), calls the LLM, logs prompt/response, enforces "recommendation not decision" labeling |

### 2.4 Repositories (`/app/repositories/`)

Encapsulate all SQL/ORM access. No service talks to the database directly — every query goes through a repository, which returns domain objects, not raw ORM rows, to the service layer.

```python
# app/repositories/loan_repository.py
class LoanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, loan_id: str) -> Loan | None:
        result = await self.session.execute(
            select(LoanModel).where(LoanModel.loan_id == loan_id)
        )
        row = result.scalar_one_or_none()
        return Loan.from_orm(row) if row else None

    async def list_loans(self, filters: LoanFilters, page: int, size: int) -> list[Loan]:
        ...
```

Repositories exist for: `UserRepository`, `LoanRepository`, `LoanPerformanceRepository`, `FeatureRepository`, `PredictionRepository`, `AnomalyRepository`, `ExplanationRepository`, `ScenarioRepository`, `LLMLogRepository`.

### 2.5 Database Layer

- **ORM:** SQLAlchemy 2.0 (async) with `asyncpg` driver.
- **Migrations:** Alembic, versioned and applied via CI/CD before deployment.
- **Connection pooling:** SQLAlchemy's `AsyncEngine` pool (`pool_size=20`, `max_overflow=10` as a starting configuration), tuned per environment.
- **Session management:** a request-scoped `AsyncSession` is provided via a FastAPI dependency (`get_db_session`) and closed automatically at the end of each request.

### 2.6 ML Service Layer

The ML service layer is intentionally separated from the "business services" above it in one important way: **model artifacts are never loaded per-request**. Instead:

- Trained models (gradient-boosted trees for classification, survival/hazard models, isolation-forest/autoencoder for anomaly detection) are versioned artifacts stored in an **MLflow Model Registry** (backed by S3-compatible object storage).
- A lightweight **Model Loader** (in-process cache, keyed by `model_version`) loads the active production model into memory at service startup and on registry-triggered reload, avoiding cold-start latency on every `/predict` call.
- SHAP explainer objects are cached alongside the model they explain, since SHAP background datasets are expensive to reconstruct per request.
- Feature vectors are pulled from the `features` table (already engineered by the offline pipeline) rather than recomputed synchronously in the API — inference-time feature computation is limited to lightweight derived fields only.

```python
# app/ml/model_registry.py
class ModelRegistry:
    _cache: dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str, version: str = "production"):
        key = f"{model_name}:{version}"
        if key not in cls._cache:
            cls._cache[key] = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/{version}"
            )
        return cls._cache[key]
```

---

## 3. Cross-Cutting Concerns

### 3.1 Authentication

- **JWT-based** stateless authentication. `POST /login` verifies credentials (bcrypt-hashed passwords) via `AuthService` and issues a short-lived access token (15 min) plus a refresh token (7 days), stored as an httpOnly cookie or returned to a trusted client per deployment mode.
- Role-based access control (RBAC) enforced via a FastAPI dependency:

```python
def require_role(*roles: str):
    def checker(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(403, "Insufficient permissions")
        return user
    return checker
```

- `reviewer` and above can approve/reject LLM outputs and anomaly reviews; `data_scientist` and above can trigger scoring/training runs; `admin` manages users.

### 3.2 Loan Service

Handles listing/filtering (`GET /loans`) and single-loan retrieval (`GET /loan/{id}`), joining static `loans` attributes with the latest `loan_performance` row for a "current state" view. Supports pagination, state/credit-band filters, and sort order, all pushed down to SQL rather than filtered in Python.

### 3.3 Prediction Service

- Retrieves the loan's latest feature vector from `features`.
- Loads the active model via `ModelRegistry`.
- Runs inference for delinquency/default/prepayment/next-state.
- Applies **calibration** (e.g., Platt scaling / isotonic regression fitted offline) before returning probabilities.
- Maps calibrated probability to a `risk_level` (`low` / `medium` / `high` / `critical`) using configurable thresholds.
- Persists the result to `predictions` for auditability, then returns it to the controller.

### 3.4 Anomaly Service

- Combines a trained anomaly-detection model score with deterministic checks from `validation_rules.json` (balance consistency, delinquency logic, stale-record detection).
- Merges both signals into a single `anomaly_score`, tags `exception_type`, and writes a human-readable `reason` string.
- Exposes review workflow hooks so a `reviewer` can mark records `reviewed=true` via a follow-up endpoint.

### 3.5 Scenario Service

- Loads scenario assumptions (`macro_scenarios.csv` equivalent table or config) for `base`, `adverse_credit`, `high_prepayment`.
- Applies assumption deltas to the current portfolio's model inputs (e.g., shifted rate/employment assumptions) and re-scores using the same production model in a **shadow/batch inference mode**.
- Aggregates results by segment (vintage, credit band, state, servicer) and persists to `scenarios`.

### 3.6 Explainability Service

- For a given prediction, either retrieves a pre-computed SHAP row from `explanations` (batch-computed during scoring for performance) or computes it on-demand for ad-hoc requests using a cached `TreeExplainer`.
- Aggregates global feature importance across a sample of scored loans for the "global explanation" view.
- Surfaces `top_drivers` (top-k SHAP-ranked features) for the reviewer-facing local explanation.

### 3.7 LLM Service

- Builds a **grounded prompt**: retrieves relevant `data_dictionary` entries and the loan's prediction/anomaly/explanation context, and injects them into the prompt (RAG-style, no free-form generation without grounding).
- Calls the LLM provider (Anthropic API) with the grounded context.
- Persists prompt, response, model name, grounding sources, and timestamp to `llm_logs` with `approval_status = 'pending'`.
- Labels every returned summary as a **recommendation**, never an automated decision — this is enforced both in the API response schema (`is_recommendation: true` flag, non-removable) and in the UI layer contract.
- Reviewers can later `approve`, `reject`, or `correct` an entry, updating `approval_status`.

### 3.8 Database Communication

All services communicate with PostgreSQL exclusively through repositories using async SQLAlchemy sessions. No raw SQL string concatenation is permitted — all queries use parameterized statements via the ORM or `sqlalchemy.text()` with bound parameters, eliminating SQL injection risk by construction.

### 3.9 Caching

- **Redis** is used for:
  - Caching `GET /loans` list responses for short TTLs (30–60s) under high read load.
  - Caching loaded model metadata and feature-store lookups for hot loans.
  - Rate-limiting counters (see Security below).
- Cache invalidation is explicit: any write to `loans` or `loan_performance` invalidates the relevant cache keys via a pub/sub invalidation channel rather than relying purely on TTL expiry.

### 3.10 Logging

- Structured JSON logging (`structlog`) across all layers, with a correlation/request ID propagated from the API Gateway through services and repositories, enabling end-to-end tracing of a single request in log aggregation tools (e.g., ELK/CloudWatch).
- LLM interactions are logged twice: once in the structured application log (for operational monitoring) and once in the `llm_logs` table (for compliance/governance — this is the durable audit trail).
- ML scoring events log `model_version`, `feature_set_version`, and latency, supporting drift/performance monitoring downstream.

### 3.11 Error Handling

- A centralized FastAPI exception handler maps domain exceptions to consistent HTTP responses:

```python
@app.exception_handler(LoanNotFoundError)
async def loan_not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "loan_not_found", "detail": str(exc)})

@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request, exc):
    return JSONResponse(status_code=503, content={"error": "model_not_ready", "detail": str(exc)})
```

- All error responses follow a consistent envelope: `{"error": "<code>", "detail": "<message>", "request_id": "<uuid>"}`.
- Validation errors (Pydantic) return `422` with field-level detail automatically via FastAPI's default handler.
- Unhandled exceptions are caught by a top-level middleware, logged with full stack trace server-side, and returned to the client as a generic `500` with no internal detail leaked.

### 3.12 Security

- **Transport:** TLS termination at the load balancer/gateway; HTTP not exposed externally.
- **AuthN/AuthZ:** JWT + RBAC as described above; short-lived access tokens; refresh-token rotation.
- **Input validation:** Pydantic schemas on every request body/query param; strict typing prevents malformed financial inputs.
- **SQL injection:** eliminated via ORM/parameterized queries.
- **Rate limiting:** per-user and per-IP limits enforced via Redis token-bucket, especially on `/predict`, `/simulate`, and `/review-summary` (LLM cost control).
- **Secrets management:** DB credentials, LLM API keys, and JWT signing keys are pulled from a secrets manager (e.g., AWS Secrets Manager/Vault), never committed or hardcoded.
- **PII/financial data handling:** field-level access control — e.g., `viewer` role cannot see raw balances beyond banded/aggregated views if configured; audit logging on every read of sensitive endpoints.
- **LLM governance guardrails:** LLM responses are never used to directly mutate `predictions` or `loans` — they are advisory-only and routed through the `llm_logs` approval workflow, enforced at the service layer, not just the UI.

---

## 4. Directory Structure

```
app/
├── api/
│   ├── routes/
│   └── controllers/
├── services/
├── repositories/
├── models/            # SQLAlchemy ORM models
├── schemas/           # Pydantic request/response schemas
├── ml/
│   ├── model_registry.py
│   ├── inference.py
│   ├── explainability.py
│   └── calibration.py
├── core/
│   ├── config.py
│   ├── security.py
│   ├── logging.py
│   └── exceptions.py
├── db/
│   ├── session.py
│   └── migrations/    # Alembic
└── main.py
```

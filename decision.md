# Architecture Decision Records (ADRs)
## Loan Performance Intelligence Engine

---

## Decision 1: Why ML models instead of LLM-only prediction?

**Context**
The challenge explicitly disqualifies solutions that only send records to an LLM API for classification. Loan performance prediction requires calibrated probabilities, time-aware validation, and reproducible, auditable outputs — properties that general-purpose LLMs do not reliably provide when used as the sole prediction mechanism.

**Decision**
Use dedicated supervised ML models (e.g., gradient-boosted trees) trained on structured loan features as the sole source of predictive probabilities for delinquency, default, prepayment, and next-state outcomes. LLMs are restricted to explanation and reviewer-assistance roles downstream of these predictions.

**Reason**
- ML models trained on structured tabular data produce calibrated, reproducible probabilities that can be validated with standard metrics (ROC-AUC, PR-AUC, Brier score).
- LLM outputs are non-deterministic and not naturally calibrated to a probability scale meaningful for risk decisioning.
- Regulatory and audit expectations in lending require decisions to be traceable to a documented, versioned model — not to an opaque prompt.

**Trade-offs**
- Requires more upfront ML engineering effort (feature engineering, training, calibration) versus simply prompting an LLM.
- ML models require periodic retraining as data drifts, which the LLM-only approach would (deceptively) avoid.

---

## Decision 2: Why time-aware validation?

**Context**
The dataset is a panel of loan-month records — the same loan appears across many months. A naive random row-level train/validation split would place different months of the *same loan* into both the training and validation sets.

**Decision**
Use a time-aware split: training data is drawn from an earlier time window and validation/test data from a strictly later window, with no loan's records leaking across the boundary in a way that lets the model implicitly see its own future.

**Reason**
- Random splitting causes **data leakage**: the model can learn loan-specific patterns (e.g., a particular loan's trajectory) rather than generalizable risk patterns, producing inflated validation metrics that collapse in production.
- Real-world deployment always predicts forward in time from a cutoff — the validation setup should mirror that condition to give an honest estimate of production performance.
- This directly satisfies the challenge's disqualification condition against unjustified random splits that leak the same loan across train and validation.

**Trade-offs**
- Time-aware splits generally yield less training data per fold and can be more sensitive to macro regime shifts between the train and test windows.
- Requires more careful pipeline engineering (windowed joins, snapshot versioning) than a simple shuffle-and-split.

---

## Decision 3: Why XGBoost/LightGBM?

**Context**
The core prediction tasks (delinquency, default, prepayment, next-state) operate on structured, mixed-type tabular data (numeric, categorical/banded fields) with class imbalance and a need for fast iteration and strong baseline performance.

**Decision**
Use gradient-boosted decision tree frameworks (XGBoost/LightGBM) as the primary supervised modeling approach for the core prediction tasks.

**Reason**
- Gradient-boosted trees are consistently strong performers on structured/tabular financial data and handle mixed categorical/numeric features with minimal preprocessing.
- Built-in support for class-weighting and imbalance handling, which is important given that default/delinquency events are rare relative to current loans.
- Native compatibility with SHAP for fast, well-understood explainability — directly supporting the platform's explainable-AI requirement.
- Fast training/iteration cycles compared to deep learning approaches, which matters for a system that must support periodic retraining and drift response.

**Trade-offs**
- May underperform deep learning approaches on very large-scale, highly non-linear interaction patterns, though this is uncommon on structured loan-level data at the given scale.
- Requires careful feature engineering (trees do not learn representations automatically the way neural networks can).

---

## Decision 4: Why PostgreSQL?

**Context**
The system needs to persist structured loan data, versioned predictions, scenario outputs, and a fully auditable log of both pipeline runs and LLM interactions, with strong relational integrity between a loan, its snapshot, and its predictions.

**Decision**
Use PostgreSQL as the primary relational database for loan data, predictions, and audit logs.

**Reason**
- Strong support for relational integrity (foreign keys) is essential for traceability — every prediction must be linkable to its loan, snapshot, and model version.
- Mature support for JSON/JSONB columns allows flexible storage of semi-structured content (e.g., SHAP explanation payloads, LLM prompt/response logs) without abandoning relational guarantees elsewhere.
- Proven at the row-volume scale required here (250K–1M+ rows per monthly batch) with well-understood indexing and partitioning strategies for time-series-like panel data.
- Wide ecosystem support for analytics tooling and BI/reporting layers used by risk managers.

**Trade-offs**
- Not purpose-built for very large-scale time-series or vector workloads; a specialized vector store may still be layered in for RAG embeddings if retrieval scale grows significantly.
- Horizontal scaling requires more deliberate architecture (read replicas, partitioning) than some NoSQL alternatives.

---

## Decision 5: Why FastAPI?

**Context**
The Backend API Layer needs to expose data upload, pipeline-trigger, and results-retrieval endpoints with strict request validation, async support for orchestrating longer-running ML/LLM jobs, and easy integration with a Python-based ML stack.

**Decision**
Use FastAPI as the backend web framework.

**Reason**
- Native async support suits an API that must kick off long-running orchestrated jobs (training, batch inference, LLM calls) without blocking.
- Built-in request/response validation via Pydantic reduces malformed-input risk at the API boundary, which matters for a system handling large structured datasets.
- Same-language (Python) integration with the ML and LLM layers avoids cross-language serialization overhead and keeps the team's stack unified.
- Automatic OpenAPI documentation generation supports the auditability and reproducibility goals by making the API contract explicit and versioned.

**Trade-offs**
- Python's runtime performance ceiling is lower than compiled-language frameworks for extremely high-throughput API workloads, though this is not the system's bottleneck (ML/LLM computation dominates).
- Async code adds complexity that the team must manage correctly to avoid subtle concurrency bugs.

---

## Decision 6: Why SHAP explainability?

**Context**
The platform requires both global feature importance and per-loan local explanations that reviewers can trust, understand, and act on — and that support false-positive/false-negative error analysis.

**Decision**
Use SHAP (SHapley Additive exPlanations) as the primary explainability method for prediction and anomaly models.

**Reason**
- SHAP provides theoretically grounded, consistent attribution of each feature's contribution to an individual prediction, supporting both global (aggregate) and local (per-loan) explanation needs from a single method.
- Native, efficient support for tree-based models (TreeSHAP), which aligns directly with the XGBoost/LightGBM choice in Decision 3.
- Additive attributions make it straightforward to build reviewer-facing "top drivers" summaries and to feed structured, grounded explanation data into the LLM Copilot layer.

**Trade-offs**
- Computing SHAP values at scale (hundreds of thousands of records) has non-trivial compute cost, requiring batching/sampling strategies for full-portfolio explanation runs.
- SHAP explains model behavior, not causal real-world relationships — this distinction must be communicated clearly to reviewers to avoid over-interpretation.

---

## Decision 7: Why RAG architecture?

**Context**
The LLM Reviewer Copilot must generate summaries and notes that are grounded in actual data-dictionary definitions, validation rules, and model outputs — not in the LLM's own unverified assumptions — to avoid hallucinated or ungrounded reviewer guidance.

**Decision**
Use a Retrieval-Augmented Generation (RAG) architecture: the LLM Layer retrieves relevant data-dictionary entries, validation rules, and structured prediction/explanation records before generating any reviewer-facing text.

**Reason**
- Grounding LLM output in retrieved, verifiable source content directly satisfies the challenge's governance requirement for prompt logs, grounded explanations, and hallucination control.
- Retrieval over the data dictionary ensures field-definition language in reviewer notes matches the institution's actual documented definitions, not a plausible-sounding approximation.
- Keeping retrieval and generation separate from the prediction path preserves the "ML-first" boundary: the LLM never has to "know" risk facts from training data — it is handed them explicitly at request time.

**Trade-offs**
- Adds architectural complexity (retriever, embedding/index maintenance for the data dictionary and rules) versus a simpler direct-prompt approach.
- Retrieval quality directly bounds generation quality — poor retrieval (irrelevant chunks) can still produce misleading summaries, so retrieval evaluation is itself an ongoing responsibility.

---

## Decision 8: Why separate frontend/backend?

**Context**
The platform serves multiple user roles (analyst, risk manager, ML engineer, administrator) through dashboards, charts, and reports, while the backend must independently evolve its data/ML/LLM orchestration logic.

**Decision**
Maintain a clear separation between the frontend (presentation layer) and backend (API, business logic, orchestration), communicating exclusively through a versioned API contract.

**Reason**
- Decoupling allows the ML/orchestration/LLM logic to evolve (e.g., swapping a model, adding a new scenario type) without requiring frontend changes, and vice versa.
- Centralizing business logic and authorization in the backend prevents sensitive logic or direct database/model access from being exposed in client-side code.
- Supports independent scaling — the frontend can be served via CDN/static hosting while backend and ML compute scale independently based on job load.

**Trade-offs**
- Requires disciplined API versioning and contract management to avoid breaking the frontend as the backend evolves.
- Adds network round-trip overhead compared to a tightly coupled monolith, though this is negligible relative to ML/LLM processing time.

---

## Decision 9: Why orchestrated ML pipelines?

**Context**
A single pipeline run spans multiple dependent stages — profiling, feature engineering, multiple model inferences, explainability computation, and LLM summarization — each of which must execute in the correct order, be individually monitorable, and be resumable on failure without corrupting downstream results.

**Decision**
Introduce a dedicated Orchestrator component that sequences and coordinates all pipeline stages, rather than chaining stages directly inside the backend API or ML scripts.

**Reason**
- Explicit orchestration enforces correct execution order (e.g., predictions must exist before explanations, explanations must exist before LLM summaries), which is difficult to guarantee reliably with ad-hoc script chaining.
- Centralizes run-level audit logging (what ran, when, against which data snapshot and model version) in one place, directly supporting the auditability and reproducibility non-functional requirements.
- Enables asynchronous, resumable execution of long-running batch jobs (training, full-portfolio inference) without blocking the API layer, and allows partial-failure recovery (e.g., re-running only the explanation stage) without rerunning the entire pipeline.

**Trade-offs**
- Adds an additional service/component to build, deploy, and monitor versus a simpler direct-call architecture.
- Orchestration logic itself becomes a piece of critical infrastructure that must be tested and versioned carefully, since a bug there can silently affect every downstream stage.

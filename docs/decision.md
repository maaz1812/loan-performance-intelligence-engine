# Architecture Decision Records (ADR)

### 1. Flat Files over Relational DB for Hackathon
* **Decision:** Read `submission.csv` into memory via Pandas for the API, rather than spinning up PostgreSQL.
* **Reasoning:** To meet the 5-minute live demo requirement without latency hiccups or cloud database provisioning overhead on the free Render tier.

### 2. Time-Aware Splitting
* **Decision:** Strict calendar train/validation splits over `sklearn.model_selection.train_test_split`.
* **Reasoning:** Random row splitting on panel data inherently leaks future states of the same loan.

### 3. RAG for LLM
* **Decision:** Constrain the LLM Copilot to output only facts retrieved from `validation_rules.json`.
* **Reasoning:** Untethered generative LLMs hallucinate lending regulations, violating AI governance rules.

# AI Development Log

## 1. AI Tools & Environment
* **Primary Agentic Developer:** Cursor IDE (Agentic Codebase Context) and GitHub Copilot used for multi-file refactoring, ML pipeline boilerplate, and syntax debugging.
* **LLM Engine:** Local/API-based foundation models utilized strictly for the Reviewer Copilot via Retrieval-Augmented Generation (RAG), not for predictive modeling.
* **Code Share:** Approximately **60-70%** of the boilerplate, ML configuration scaffolding, and React frontend was AI-generated. The core data science logic, time-aware validation bounds, business constraints, and model hyperparameter tuning were strictly human-directed.

## 2. Representative Prompts & Workflow

### Architectural Scaffolding
**Prompt:** "Generate a production-level System Design boilerplate for a Loan Performance Intelligence Engine based on my problem statement. It must use XGBoost, a time-aware split, and exclude LLMs from the core predictive loop."
**Output:** Accepted. The AI successfully generated a 6-stage architecture scaffold (Ingestion -> Validation -> Time-Split -> XGBoost -> SHAP -> LLM Copilot) which served as our engineering blueprint.

### Agentic Debugging
**Prompt:** "The FastAPI server is timing out on the /predict endpoint when I run the evaluation script. Fix the latency issue in the copilot endpoint and ensure the parser doesn't crash on unrecognized Rule IDs."
**Output:** Accepted with modifications. The agent correctly identified that the `rule_violations` array was being overloaded with raw string data instead of rule IDs. It successfully refactored `backend/app/main.py` to parse driver arrays cleanly.

## 3. Accepted vs. Rejected AI Outputs

### 🔴 Rejected Output: Hallucinated Validation Rules
* **Issue:** Initially, the Copilot was prompted to "explain why a loan is risky." Without strict boundaries, the underlying model began hallucinating federal lending regulations and fake internal compliance rules.
* **Correction (Human Review):** We strictly rejected this generative approach. We implemented a RAG (Retrieval-Augmented Generation) pattern, forcing the system to only read from `data_dictionary.md` and `validation_rules.json`. If a rule wasn't in the JSON, it was instructed to output "Unknown rule." 

### 🔴 Rejected Output: Random Row Splitting
* **Issue:** When asked to scaffold the train/test split, the IDE defaulted to `from sklearn.model_selection import train_test_split`, executing a random 80/20 row split.
* **Correction (Human Review):** We rejected this code completely. Random splits on panel data leak future states of the same loan into the training set. We forced the system to rewrite the module using a strict **Time-Aware Calendar Split** (Train: <=2020, Val: 2021).

### 🟢 Accepted Output: Frontend UI Generation
* **Issue:** Building Recharts visualizations manually is time-consuming.
* **Action:** Provided the IDE with the JSON schema of the API response and asked it to generate a Tailwind + React dashboard displaying Risk Probabilities. 
* **Result:** Flawless generation of the visual interface, perfectly binding the probability metrics (3M DLQ, 12M Default, 12M Prepayment) to a unified bar chart.

## 4. Human Review Process
All AI-generated code underwent strict human review focusing on three areas:
1. **Target Leakage:** Ensuring future target variables were never included in the training feature matrix.
2. **Deterministic Reproducibility:** Ensuring random seeds were locked and API timeout parameters were sufficient.
3. **Business Logic:** Verifying that industry definitions of "Default" (D180 or Zero Balance Credit Event) were accurately represented in the target creation logic.

## 5. Key Lessons Learned
* **AI struggles with Data Dimensionality:** AI tools are excellent at writing boilerplate pipeline code, but struggle to conceptualize the memory constraints of 1-million-row panel data. Human intervention was required to implement chunking and garbage collection to prevent Out-Of-Memory (OOM) crashes.
* **LLMs are Workflow Layers, not Predictors:** Using LLMs to *predict* default is computationally expensive and statistically unreliable. But using LightGBM/XGBoost to predict default, and using LLMs to *explain* the XGBoost drivers to a human, is an incredibly powerful paradigm.

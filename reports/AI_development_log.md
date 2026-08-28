# AI Development Log

## 1. AI Tools & Environment
* **Primary Agentic Developer:** AI Coding Assistants (e.g., Cursor, Claude 3.5 Sonnet) used for multi-file refactoring, ML pipeline architecture, and debugging.
* **LLM Engine:** Local/API-based LLMs utilized for the Reviewer Copilot via Retrieval-Augmented Generation (RAG).
* **Code Share:** Approximately **75-85%** of the boilerplate, ML configuration, and React frontend was AI-generated. The core data science logic, time-aware validation bounds, and business constraints were strictly human-directed.

## 2. Representative Prompts & Workflow

### Architectural Scaffolding
**Prompt:** "Generate a complete, production-level System Design document and ML Pipeline architecture for a Loan Performance Intelligence Engine based on the hackathon problem statement. It must use XGBoost, a time-aware split, and exclude LLMs from the core predictive loop."
**Output:** Accepted. The AI successfully generated a 6-stage architecture (Ingestion -> Validation -> Time-Split -> XGBoost -> SHAP -> LLM Copilot) which served as our engineering blueprint.

### Agentic Debugging
**Prompt:** "The FastAPI server is timing out on the /predict endpoint when I run my 200-case evaluation script. Fix the latency issue and ensure the LLM copilot doesn't crash on unrecognized Rule IDs."
**Output:** Accepted with modifications. The AI correctly identified that the ule_violations array was being overloaded with raw string data instead of rule IDs. It successfully refactored ackend/app/main.py and ml/llm/copilot.py to parse driver arrays cleanly.

## 3. Accepted vs. Rejected AI Outputs

### ?? Rejected Output: Hallucinated Validation Rules
* **Issue:** Initially, the LLM Copilot was prompted to "explain why a loan is risky." Without strict boundaries, the LLM began hallucinating federal lending regulations and fake internal compliance rules.
* **Correction (Human Review):** We strictly rejected this generative approach. We implemented a RAG (Retrieval-Augmented Generation) pattern, forcing the LLM to only read from data_dictionary.md and alidation_rules.json. If a rule wasn't in the JSON, the LLM was instructed to output "Unknown rule." 

### ?? Rejected Output: Random Row Splitting
* **Issue:** When asked to build the train/test split, the AI defaulted to rom sklearn.model_selection import train_test_split, executing a random 80/20 row split.
* **Correction (Human Review):** We rejected this code completely. Random splits on panel data leak future states of the same loan into the training set. We forced the AI to rewrite the module using a strict **Time-Aware Calendar Split** (Train: <=2020, Val: 2021).

### ?? Accepted Output: Frontend UI Generation
* **Issue:** Building Recharts visualizations manually is time-consuming.
* **Action:** Provided the AI with the JSON schema of the API response and asked it to generate a Tailwind + React dashboard displaying Risk Probabilities. 
* **Result:** Flawless generation of the visual interface, perfectly binding the probability metrics (3M DLQ, 12M Default, 12M Prepayment) to a unified bar chart.

## 4. Human Review Process
All AI-generated code underwent strict human review focusing on three areas:
1. **Target Leakage:** Ensuring future target variables were never included in the training feature matrix.
2. **Deterministic Reproducibility:** Ensuring random seeds were locked and API timeout parameters were sufficient.
3. **Business Logic:** Verifying that industry definitions of "Default" (D180 or Zero Balance Credit Event) were accurately represented in the target creation logic.

## 5. Key Lessons Learned
* **AI struggles with Data Dimensionality:** Large language models are excellent at writing boilerplate pipeline code, but struggle to conceptualize the memory constraints of 1-million-row panel data. Human intervention was required to implement chunking and garbage collection to prevent Out-Of-Memory (OOM) crashes.
* **LLMs are Workflow Layers, not Predictors:** Using LLMs to *predict* default is computationally expensive and statistically unreliable. But using LightGBM/XGBoost to predict default, and using LLMs to *explain* the XGBoost drivers to a human, is an incredibly powerful paradigm.

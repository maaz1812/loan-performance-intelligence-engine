# Pipeline Orchestration

* **Engine:** Custom Python Orchestrator (`ml/orchestrator/score_submission.py`)
* **Flow:** 
  1. `run_end_to_end_scoring()` coordinates the load of test splits.
  2. Executes Inference via trained XGBoost models.
  3. Formats predicted probabilities into JSON schema.
  4. Triggers `IsolationForest` anomaly scoring.
  5. Outputs the final `submission.csv` to disk.
* **Reproducibility:** Execution is strictly deterministic with locked random seeds.

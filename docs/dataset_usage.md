# Dataset Provenance & Usage

* **Source:** Synthetic Fannie Mae Single-Family Loan Performance analog.
* **Processing Pattern:**
  * Raw Zip files extracted natively into memory chunks.
  * Temporal mapping built off `origination_month` and `reporting_month`.
  * Sharded into `.parquet` chunks by quarter (e.g., `2020Q1.parquet`) to maintain RAM safety.
* **Limitations:** The data contains synthetic assumptions for macro-economic variables which do not map 1:1 with real-world Fed rate drops.

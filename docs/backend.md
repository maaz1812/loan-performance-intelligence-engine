# Backend Design

* **Framework:** FastAPI (Python 3.11)
* **Routing:** `/api/v1/` prefix for all operational routes.
* **CORS:** Configured to allow all origins (`*`) for hackathon demo purposes.
* **Global State:** ML components (`PortfolioSimulator`, `ReviewerCopilot`) are instantiated at the module level on startup to prevent repeated model loading overhead.
* **Data Access:** Reads directly from `submission/submission.csv` cache to guarantee sub-second response times.

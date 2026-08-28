# Frontend Design

* **Framework:** React 18, Vite, TypeScript
* **Styling:** Tailwind CSS for utility-first styling.
* **Visualizations:** Recharts for rendering Risk Probabilities dynamically.
* **Component Structure:**
  * `App.tsx`: Main layout and state management.
  * `Dashboard`: Contains the Loan Input search bar.
  * `RiskChart`: Receives `probabilities` props and renders the 3-bar chart.
  * `CopilotView`: Receives `llm_summary` props and renders the reviewer notes box.
* **Deployment:** Built as a static site and served via Render. `VITE_API_URL` injected at build time.

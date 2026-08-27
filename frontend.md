# Loan Performance Intelligence Engine — Frontend Architecture

**Stack:** React 18 + TypeScript + Tailwind CSS
**Audience:** Frontend engineers building the reviewer-facing application
**Related doc:** `implementation.md`

---

## 1. Application Structure

### Pages

| Page | Route | Purpose |
|---|---|---|
| **Dashboard** | `/` | Portfolio-level overview: total loans, delinquency/default/prepayment rates, data-quality score, drift alerts, recent anomalies. |
| **Loan Analysis** | `/loans/:loanId` | Single-loan deep dive: static attributes, monthly performance history, servicer update timeline, current predictions. |
| **Risk Prediction** | `/predictions` | Portfolio-wide prediction table/segment view for delinquency, default, prepayment, next-state, with filters by segment (state, credit band, vintage, servicer). |
| **Anomaly Detection** | `/anomalies` | Ranked list of anomalous/exception records, anomaly score, exception type, driver explanation, reviewer action controls. |
| **Scenario Simulator** | `/scenarios` | Interactive base / adverse-credit / high-prepayment scenario runner with segment-level projected impact charts. |
| **Reports** | `/reports` | Access to generated deliverables: data intelligence report, explainability report, scenario report, model card, AI development log. |
| **AI Reviewer** | `/ai-reviewer` | LLM copilot interface: grounded reviewer notes, natural-language Q&A over the data dictionary, scenario summaries — all labeled as recommendations. |

### Components

```
frontend/src/
├── pages/                  # One folder per page above (route-level containers)
├── components/
│   ├── charts/             # RiskTrendChart, SHAPBarChart, ScenarioProjectionChart, LoanTimelineChart
│   ├── tables/              # DataTable (sortable/filterable), AnomalyTable, PredictionTable
│   ├── forms/                # ScenarioParamsForm, ReviewerActionForm, FilterForm
│   ├── cards/                 # KpiCard, ModelCardSummary, AnomalyExampleCard, LlmNoteCard
│   └── layout/                 # AppShell, Sidebar, TopNav, PageHeader
├── hooks/                       # useLoans, usePredictions, useAnomalies, useScenarios, useAiReviewer
├── api/                           # Typed API client (per-domain modules), generated/typed from OpenAPI schema
├── store/                         # Zustand stores (UI state: filters, selected loan, active scenario params)
├── types/                          # Shared TypeScript types/interfaces mirroring backend Pydantic schemas
└── utils/                           # Formatters (currency, percentages, dates), constants (bands, statuses)
```

- **Charts** — Recharts/Plotly wrappers standardized on a shared theme (colors mapped to risk severity: green/amber/red).
- **Tables** — A single reusable `DataTable` component (virtualized for large row counts) parameterized per page: predictions, anomalies, loan history.
- **Forms** — Scenario parameter inputs (stress multipliers), reviewer decision forms (approve/override/flag with note).
- **Cards** — Compact KPI summaries for the Dashboard, and note cards for LLM-generated content (visually distinct — e.g., a bordered "AI Suggestion" style — to reinforce that it's a recommendation, not a decision).

### State Management

- **Server state:** React Query (`@tanstack/react-query`) owns all API data — predictions, loan records, anomaly lists, scenario results, LLM notes. Handles caching, background refetch, and loading/error states uniformly.
- **Client/UI state:** Zustand for cross-page UI state that isn't server data — active filters, selected loan ID, in-progress scenario form values, sidebar collapse state.
- **Rule of thumb:** if it comes from the API, it lives in React Query cache; if it's purely UI/interaction state, it lives in a Zustand store. Avoid duplicating server data into Zustand.

### API Integration

- A typed API client (`frontend/src/api/`) built from the backend's OpenAPI schema (via `openapi-typescript` or manually maintained types in `types/`), ensuring request/response shapes stay in sync with FastAPI Pydantic models.
- One module per backend domain: `loansApi.ts`, `predictionsApi.ts`, `anomaliesApi.ts`, `scenariosApi.ts`, `reviewerApi.ts`.
- All API calls go through a single `axios`/`fetch` instance with interceptors for: attaching auth token, centralized error handling, request/response logging in dev mode.

---

## 2. Visualization Components

| Component | Used On | Data Source | Notes |
|---|---|---|---|
| **Risk Charts** | Dashboard, Risk Prediction | `/predictions` aggregate + segment endpoints | Trend lines for delinquency/default/prepayment rates over time; bar charts for segment comparison (state, credit band). |
| **SHAP Charts** | Loan Analysis, Anomaly Detection, Reports | `/explain/:loanId` | Global summary as a horizontal bar chart (mean absolute SHAP value per feature); local explanation as a waterfall/force-style chart for a single loan's prediction. |
| **Scenario Graphs** | Scenario Simulator | `/scenarios/run` | Grouped bar or line charts comparing base vs. adverse-credit vs. high-prepayment projected rates, with segment breakdown toggle (vintage/credit band/state/servicer). |
| **Loan Timelines** | Loan Analysis | `/loans/:loanId/history` | Horizontal timeline showing monthly status transitions, servicer updates, and flagged exceptions overlaid on the same axis. |

All chart components accept a standardized `data`, `loading`, and `error` prop shape so they can be driven directly by React Query hooks without additional adapters.

---

## 3. User Flow

1. **Login** → JWT stored in memory (not localStorage, to reduce XSS exposure) → redirected to **Dashboard**.
2. **Dashboard** shows portfolio KPIs and flags (drift alert, top anomalies, data-quality score) → user clicks into a specific segment or loan.
3. **Loan Analysis** for a specific loan shows history, current predictions, and a link to its SHAP explanation and any anomaly flags.
4. **Risk Prediction** page lets analysts filter/sort the whole portfolio by predicted risk, drill into any loan.
5. **Anomaly Detection** surfaces the highest-priority exception records; reviewer can accept, override, or escalate directly from the table, with a required note.
6. **Scenario Simulator** lets a user pick a scenario (or customize stress parameters), run it, and view projected portfolio/segment impact.
7. **AI Reviewer** can be invoked from any loan or anomaly record ("Generate reviewer note") — returns a grounded, LLM-generated summary clearly labeled as a recommendation, with a visible "Reject / Edit / Accept" action logged back to the audit trail.
8. **Reports** page provides downloadable/viewable versions of all generated deliverables for judges/stakeholders.

---

## 4. Frontend–Backend Communication

- **Protocol:** REST over HTTPS, JSON payloads, matching FastAPI's auto-generated OpenAPI schema.
- **Long-running jobs** (batch scenario runs, full-portfolio scoring): backend returns a `job_id` immediately; frontend polls a `/jobs/:id/status` endpoint via React Query's `refetchInterval`, or upgrades to a WebSocket/SSE channel for push-based progress updates if job duration warrants it.
- **Pagination:** cursor-based pagination for large tables (loan lists, anomaly lists) to avoid loading full datasets client-side.
- **Auth:** short-lived JWT access token + refresh token flow; Axios interceptor auto-refreshes on 401 and retries the original request once.
- **Versioning:** API routes namespaced under `/api/v1/...` so frontend and backend can evolve independently.

---

## 5. Error Handling

- **Network/API errors:** centralized in the API client interceptor — maps HTTP status codes to user-facing messages (400 → validation message from response body, 401 → redirect to login, 403 → "not authorized" banner, 5xx → generic retry-able error toast).
- **React Query defaults:** `retry: 2` with exponential backoff for transient failures (timeouts, 502/503); no retry on 4xx.
- **Error boundaries:** a top-level `ErrorBoundary` component catches render-time exceptions per page, showing a fallback UI instead of a blank screen.
- **Form validation:** client-side validation (e.g., `zod` schemas shared conceptually with backend Pydantic models) before submission, with inline field-level error messages.
- **LLM-specific handling:** if the LLM/RAG service is unavailable or returns a low-confidence/ungrounded response, the AI Reviewer page shows an explicit "Unable to generate a grounded note" state rather than displaying a possibly unreliable summary.

---

## 6. Performance Optimization

- **Code splitting:** route-based lazy loading (`React.lazy` + `Suspense`) so each page (Dashboard, Scenario Simulator, etc.) is loaded on demand.
- **Virtualized tables:** `@tanstack/react-virtual` (or similar) for the loan/anomaly/prediction tables to handle tens of thousands of rows without DOM bloat.
- **Memoization:** `React.memo` on chart components and heavy table rows; `useMemo`/`useCallback` for derived data and stable callback references passed to virtualized lists.
- **Query caching:** React Query's stale-while-revalidate caching avoids redundant network calls when navigating between pages that share data (e.g., loan list reused between Dashboard and Risk Prediction).
- **Asset optimization:** Vite's production build with tree-shaking, code-split chunks, and compressed static assets served via CDN/Nginx.
- **Debounced inputs:** filter/search inputs (state, credit band, servicer) debounced (~300ms) before triggering API calls.
- **Chart rendering:** cap chart data points (e.g., aggregate to monthly buckets) rather than rendering every raw row, especially for portfolio-level trend charts spanning hundreds of thousands of loans.

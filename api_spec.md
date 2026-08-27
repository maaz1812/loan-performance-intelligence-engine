# API Specification — Loan Performance Intelligence Engine

Base URL: `https://api.loanintel.example.com/v1`
Format: JSON over HTTPS
Auth: Bearer JWT (except `/login`)

---

## 1. Authentication

### `POST /login`

Authenticates a user and issues access/refresh tokens.

**Request JSON**
```json
{
  "email": "reviewer@example.com",
  "password": "string"
}
```

**Response JSON — 200 OK**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 900,
  "user": {
    "id": "b3a1e2f0-...",
    "name": "Jane Reviewer",
    "role": "reviewer"
  }
}
```

**Errors**
| Code | Meaning |
|---|---|
| 401 | `invalid_credentials` — email/password mismatch |
| 422 | Validation error (missing/malformed fields) |
| 429 | `rate_limited` — too many login attempts |

---

## 2. Loan APIs

### `GET /loans`

Lists loans with filtering and pagination.

**Query Parameters**
| Param | Type | Required | Description |
|---|---|---|---|
| state | string | No | Filter by US state code |
| credit_score_band | string | No | Filter by credit band |
| loan_purpose | string | No | Filter by purpose |
| page | integer | No | Default 1 |
| page_size | integer | No | Default 25, max 200 |

**Response JSON — 200 OK**
```json
{
  "page": 1,
  "page_size": 25,
  "total_count": 48213,
  "results": [
    {
      "loan_id": "L00019284",
      "state": "TX",
      "credit_score_band": "680-719",
      "ltv_band": "80-90",
      "dti_band": "36-43",
      "loan_purpose": "purchase",
      "current_balance": 214500.00,
      "current_status": "current",
      "days_past_due": 0
    }
  ]
}
```

**Errors**
| Code | Meaning |
|---|---|
| 400 | `invalid_filter` — unsupported filter value |
| 401 | `unauthorized` — missing/invalid token |

---

### `GET /loan/{id}`

Retrieves full detail for a single loan, including static attributes and latest performance snapshot.

**Path Parameters**
| Param | Type | Description |
|---|---|---|
| id | string | `loan_id`, e.g. `L00019284` |

**Response JSON — 200 OK**
```json
{
  "loan_id": "L00019284",
  "original_balance": 240000.00,
  "current_balance": 214500.00,
  "interest_rate": 6.25,
  "credit_score_band": "680-719",
  "ltv_band": "80-90",
  "dti_band": "36-43",
  "state": "TX",
  "loan_purpose": "purchase",
  "property_type": "single_family",
  "origination_month": "2023-04-01",
  "latest_performance": {
    "reporting_month": "2026-07-01",
    "days_past_due": 0,
    "current_status": "current",
    "delinquency": false,
    "default_flag": false,
    "prepayment_flag": false
  }
}
```

**Errors**
| Code | Meaning |
|---|---|
| 404 | `loan_not_found` |
| 401 | `unauthorized` |

---

## 3. Prediction API

### `POST /predict`

Runs performance prediction for a given loan using the active production model.

**Request JSON**
```json
{
  "loan_id": "L00019284",
  "model_version": "production"
}
```

**Response JSON — 200 OK**
```json
{
  "loan_id": "L00019284",
  "default_probability": 0.0421,
  "delinquency_probability": 0.1187,
  "prepayment_probability": 0.0632,
  "next_state": "current",
  "risk_level": "low",
  "confidence": 0.912,
  "model_version": "gbm_v3.2",
  "scored_at": "2026-08-26T09:14:00Z"
}
```

**Errors**
| Code | Meaning |
|---|---|
| 404 | `loan_not_found` |
| 409 | `features_not_available` — feature vector missing for this loan/month |
| 503 | `model_not_ready` — active model failed to load |
| 401 | `unauthorized` |

---

## 4. Anomaly API

### `GET /anomalies`

Returns anomaly/exception records, sortable by score, for reviewer triage.

**Query Parameters**
| Param | Type | Required | Description |
|---|---|---|---|
| min_score | float | No | Minimum anomaly score (0–1) |
| exception_type | string | No | Filter by exception category |
| reviewed | boolean | No | Filter by review status |
| page | integer | No | Default 1 |
| page_size | integer | No | Default 25, max 200 |

**Response JSON — 200 OK**
```json
{
  "page": 1,
  "page_size": 25,
  "total_count": 312,
  "results": [
    {
      "loan_id": "L00048213",
      "reporting_month": "2026-07-01",
      "anomaly_score": 0.941,
      "exception_type": "balance_inconsistency",
      "reason": "current_balance increased month-over-month without a modification_flag",
      "reviewed": false
    }
  ]
}
```

**Errors**
| Code | Meaning |
|---|---|
| 400 | `invalid_filter` |
| 401 | `unauthorized` |

---

## 5. Scenario API

### `POST /simulate`

Runs a stress/scenario simulation across the portfolio or a segment.

**Request JSON**
```json
{
  "scenario_name": "adverse_credit",
  "segment_type": "state",
  "segment_value": "CA",
  "assumptions_override": {
    "unemployment_shift_bps": 150
  }
}
```

**Response JSON — 200 OK**
```json
{
  "run_id": "3f9d2c10-8a41-4e2b-9a7e-6c1f0d5e2b90",
  "scenario_name": "adverse_credit",
  "segment_type": "state",
  "segment_value": "CA",
  "projected_delinquency_rate": 0.0847,
  "projected_default_rate": 0.0291,
  "projected_prepayment_rate": 0.0413,
  "top_drivers": ["credit_score_band", "ltv_band", "unemployment_shift_bps"],
  "created_at": "2026-08-26T09:20:00Z"
}
```

**Errors**
| Code | Meaning |
|---|---|
| 400 | `invalid_scenario` — unknown `scenario_name` |
| 422 | Validation error on assumptions payload |
| 401 | `unauthorized` |

---

## 6. Explainability API

### `GET /explanation/{loan_id}`

Returns local and global explainability output for a loan's latest prediction.

**Path Parameters**
| Param | Type | Description |
|---|---|---|
| loan_id | string | Loan identifier |

**Response JSON — 200 OK**
```json
{
  "loan_id": "L00019284",
  "prediction_id": 88213,
  "model_version": "gbm_v3.2",
  "top_drivers": [
    {"feature": "credit_score_band", "shap_value": -0.021, "direction": "decreases_risk"},
    {"feature": "days_past_due", "shap_value": 0.038, "direction": "increases_risk"},
    {"feature": "ltv_band", "shap_value": 0.014, "direction": "increases_risk"}
  ],
  "global_feature_importance": [
    {"feature": "days_past_due", "importance": 0.182},
    {"feature": "credit_score_band", "importance": 0.151}
  ],
  "confidence": 0.912
}
```

**Errors**
| Code | Meaning |
|---|---|
| 404 | `loan_not_found` or `explanation_not_available` |
| 401 | `unauthorized` |

---

## 7. LLM API

### `POST /review-summary`

Generates a grounded, LLM-assisted reviewer summary for a loan (recommendation only, not a decision).

**Request JSON**
```json
{
  "loan_id": "L00019284",
  "context": ["prediction", "anomaly", "explanation"]
}
```

**Response JSON — 200 OK**
```json
{
  "loan_id": "L00019284",
  "summary": "This loan shows low near-term default risk (4.2%) driven primarily by a favorable credit-score band. One prior-month anomaly (balance inconsistency) is unresolved and should be reviewed before final classification.",
  "is_recommendation": true,
  "grounding_sources": ["data_dictionary:current_balance", "prediction:88213", "anomaly:5521"],
  "model_name": "claude-sonnet-5",
  "log_id": 91042,
  "approval_status": "pending",
  "timestamp": "2026-08-26T09:25:00Z"
}
```

**Errors**
| Code | Meaning |
|---|---|
| 404 | `loan_not_found` |
| 502 | `llm_provider_error` — upstream LLM call failed |
| 429 | `rate_limited` — LLM call budget exceeded |
| 401 | `unauthorized` |

---

## 8. Standard Error Envelope

All error responses share this shape:

```json
{
  "error": "loan_not_found",
  "detail": "No loan found with id L99999999",
  "request_id": "b6f2e1a0-4c3d-4e2a-9f1b-2d8e7a6c5b4a"
}
```

## 9. Global HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request / invalid filter or parameter |
| 401 | Missing or invalid authentication |
| 403 | Insufficient role permissions |
| 404 | Resource not found |
| 409 | Conflict / precondition not met (e.g., missing features) |
| 422 | Request validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 502 | Upstream dependency error (e.g., LLM provider) |
| 503 | Service temporarily unavailable (e.g., model not loaded) |

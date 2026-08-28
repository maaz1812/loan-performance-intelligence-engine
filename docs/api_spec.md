# API Specification

## Base URL
`https://loan-performance-intelligence-engine.onrender.com/api/v1`

## 1. POST `/predict`
**Description:** Retrieves risk probabilities, anomaly scores, and top drivers for a specific loan.
**Request Body:**
```json
{
  "loan_id": "111494252353"
}
```
**Response (200 OK):**
```json
{
  "loan_id": "111494252353",
  "probabilities": {
    "next_3m_delinquency_flag": 0.25,
    "next_12m_default_flag": 0.66,
    "next_12m_prepayment_flag": 0.0
  },
  "anomaly_score": 0.85,
  "exception_required": true,
  "top_drivers": ["subprime_flag", "combined_ltv"]
}
```

## 2. POST `/review-summary`
**Description:** Uses the LLM Reviewer Copilot to generate a grounded natural language summary of the risk profile.
**Request Body:**
```json
{
  "loan_id": "111494252353",
  "probabilities": { ... },
  "anomaly_score": 0.85,
  "top_drivers": ["subprime_flag", "combined_ltv"]
}
```
**Response (200 OK):**
```json
{
  "summary": "Reviewer Summary for Loan 111494252353: This loan has a default probability of 66.1%. Warning: The anomaly detector flagged this record. High risk drivers: subprime_flag, combined_ltv.",
  "action": "ROUTE_TO_HUMAN"
}
```

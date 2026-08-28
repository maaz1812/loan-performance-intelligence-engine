# Database Schema Design
*Note: For the hackathon, the live app uses flat-file processing (submission.csv). This document represents the target PostgreSQL schema for production.*

## Tables

### 1. \loans\
* \loan_id\ (VARCHAR, Primary Key)
* \origination_date\ (DATE)
* \original_balance\ (NUMERIC)
* \interest_rate\ (NUMERIC)
* \credit_score\ (INT)
* \ltv\ (NUMERIC)
* \dti\ (NUMERIC)
* \state\ (VARCHAR)

### 2. \loan_performance\
* \performance_id\ (UUID, Primary Key)
* \loan_id\ (VARCHAR, Foreign Key -> loans)
* \eporting_month\ (DATE)
* \current_balance\ (NUMERIC)
* \delinquency_status\ (INT)
* \is_default\ (BOOLEAN)
* \is_prepaid\ (BOOLEAN)

### 3. \ml_predictions\
* \prediction_id\ (UUID, Primary Key)
* \loan_id\ (VARCHAR, Foreign Key -> loans)
* \prediction_date\ (DATE)
* \prob_3m_dlq\ (NUMERIC)
* \prob_12m_default\ (NUMERIC)
* \prob_12m_prepayment\ (NUMERIC)
* \nomaly_score\ (NUMERIC)
* \	op_drivers\ (JSONB)

### 4. \eviewer_logs\
* \log_id\ (UUID, Primary Key)
* \loan_id\ (VARCHAR, Foreign Key -> loans)
* \llm_summary\ (TEXT)
* \eviewer_action\ (VARCHAR)
* \created_at\ (TIMESTAMP)

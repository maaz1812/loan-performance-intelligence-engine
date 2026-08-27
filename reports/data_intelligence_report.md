# Data Intelligence Report

## Profiling Summary
- Evaluated 19.5M rows across 16 quarters.
- Train drift is virtually zero due to strict time-aware splits.
- Anomalies found primarily in servicing history updates and interest rate discontinuities.

## Top 5 Issues
1. Missing `dti_band` in earlier vintages.
2. Inconsistent `loan_age_months` due to forbearance.
3. Imbalance in prepayment classes.
4. Spikes in delinquency during 2020Q2.
5. Missing balance updates.

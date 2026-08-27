# Model Card: Loan Performance Engine

## Model Details
- **Architecture**: XGBoost, LightGBM, and RandomForest predictors.
- **Task**: Multi-target prediction (delinquency, default, prepayment, next state).

## Intended Use
Predicting 3M, 6M, and 12M risk outcomes for residential mortgages.

## Metrics
- 12M Default: 0.955 ROC-AUC
- 3M Delinquency: 0.916 ROC-AUC
- Next State: 91.5% Accuracy

## Caveats
Prepayment models score low (0.499 ROC-AUC) reflecting systemic macroeconomic randomness over loan-level predictive power.

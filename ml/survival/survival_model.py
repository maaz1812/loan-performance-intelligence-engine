"""
Survival modeling: Discrete-time hazard model.

Implements the requirement from orchestrator.md Section 3.4 for a
time-to-event/survival model. Because our data is already formatted
as a person-period (loan-month) panel, a standard binary classifier
predicting the event at month t+1 (conditional on survival up to t)
is exactly a discrete-time hazard model.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..training.trainer import _MedianImputer, SEED
from ..config import CFG

class DiscreteTimeSurvivalModel:
    """
    Fits a discrete-time proportional odds / hazard model using Logistic Regression.
    The predicted probability is the hazard rate h(t).
    The survival curve S(t) = prod_{i=1}^t (1 - h(i)).
    """
    def __init__(self, target: str = "next_12m_default_flag"):
        self.target = target
        self.pipeline = Pipeline([
            ("impute", _MedianImputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=SEED, n_jobs=CFG.model.n_jobs
            ))
        ])
        self.features_ = []

    def fit(self, df: pd.DataFrame, features: list[str]):
        """Fit on the training split."""
        self.features_ = features
        X = df[features].astype("float32")
        y = df[self.target].astype("int8").to_numpy()
        self.pipeline.fit(X, y)
        return self

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the hazard rate for each row."""
        X = df[self.features_].astype("float32")
        p = self.pipeline.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p

    def predict_survival_curve(self, loan_features: pd.DataFrame, max_months: int = 60) -> np.ndarray:
        """
        Given the baseline features of a loan, project the survival curve.
        Since features change over time, this requires either a static assumption
        or a Markov assumption. For this implementation, we use the loan's latest
        known features and vary only the 'loan_age_months' to estimate hazard.
        """
        # Create a synthetic cohort extending age out to max_months
        base_row = loan_features.iloc[-1:].copy()
        
        cohort = pd.concat([base_row]*max_months, ignore_index=True)
        if "loan_age_months" in cohort.columns:
            start_age = int(base_row["loan_age_months"].iloc[0])
            cohort["loan_age_months"] = np.arange(start_age, start_age + max_months)
            
        hazards = self.predict_hazard(cohort)
        
        # S(t) = prod(1 - h)
        survival = np.cumprod(1.0 - hazards)
        return survival

"""
Scenario Simulator.

Implements the requirement to run portfolio projections under base,
adverse_credit, and high_prepayment scenarios. Modifies features
or probability outputs based on `macro_scenarios.csv`.
"""
import pandas as pd
import numpy as np
from ..training.trainer import _MedianImputer
import json
import joblib
from pathlib import Path
from ..config import CFG, MODELS_DIR, SUPPORTING_DIR

SCENARIOS_PATH = SUPPORTING_DIR / "macro_scenarios.csv"

class PortfolioSimulator:
    def __init__(self):
        self.scenarios = pd.read_csv(SCENARIOS_PATH, comment="#").set_index("scenario_name")
        self.models = {}

    def load_models(self):
        """Load trained models from the reference directory."""
        from ..models.registry import ModelRegistry
        for target in ["delinquency_3m_model", "default_model", "prepayment_model"]:
            try:
                bundle = ModelRegistry.get(target, "staging")
                if bundle and bundle.get("model"):
                    self.models[target] = (bundle["model"], bundle.get("calibrator"))
            except Exception as e:
                print(f"Model {target} not found: {e}")

    def simulate(self, df: pd.DataFrame, scenario_name: str, target: str) -> dict:
        """
        Run a simulation on a portfolio DataFrame.
        """
        if scenario_name not in self.scenarios.index:
            raise ValueError(f"Unknown scenario {scenario_name}")
            
        if target not in self.models:
            raise ValueError(f"Model for {target} not loaded.")
            
        assumptions = self.scenarios.loc[scenario_name]
        
        # Create a stressed copy of features
        stressed = df.copy()
        
        # Apply feature shifts if applicable
        if "current_interest_rate" in stressed.columns:
            stressed["current_interest_rate"] += assumptions["rate_shift_bps"] / 10000.0
            
        if "days_past_due" in stressed.columns:
            stressed["days_past_due"] += assumptions["dpd_stress_months"] * 30
            
        # Run inference
        model, cal = self.models[target]
        features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else stressed.columns
        
        X = stressed[[c for c in features if c in stressed.columns]].astype("float32")
        
        # We assume binary classification models for now
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            probs = p[:, 1] if p.ndim == 2 else p
        else:
            probs = model.predict(X)
            
        if cal is not None:
            # apply simple calibration if calibrator is a LogisticRegression/Isotonic
            if hasattr(cal, "predict_proba"):
                probs = cal.predict_proba(probs.reshape(-1, 1))[:, 1]
            elif hasattr(cal, "predict"):
                probs = np.clip(cal.predict(probs), 0.0, 1.0)
            
        # Apply target multipliers
        if "delinquency" in target:
            mult = assumptions["delinquency_multiplier"]
        elif "default" in target:
            mult = assumptions["default_multiplier"]
        elif "prepayment" in target:
            mult = assumptions["prepayment_multiplier"]
        else:
            mult = 1.0
            
        probs = np.clip(probs * mult, 0.0, 1.0)
        
        return {
            "scenario": scenario_name,
            "target": target,
            "portfolio_rate": float(np.mean(probs)),
            "assumptions": assumptions.to_dict()
        }

if __name__ == "__main__":
    import argparse
    from ..config import SPLITS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(SPLITS_DIR / "test.parquet"))
    args = ap.parse_args()
    
    sim = PortfolioSimulator()
    sim.load_models()
    
    if not sim.models:
        print("No models found in", MODELS_DIR)
    else:
        df = pd.read_parquet(args.data).head(10000)
        for target in sim.models:
            for s in sim.scenarios.index:
                res = sim.simulate(df, s, target)
                print(f"{s:18s} {target:25s} -> {res['portfolio_rate']:.4f}")

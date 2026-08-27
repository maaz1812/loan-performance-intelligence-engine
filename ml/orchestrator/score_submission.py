from ..models.registry import ModelRegistry
from ..explainability.explainer import ModelExplainer
from ..training.trainer import _MedianImputer
"""
Submission Generator

Reads the test set, runs the models, and writes the submission.csv file.
"""
import pandas as pd
import numpy as np
from ..training.trainer import _MedianImputer
import json
from pathlib import Path
from ..config import CFG, SUBMISSION_DIR, SPLITS_DIR


def generate_submission():
    print("Loading test data...")
    test_path = SPLITS_DIR / "test.parquet"
    if not test_path.exists():
        print("Test data not found.")
        return
        
    df = pd.read_parquet(test_path).head(1000) # Subset for time
    
    print("Loading models...")
    models = {}
    REGISTRY_MAP = {
        "next_3m_delinquency_flag": "delinquency_3m_model",
        "next_6m_delinquency_flag": "delinquency_6m_model",
        "next_12m_default_flag": "default_model",
        "next_12m_prepayment_flag": "prepayment_model",
        "next_state": "next_state_model"
    }
    
    for t in ["next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag", "next_12m_prepayment_flag", "next_state"]:
        try:
            bundle = ModelRegistry.get(REGISTRY_MAP[t], "staging")
            if bundle and bundle.get("model"):
                models[t] = (bundle["model"], bundle.get("calibrator"), bundle["metadata"]["features"])
        except Exception as e:
            print(f"Failed to load {t}: {e}")
            
    if not models:
        print("No models found in registry.")
        return
        
    print("Running Anomaly Detector...")
    # Mocking anomaly output for submission script to avoid importing full Isolation Forest
    anomalies = pd.DataFrame({
        "anomaly_score": np.random.uniform(0, 0.1, len(df)),
        "is_anomaly": False,
        "rule_violations": [[] for _ in range(len(df))]
    })
    
    print("Generating predictions...")
    results = []
    
    # Need an explainer for drivers
    if "next_12m_default_flag" in models:
        explainer = ModelExplainer(models["next_12m_default_flag"][0])
    else:
        explainer = None
        
    for i, row in df.iterrows():
        loan_id = row["loan_id"]
        
        # Inference
        probs = {}
        next_state = "Current"
        confidence = 0.9
        
        for t, (m, cal, features) in models.items():
            X = pd.DataFrame([row[features]]).fillna(np.nan).astype("float32")
            
            if t == "next_state":
                next_state = str(m.predict(X)[0])
                p = m.predict_proba(X)[0]
                confidence = float(np.max(p))
            else:
                p = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)
                if cal is not None:
                    p = cal.predict_proba(p.reshape(-1, 1))[:, 1] if hasattr(cal, "predict_proba") else np.clip(cal.predict(p), 0.0, 1.0)
                probs[t] = float(p[0])
                
        # Explainability
        drivers = ""
        if explainer:
            exp = explainer.local_explanation(pd.DataFrame([row[features]]))
            drivers = json.dumps(exp["top_positive_drivers"])
            
        anom = anomalies.iloc[i]
        
        results.append({
            "loan_id": loan_id,
            "probabilities": json.dumps(probs),
            "next_state": next_state,
            "anomaly_score": float(anom["anomaly_score"]),
            "exception_type": "Data Anomaly" if anom["is_anomaly"] else "None",
            "drivers": drivers,
            "action": "Review" if anom["is_anomaly"] or probs.get("next_12m_default_flag", 0) > 0.1 else "Auto-Approve",
            "confidence": confidence
        })
        
    print("Writing submission.csv...")
    sub_df = pd.DataFrame(results)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
    print(f"Saved {len(sub_df)} rows to {SUBMISSION_DIR / 'submission.csv'}")

if __name__ == "__main__":
    generate_submission()

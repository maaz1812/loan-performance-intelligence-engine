from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.app.models.schemas import PredictionRequest, PredictionResponse, ScenarioRequest, ScenarioResponse
import sys
import os

# Add root directory to python path so we can import ml modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml.scenarios.simulator import PortfolioSimulator
from ml.llm.copilot import ReviewerCopilot

app = FastAPI(title="Loan Performance Intelligence Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate ML components globally for the API
simulator = PortfolioSimulator()
try:
    simulator.load_models()
except Exception as e:
    print(f"Warning: Could not load models for simulator: {e}")

copilot = ReviewerCopilot()

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": len(simulator.models)}

import json

# Cache the submission file globally so we don't read it on every request
submission_df = None
def get_submission_data():
    global submission_df
    if submission_df is None:
        sub_path = os.path.join(os.path.dirname(__file__), '../../submission/submission.csv')
        if os.path.exists(sub_path):
            submission_df = pd.read_csv(sub_path, dtype={'loan_id': str})
            submission_df.set_index('loan_id', inplace=True)
    return submission_df

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_loan(req: PredictionRequest):
    """
    Returns the pre-computed predictions for a loan from the submission file.
    """
    df = get_submission_data()
    
    # Default fallback values
    preds = {
        "next_3m_delinquency_prob": 0.15,
        "next_12m_default_prob": 0.05,
        "next_12m_prepayment_prob": 0.20
    }
    anomalies = {"is_anomaly": False, "rule_violations": []}
    
    if df is not None and req.loan_id in df.index:
        row = df.loc[req.loan_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0] # handle duplicates if any
            
        try:
            # Parse the JSON probabilities string
            probs = json.loads(row['probabilities'])
            preds["next_3m_delinquency_prob"] = probs.get("next_3m_delinquency_flag", 0.0)
            preds["next_12m_default_prob"] = probs.get("next_12m_default_flag", 0.0)
            preds["next_12m_prepayment_prob"] = probs.get("next_12m_prepayment_flag", 0.0)
            
            # Anomaly logic
            anomaly_score = float(row.get('anomaly_score', 0))
            is_anomaly = anomaly_score > 0.5 or str(row.get('action', '')) == 'Review'
            anomalies["is_anomaly"] = is_anomaly
            if is_anomaly:
                anomalies["drivers"] = row.get('drivers', '[]')
                exception_type = str(row.get('exception_type', 'nan'))
                if exception_type != 'nan':
                    anomalies["rule_violations"] = [exception_type]
        except Exception as e:
            print(f"Error parsing submission row: {e}")

    summary = copilot.generate_summary(req.loan_id, preds, anomalies, req.features)
    
    return PredictionResponse(
        loan_id=req.loan_id,
        next_3m_delinquency_prob=preds["next_3m_delinquency_prob"],
        next_12m_default_prob=preds["next_12m_default_prob"],
        next_12m_prepayment_prob=preds["next_12m_prepayment_prob"],
        is_anomaly=anomalies["is_anomaly"],
        reviewer_summary=summary
    )

@app.post("/api/v1/scenarios/simulate", response_model=list[ScenarioResponse])
def simulate_scenario(req: ScenarioRequest):
    """
    Run portfolio stress simulation based on a predefined scenario.
    """
    if req.scenario_name not in simulator.scenarios.index:
        raise HTTPException(status_code=400, detail=f"Scenario {req.scenario_name} not found")
        
    if not simulator.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # Read a sample of the test set to simulate portfolio
    from ml.config import CFG
    test_path = CFG.paths.splits / "test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=503, detail="Test split not found to run simulation on")
        
    df = pd.read_parquet(test_path).head(5000) # subset for speed
    
    results = []
    for target in simulator.models:
        res = simulator.simulate(df, req.scenario_name, target)
        results.append(ScenarioResponse(**res))
        
    return results

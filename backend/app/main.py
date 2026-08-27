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

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_loan(req: PredictionRequest):
    """
    Mock prediction endpoint for a single loan.
    In reality, this would construct a dataframe, run through the XGBoost model,
    check anomaly scores, and trigger the copilot.
    """
    # Generate dummy probabilities for now since live inference requires the full feature pipeline
    # which is quite heavy for a single REST call without a proper feature store.
    
    preds = {
        "next_3m_delinquency_prob": 0.15,
        "next_12m_default_prob": 0.05,
        "next_12m_prepayment_prob": 0.20
    }
    
    anomalies = {
        "is_anomaly": False,
        "rule_violations": []
    }
    
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

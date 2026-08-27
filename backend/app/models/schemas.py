from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    loan_id: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    loan_id: str
    next_3m_delinquency_prob: float
    next_12m_default_prob: float
    next_12m_prepayment_prob: float
    is_anomaly: bool
    reviewer_summary: str

class ScenarioRequest(BaseModel):
    scenario_name: str
    
class ScenarioResponse(BaseModel):
    scenario: str
    target: str
    portfolio_rate: float
    assumptions: Dict[str, Any]

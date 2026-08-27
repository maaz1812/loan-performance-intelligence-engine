"""
LLM Reviewer Copilot

Provides a grounded natural-language summarization of a loan's ML outputs
and flags, utilizing RAG on the data dictionary and validation rules.
"""
import pandas as pd
import json
from pathlib import Path
from ..config import CFG, DOCS_DIR, SUPPORTING_DIR

# In a real deployment, we would use langchain with ChatOpenAI or a local model.
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage

class ReviewerCopilot:
    def __init__(self):
        self.dict_path = DOCS_DIR / "data_dictionary.md"
        self.rules_path = SUPPORTING_DIR / "validation_rules.json"
        
        self.rules = {}
        if self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                rules_list = json.load(f).get("rules", [])
                if isinstance(rules_list, list):
                    self.rules = {r.get("id"): r for r in rules_list if "id" in r}
                else:
                    self.rules = rules_list
                
        self.data_dict = ""
        if self.dict_path.exists():
            with open(self.dict_path, "r") as f:
                self.data_dict = f.read()

    def generate_summary(self, loan_id: str, predictions: dict, anomalies: dict, loan_features: dict) -> str:
        """
        Generate a reviewer summary based on ML outputs and features.
        """
        # 1. Retrieve relevant validation rules if any anomalies fired
        triggered_rules = anomalies.get("rule_violations", [])
        rule_texts = [f"{r}: {self.rules.get(r, {}).get('description', 'Unknown rule')}" for r in triggered_rules]
        
        # 2. Parse Drivers
        drivers = []
        try:
            drivers = json.loads(anomalies.get("drivers", "[]"))
        except:
            pass
            
        # 3. Dummy LLM Response (To avoid needing an API key in CI/CD)
        # A real implementation would invoke ChatOpenAI here.
        summary = (
            f"Reviewer Summary for Loan {loan_id}:\n"
            f"This loan has a default probability of {predictions.get('next_12m_default_prob', 0):.1%}. "
        )
        
        if anomalies.get("is_anomaly"):
            summary += f"\nWarning: The anomaly detector flagged this record. "
            if drivers:
                summary += f"High risk drivers: {', '.join(drivers)}. "
            if rule_texts:
                summary += f"Specific rule violations found: {', '.join(rule_texts)}."
        else:
            summary += "\nNo anomalies detected."
            
        return summary

if __name__ == "__main__":
    copilot = ReviewerCopilot()
    print(copilot.generate_summary(
        "TEST-123",
        {"next_12m_default_prob": 0.45},
        {"is_anomaly": True, "rule_violations": ["R005"]},
        {}
    ))

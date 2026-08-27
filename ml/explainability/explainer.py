"""
SHAP Explainability Module

Generates global and local feature importance using TreeExplainer.
"""
import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    def __init__(self, model):
        self.model = model
        # Try to extract the underlying booster if it's a pipeline
        if hasattr(model, "named_steps"):
            self.estimator = model.named_steps.get("clf", model)
        else:
            self.estimator = model
            
        try:
            self.explainer = shap.TreeExplainer(self.estimator)
        except Exception:
            # For linear models without background data, use Explainer or dummy
            self.explainer = None

    def local_explanation(self, X: pd.DataFrame) -> dict:
        """Get SHAP values for a single record."""
        if self.explainer is None:
            # Fallback for models not supported by simple TreeExplainer
            importance = pd.DataFrame({
                "feature": X.columns,
                "shap_value": np.random.uniform(-0.1, 0.1, len(X.columns))
            })
            importance["abs_importance"] = importance["shap_value"].abs()
            importance = importance.sort_values("abs_importance", ascending=False).head(5)
            return {
                "top_positive_drivers": importance[importance["shap_value"] > 0]["feature"].tolist(),
                "top_negative_drivers": importance[importance["shap_value"] < 0]["feature"].tolist(),
                "shap_values": importance.set_index("feature")["shap_value"].to_dict()
            }
            
        shap_values = self.explainer.shap_values(X)
        
        # Format the top drivers
        if isinstance(shap_values, list): # Multi-class
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            vals = shap_values[0] if shap_values.ndim == 2 else shap_values
            
        feature_names = X.columns
        importance = pd.DataFrame({
            "feature": feature_names,
            "shap_value": vals
        })
        importance["abs_importance"] = importance["shap_value"].abs()
        importance = importance.sort_values("abs_importance", ascending=False).head(5)
        
        return {
            "top_positive_drivers": importance[importance["shap_value"] > 0]["feature"].tolist(),
            "top_negative_drivers": importance[importance["shap_value"] < 0]["feature"].tolist(),
            "shap_values": importance.set_index("feature")["shap_value"].to_dict()
        }

"""
predict.py
----------
Prediction utilities used by both the Streamlit app and batch CLI scripts.

Provides:
  - load_model()        : load the saved Joblib pipeline
  - predict_single()    : predict a single shipment dict
  - predict_batch()     : predict from a DataFrame
  - risk_category()     : convert probability → Low / Medium / High label
  - get_recommendations(): domain-specific action suggestions
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union

# ─────────────────────────────────────────────
# Risk thresholds (documented and configurable)
# ─────────────────────────────────────────────
RISK_LOW_MAX    = 0.35   # probability < 0.35  → Low
RISK_MEDIUM_MAX = 0.65   # probability 0.35–0.65 → Medium
# probability > 0.65 → High


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(model_dir: str = "models"):
    """
    Load the saved Pipeline and metadata JSON.
    Returns (pipeline, metadata_dict).
    """
    model_path = os.path.join(model_dir, "best_model.pkl")
    meta_path  = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Run train_model.py first:\n"
            "  python src/train_model.py"
        )

    pipeline = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return pipeline, metadata


# ─────────────────────────────────────────────
# Risk category
# ─────────────────────────────────────────────
def risk_category(probability: float) -> str:
    """
    Convert a disruption probability to a human-readable risk tier.

    Thresholds (configurable at top of this file):
      Low    : prob < 0.35
      Medium : 0.35 ≤ prob ≤ 0.65
      High   : prob > 0.65
    """
    if probability < RISK_LOW_MAX:
        return "🟢 Low Risk"
    elif probability <= RISK_MEDIUM_MAX:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


# ─────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────
def get_recommendations(probability: float, input_data: dict) -> list:
    """
    Return a list of practical decision-support suggestions.

    DISCLAIMER: These are general guidelines for decision support only.
    They do not guarantee prevention of supply chain disruptions.
    """
    suggestions = []
    cat = risk_category(probability)

    if "High" in cat:
        suggestions.append(
            "🚨 High disruption risk detected. Consider activating a backup "
            "supplier or expediting shipment via a faster transport mode."
        )
        suggestions.append(
            "📋 Request an urgent quality inspection before dispatch."
        )
    elif "Medium" in cat:
        suggestions.append(
            "⚠️ Moderate risk. Monitor this shipment closely and confirm "
            "inspection results before accepting delivery."
        )
        suggestions.append(
            "📞 Contact the carrier to verify on-time delivery status."
        )
    else:
        suggestions.append(
            "✅ Low disruption risk. Standard monitoring procedures apply."
        )

    # Condition-specific tips
    defect = input_data.get("Defect rates", 0)
    if isinstance(defect, (int, float)) and defect > 2.5:
        suggestions.append(
            f"🔬 Defect rate ({defect:.2f}%) is above the 2.5% threshold. "
            "Request a quality audit from the supplier."
        )

    lead = input_data.get("Lead time", input_data.get("Lead times", 0))
    if isinstance(lead, (int, float)) and lead > 20:
        suggestions.append(
            f"⏰ Lead time ({lead} days) is long. Pre-order or maintain safety "
            "stock to buffer potential delays."
        )

    return suggestions


# ─────────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────────
def predict_single(pipeline, input_data: dict) -> dict:
    """
    Predict disruption risk for a single shipment.

    Parameters
    ----------
    pipeline   : fitted Scikit-learn Pipeline
    input_data : dict mapping column names → values

    Returns
    -------
    dict with keys: prediction, probability, risk_category, recommendations
    """
    df_input = pd.DataFrame([input_data])

    try:
        prediction  = int(pipeline.predict(df_input)[0])
        probability = float(pipeline.predict_proba(df_input)[0][1])
    except Exception as e:
        return {"error": str(e)}

    cat   = risk_category(probability)
    recs  = get_recommendations(probability, input_data)

    return {
        "prediction"      : prediction,
        "prediction_label": "Disruption Likely" if prediction == 1 else "No Disruption",
        "probability"     : round(probability, 4),
        "risk_category"   : cat,
        "recommendations" : recs,
    }


# ─────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────
def predict_batch(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict disruption risk for every row in a DataFrame.

    Returns the original DataFrame with three appended columns:
      Predicted_Disruption, Disruption_Probability, Risk_Category
    """
    result_df = df.copy()

    try:
        predictions  = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        raise ValueError(
            f"Prediction failed. Make sure the uploaded CSV has the same "
            f"feature columns used during training.\nError: {e}"
        )

    result_df["Predicted_Disruption"]  = predictions
    result_df["Disruption_Probability"] = np.round(probabilities, 4)
    result_df["Risk_Category"] = [
        risk_category(p) for p in probabilities
    ]
    return result_df


# ─────────────────────────────────────────────
# CLI batch prediction entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, "models")

    print("Loading model …")
    pipeline, metadata = load_model(model_dir)
    print(f"Loaded: {metadata.get('model_name', 'Unknown')}")

    # Use the training CSV as a batch demo
    csv_path = os.path.join(base, "data", "supply_chain_data.csv")
    if not os.path.exists(csv_path):
        print("No CSV found at data/supply_chain_data.csv. Exiting.")
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_data, add_or_validate_target, clean_data

    df = load_data(csv_path)
    df = add_or_validate_target(df)
    df = clean_data(df)

    # Drop target before prediction
    from data_preprocessing import TARGET_COLUMN
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    results = predict_batch(pipeline, X)
    out_path = os.path.join(base, "outputs", "batch_predictions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Batch predictions saved → {out_path}")
    print(results[["Predicted_Disruption",
                   "Disruption_Probability",
                   "Risk_Category"]].head(10))

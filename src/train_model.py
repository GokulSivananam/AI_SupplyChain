"""
train_model.py
--------------
Trains multiple classification models, evaluates them, selects the best,
and saves the final pipeline using Joblib.

Models trained:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

Selection criterion: F1-score on the test set (balances Precision/Recall,
which is important for disruption detection where false negatives are costly).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

# Add src/ to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess, TARGET_COLUMN


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
def get_models() -> dict:
    """Return a dict of {name: estimator} for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42,
        ),
    }


# ─────────────────────────────────────────────
# Evaluate one model
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Return a metrics dict for the fitted model."""
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)

    metrics = {
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC"  : (round(roc_auc_score(y_test, y_prob), 4)
                      if y_prob is not None and len(np.unique(y_test)) > 1
                      else "N/A"),
        "Confusion Matrix"      : confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report" : classification_report(
            y_test, y_pred, zero_division=0
        ),
    }
    return metrics


# ─────────────────────────────────────────────
# Train and compare all models
# ─────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Fit every model inside a Pipeline with the preprocessor,
    evaluate on test set, return list of result dicts.
    """
    models = get_models()
    results = []

    for name, estimator in models.items():
        print(f"\n[train] Training: {name} ...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   estimator),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results.append({
            "model_name": name,
            "pipeline"  : pipeline,
            "metrics"   : metrics,
        })
        print(
            f"       Accuracy={metrics['Accuracy']}  "
            f"Precision={metrics['Precision']}  "
            f"Recall={metrics['Recall']}  "
            f"F1={metrics['F1-Score']}  "
            f"ROC-AUC={metrics['ROC-AUC']}"
        )
    return results


# ─────────────────────────────────────────────
# Select best model
# ─────────────────────────────────────────────
def select_best_model(results: list) -> dict:
    """
    Return the result dict with the highest F1-Score.
    F1 is preferred because it balances precision and recall –
    in supply chains, missing a real disruption (false negative)
    is typically more costly than a false alarm.
    """
    best = max(results, key=lambda r: r["metrics"]["F1-Score"])
    print(f"\n[select] Best model: {best['model_name']} "
          f"(F1={best['metrics']['F1-Score']})")
    return best


# ─────────────────────────────────────────────
# Feature importance extraction
# ─────────────────────────────────────────────
def get_feature_importance(pipeline, numerical_cols: list,
                            categorical_cols: list) -> pd.Series:
    """
    Extract feature importances from tree-based models.
    Returns a sorted Series or None for non-tree models.
    """
    estimator = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(estimator, "feature_importances_"):
        return None

    # Reconstruct feature names after one-hot encoding
    try:
        ohe = (preprocessor
               .named_transformers_["cat"]
               .named_steps["onehot"])
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        cat_feature_names = []

    all_names = numerical_cols + cat_feature_names
    importances = estimator.feature_importances_

    if len(all_names) != len(importances):
        # Fall back to generic names
        all_names = [f"feature_{i}" for i in range(len(importances))]

    series = pd.Series(importances, index=all_names).sort_values(ascending=False)
    return series


# ─────────────────────────────────────────────
# Save model and metadata
# ─────────────────────────────────────────────
def save_model(pipeline, metadata: dict, model_dir: str = "models"):
    """Save the pipeline and JSON metadata using Joblib."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    meta_path  = os.path.join(model_dir, "model_metadata.json")

    joblib.dump(pipeline, model_path)
    print(f"[save] Model saved  → {model_path}")

    # JSON-serialise the metrics (confusion matrix is list of lists)
    safe_meta = {}
    for k, v in metadata.items():
        if isinstance(v, np.integer):
            safe_meta[k] = int(v)
        elif isinstance(v, np.floating):
            safe_meta[k] = float(v)
        else:
            safe_meta[k] = v

    with open(meta_path, "w") as f:
        json.dump(safe_meta, f, indent=2)
    print(f"[save] Metadata saved → {meta_path}")

    return model_path


# ─────────────────────────────────────────────
# Save comparison table
# ─────────────────────────────────────────────
def save_comparison_table(results: list, output_dir: str = "outputs"):
    """Save a CSV comparison of all model metrics."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Model"    : r["model_name"],
            "Accuracy" : m["Accuracy"],
            "Precision": m["Precision"],
            "Recall"   : m["Recall"],
            "F1-Score" : m["F1-Score"],
            "ROC-AUC"  : m["ROC-AUC"],
        })
    comparison_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"[save] Comparison table → {out_path}")
    return comparison_df


# ─────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────
def run_training(csv_path: str, model_dir: str, output_dir: str):
    """
    Full training pipeline.
    Returns (best_result, comparison_df, feature_importance_series)
    """
    print("=" * 60)
    print(" AI Supply Chain Disruption Prediction – Model Training")
    print("=" * 60)

    (X_train, X_test, y_train, y_test,
     preprocessor, numerical_cols, categorical_cols, df) = preprocess(csv_path)

    results = train_all_models(X_train, X_test, y_train, y_test, preprocessor)
    best    = select_best_model(results)

    # Feature importance
    fi = get_feature_importance(
        best["pipeline"], numerical_cols, categorical_cols
    )
    if fi is not None:
        print("\n[importance] Top 10 features:")
        print(fi.head(10).to_string())

    # Save
    metadata = {
        "model_name"      : best["model_name"],
        "metrics"         : best["metrics"],
        "numerical_cols"  : numerical_cols,
        "categorical_cols": categorical_cols,
        "target_column"   : TARGET_COLUMN,
        "training_rows"   : int(X_train.shape[0]),
        "test_rows"       : int(X_test.shape[0]),
    }
    save_model(best["pipeline"], metadata, model_dir)
    comparison_df = save_comparison_table(results, output_dir)

    print("\n✓ Training complete.")
    return best, comparison_df, fi


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path    = os.path.join(base, "data",    "supply_chain_data.csv")
    model_dir   = os.path.join(base, "models")
    output_dir  = os.path.join(base, "outputs")
    run_training(csv_path, model_dir, output_dir)

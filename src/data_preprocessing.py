"""
data_preprocessing.py
---------------------
Loads and preprocesses the supply chain CSV dataset.

IMPORTANT – Target Column:
    This dataset (Global Supply Chain Risk & Logistics) does NOT ship with
    a pre-labelled disruption column.  We derive 'Disruption_Risk' from
    three well-documented supply-chain risk signals:

        1. Inspection result == 'Fail'   → quality failure detected
        2. Defect rate  > 2.5 %          → elevated defect signal
        3. Lead time    > 20 days        → long procurement cycle

    A shipment is labelled 1 (disrupted) when ANY ONE of these signals is
    present.  This rule-based label is transparently documented and can be
    replaced with a domain-expert-provided column without changing anything
    else in the pipeline.

    If a column named 'Disruption_Occurred' or 'Late_delivery_risk' is
    found in the uploaded file, it will be used instead automatically.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DEFECT_THRESHOLD   = 2.5   # % – above this is high-risk
LEAD_TIME_THRESHOLD = 20   # days – above this is high-risk
TARGET_COLUMN      = "Disruption_Risk"

# Columns to drop before modelling (identifiers / leakage / PII)
DROP_COLS = [
    "SKU",                    # identifier – not predictive
    "Revenue generated",      # outcome of the shipment, not available upfront
    "Number of products sold",# outcome
    "Costs",                  # can encode outcome; drop to be safe
]


# ─────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame. Raises FileNotFoundError if missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place your CSV in the data/ folder and update the path."
        )
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Inspect dataset
# ─────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> dict:
    """Return a summary dict useful for the Streamlit dashboard."""
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "sample": df.head(5),
    }
    return summary


# ─────────────────────────────────────────────
# 3. Derive / validate the target column
# ─────────────────────────────────────────────
def add_or_validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for known target columns in priority order:
      1. Disruption_Occurred
      2. Late_delivery_risk
      3. Derive from Inspection results + Defect rates + Lead time

    Returns the DataFrame with a binary integer column TARGET_COLUMN.
    Raises ValueError if derivation is impossible.
    """
    df = df.copy()

    # Priority 1 – primary dataset label
    if "Disruption_Occurred" in df.columns:
        print("[target] Using existing column: Disruption_Occurred")
        df[TARGET_COLUMN] = df["Disruption_Occurred"].astype(int)
        return df

    # Priority 2 – DataCo label
    if "Late_delivery_risk" in df.columns:
        print("[target] Using existing column: Late_delivery_risk")
        df[TARGET_COLUMN] = df["Late_delivery_risk"].astype(int)
        return df

    # Priority 3 – Rule-based derivation
    required_cols = {"Inspection results", "Defect rates", "Lead time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot derive disruption label. Missing columns: {missing}\n"
            "Please provide a dataset with a labelled target column "
            "(Disruption_Occurred or Late_delivery_risk)."
        )

    inspection_fail = (df["Inspection results"].str.strip().str.lower() == "fail")
    high_defect     = (df["Defect rates"] > DEFECT_THRESHOLD)
    long_lead       = (df["Lead time"] > LEAD_TIME_THRESHOLD)

    df[TARGET_COLUMN] = ((inspection_fail | high_defect | long_lead)
                         .astype(int))

    n_pos = df[TARGET_COLUMN].sum()
    n_total = len(df)
    print(
        f"[target] Derived '{TARGET_COLUMN}' using rule-based logic.\n"
        f"         Rules: Inspection=='Fail' OR Defect rate>{DEFECT_THRESHOLD}% "
        f"OR Lead time>{LEAD_TIME_THRESHOLD} days\n"
        f"         Disrupted: {n_pos}/{n_total} "
        f"({100*n_pos/n_total:.1f}%)"
    )
    return df


# ─────────────────────────────────────────────
# 4. Clean the DataFrame
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, drop leakage / identifier columns."""
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"[clean] Removed {before - after} duplicate rows")

    # Drop columns that should not be used as features
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    # Also drop target leakage: Inspection results after deriving target
    for leakage_col in ["Inspection results"]:
        if leakage_col in df.columns and leakage_col not in cols_to_drop:
            cols_to_drop.append(leakage_col)

    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(f"[clean] Dropped columns: {cols_to_drop}")

    return df


# ─────────────────────────────────────────────
# 5. Identify feature types
# ─────────────────────────────────────────────
def identify_features(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """
    Return (numerical_cols, categorical_cols) excluding the target.
    """
    feature_cols = [c for c in df.columns if c != target]
    numerical_cols   = df[feature_cols].select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    print(f"[features] Numerical  : {numerical_cols}")
    print(f"[features] Categorical: {categorical_cols}")
    return numerical_cols, categorical_cols


# ─────────────────────────────────────────────
# 6. Build Scikit-learn preprocessing pipeline
# ─────────────────────────────────────────────
def build_preprocessor(numerical_cols: list, categorical_cols: list):
    """
    Returns a ColumnTransformer that:
      - Scales numerical features with StandardScaler
      - One-hot-encodes categorical features
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])
    return preprocessor


# ─────────────────────────────────────────────
# 7. Train / test split
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame, target: str = TARGET_COLUMN,
               test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Use stratify only when both classes are present
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    print(f"[split] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"[split] Class balance (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 8. Full preprocessing pipeline (entry point)
# ─────────────────────────────────────────────
def preprocess(filepath: str):
    """
    End-to-end preprocessing.
    Returns:
        X_train, X_test, y_train, y_test,
        preprocessor, numerical_cols, categorical_cols,
        full cleaned DataFrame
    """
    df = load_data(filepath)
    df = add_or_validate_target(df)
    df = clean_data(df)
    numerical_cols, categorical_cols = identify_features(df)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train, X_test, y_train, y_test = split_data(df)
    return (X_train, X_test, y_train, y_test,
            preprocessor, numerical_cols, categorical_cols, df)


# ─────────────────────────────────────────────
# Quick self-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "supply_chain_data.csv")
    result = preprocess(csv_path)
    X_train, X_test, y_train, y_test = result[:4]
    df = result[7]
    print(f"\nPreprocessing complete.")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"Target balance:\n{df[TARGET_COLUMN].value_counts()}")

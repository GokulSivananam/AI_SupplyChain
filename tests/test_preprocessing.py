"""
tests/test_preprocessing.py
----------------------------
Unit tests for dataset validation and preprocessing logic.
Run with:  python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from data_preprocessing import (
    add_or_validate_target, clean_data, identify_features,
    build_preprocessor, split_data, TARGET_COLUMN,
    DEFECT_THRESHOLD, LEAD_TIME_THRESHOLD,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
def make_sample_df(n=50) -> pd.DataFrame:
    """Create a minimal sample DataFrame matching the dataset schema."""
    np.random.seed(42)
    return pd.DataFrame({
        "Product type"          : np.random.choice(["haircare", "skincare"], n),
        "SKU"                   : [f"SKU{i}" for i in range(n)],
        "Price"                 : np.random.uniform(10, 100, n),
        "Availability"          : np.random.randint(10, 100, n),
        "Number of products sold": np.random.randint(50, 500, n),
        "Revenue generated"     : np.random.uniform(1000, 10000, n),
        "Customer demographics" : np.random.choice(["Male", "Female", "Unknown"], n),
        "Stock levels"          : np.random.randint(5, 100, n),
        "Lead times"            : np.random.randint(1, 30, n),
        "Order quantities"      : np.random.randint(10, 100, n),
        "Shipping times"        : np.random.randint(1, 10, n),
        "Shipping carriers"     : np.random.choice(["Carrier A", "Carrier B"], n),
        "Shipping costs"        : np.random.uniform(1, 20, n),
        "Supplier name"         : np.random.choice(["Supplier 1", "Supplier 2"], n),
        "Location"              : np.random.choice(["Mumbai", "Delhi"], n),
        "Lead time"             : np.random.randint(1, 30, n),
        "Production volumes"    : np.random.randint(100, 1000, n),
        "Manufacturing lead time": np.random.randint(5, 35, n),
        "Manufacturing costs"   : np.random.uniform(20, 60, n),
        "Inspection results"    : np.random.choice(["Pass", "Fail", "Pending"], n),
        "Defect rates"          : np.random.uniform(0, 5, n),
        "Transportation modes"  : np.random.choice(["Road", "Air", "Rail"], n),
        "Routes"                : np.random.choice(["Route A", "Route B"], n),
        "Costs"                 : np.random.uniform(100, 600, n),
    })


# ─────────────────────────────────────────────
# 1. Target derivation tests
# ─────────────────────────────────────────────
class TestTargetDerivation:

    def test_target_column_created(self):
        df = add_or_validate_target(make_sample_df())
        assert TARGET_COLUMN in df.columns, "Target column must be created"

    def test_target_binary(self):
        df = add_or_validate_target(make_sample_df())
        unique_vals = set(df[TARGET_COLUMN].unique())
        assert unique_vals.issubset({0, 1}), f"Target must be binary, got {unique_vals}"

    def test_inspection_fail_sets_disruption(self):
        """A row with Inspection='Fail' must be labelled disrupted."""
        df = make_sample_df(10)
        df["Inspection results"] = "Pass"
        df["Defect rates"]       = 0.5     # below threshold
        df["Lead time"]          = 5       # below threshold
        df.loc[0, "Inspection results"] = "Fail"
        df = add_or_validate_target(df)
        assert df.loc[0, TARGET_COLUMN] == 1

    def test_high_defect_sets_disruption(self):
        """A row with defect rate above threshold must be labelled disrupted."""
        df = make_sample_df(10)
        df["Inspection results"] = "Pass"
        df["Defect rates"]       = 0.5
        df["Lead time"]          = 5
        df.loc[0, "Defect rates"] = DEFECT_THRESHOLD + 1.0
        df = add_or_validate_target(df)
        assert df.loc[0, TARGET_COLUMN] == 1

    def test_long_lead_sets_disruption(self):
        """A row with lead time above threshold must be labelled disrupted."""
        df = make_sample_df(10)
        df["Inspection results"] = "Pass"
        df["Defect rates"]       = 0.5
        df["Lead time"]          = 5
        df.loc[0, "Lead time"] = LEAD_TIME_THRESHOLD + 5
        df = add_or_validate_target(df)
        assert df.loc[0, TARGET_COLUMN] == 1

    def test_clean_row_not_disrupted(self):
        """A row with all signals below threshold must NOT be disrupted."""
        df = make_sample_df(5)
        df["Inspection results"] = "Pass"
        df["Defect rates"]       = 0.5
        df["Lead time"]          = 5
        df = add_or_validate_target(df)
        assert (df[TARGET_COLUMN] == 0).all()

    def test_preexisting_disruption_column_used(self):
        """If 'Disruption_Occurred' is present, it must be used directly."""
        df = make_sample_df(10)
        df["Disruption_Occurred"] = np.random.randint(0, 2, 10)
        df_out = add_or_validate_target(df)
        assert TARGET_COLUMN in df_out.columns
        assert (df_out[TARGET_COLUMN] == df["Disruption_Occurred"]).all()

    def test_missing_required_columns_raises(self):
        """Missing derivation columns should raise ValueError."""
        df = make_sample_df(10).drop(
            columns=["Inspection results", "Defect rates", "Lead time"]
        )
        with pytest.raises(ValueError):
            add_or_validate_target(df)


# ─────────────────────────────────────────────
# 2. Cleaning tests
# ─────────────────────────────────────────────
class TestCleanData:

    def test_duplicates_removed(self):
        df = make_sample_df(10)
        df = add_or_validate_target(df)
        df_dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
        df_clean = clean_data(df_dup)
        assert len(df_clean) == 10, "Duplicates must be removed"

    def test_sku_dropped(self):
        df = make_sample_df(10)
        df = add_or_validate_target(df)
        df_clean = clean_data(df)
        assert "SKU" not in df_clean.columns, "SKU identifier must be dropped"

    def test_inspection_results_dropped(self):
        """Inspection results is a target-leakage column and must be removed."""
        df = make_sample_df(10)
        df = add_or_validate_target(df)
        df_clean = clean_data(df)
        assert "Inspection results" not in df_clean.columns

    def test_target_preserved(self):
        df = make_sample_df(10)
        df = add_or_validate_target(df)
        df_clean = clean_data(df)
        assert TARGET_COLUMN in df_clean.columns


# ─────────────────────────────────────────────
# 3. Feature identification tests
# ─────────────────────────────────────────────
class TestIdentifyFeatures:

    def setup_method(self):
        df = make_sample_df(20)
        df = add_or_validate_target(df)
        self.df = clean_data(df)

    def test_numerical_cols_are_numeric(self):
        num_cols, _ = identify_features(self.df)
        for col in num_cols:
            assert pd.api.types.is_numeric_dtype(self.df[col]), \
                f"{col} is not numeric"

    def test_categorical_cols_are_object(self):
        _, cat_cols = identify_features(self.df)
        for col in cat_cols:
            assert self.df[col].dtype in [object, "string", "category"], \
                f"{col} is not categorical"

    def test_target_not_in_features(self):
        num_cols, cat_cols = identify_features(self.df)
        assert TARGET_COLUMN not in num_cols
        assert TARGET_COLUMN not in cat_cols


# ─────────────────────────────────────────────
# 4. Train/test split tests
# ─────────────────────────────────────────────
class TestSplitData:

    def setup_method(self):
        df = make_sample_df(50)
        df = add_or_validate_target(df)
        self.df = clean_data(df)

    def test_split_sizes(self):
        X_train, X_test, y_train, y_test = split_data(self.df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(self.df)

    def test_no_overlap(self):
        X_train, X_test, _, _ = split_data(self.df)
        train_idx = set(X_train.index)
        test_idx  = set(X_test.index)
        assert len(train_idx & test_idx) == 0, "Train and test must not overlap"

    def test_target_not_in_features(self):
        X_train, X_test, _, _ = split_data(self.df)
        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns


# ─────────────────────────────────────────────
# 5. Preprocessor pipeline tests
# ─────────────────────────────────────────────
class TestPreprocessor:

    def setup_method(self):
        df = make_sample_df(50)
        df = add_or_validate_target(df)
        self.df = clean_data(df)
        self.num_cols, self.cat_cols = identify_features(self.df)
        self.preprocessor = build_preprocessor(self.num_cols, self.cat_cols)

    def test_preprocessor_fits_without_error(self):
        X = self.df.drop(columns=[TARGET_COLUMN])
        self.preprocessor.fit(X)  # Should not raise

    def test_preprocessor_transforms_to_array(self):
        X = self.df.drop(columns=[TARGET_COLUMN])
        self.preprocessor.fit(X)
        X_transformed = self.preprocessor.transform(X)
        assert isinstance(X_transformed, np.ndarray), \
            "Transformed output must be ndarray"

    def test_no_nan_after_transform(self):
        X = self.df.drop(columns=[TARGET_COLUMN])
        self.preprocessor.fit(X)
        X_transformed = self.preprocessor.transform(X)
        assert not np.isnan(X_transformed).any(), \
            "No NaN values should remain after preprocessing"

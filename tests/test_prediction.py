"""
tests/test_prediction.py
------------------------
Unit tests for prediction input handling and risk categorisation.
Run with:  python -m pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from predict import risk_category, get_recommendations, RISK_LOW_MAX, RISK_MEDIUM_MAX


# ─────────────────────────────────────────────
# Risk category tests
# ─────────────────────────────────────────────
class TestRiskCategory:

    def test_low_risk(self):
        cat = risk_category(0.10)
        assert "Low" in cat

    def test_medium_risk(self):
        cat = risk_category(0.50)
        assert "Medium" in cat

    def test_high_risk(self):
        cat = risk_category(0.90)
        assert "High" in cat

    def test_boundary_low_medium(self):
        # Exactly at boundary → Medium
        cat = risk_category(RISK_LOW_MAX)
        assert "Medium" in cat

    def test_boundary_medium_high(self):
        # Just above medium boundary → High
        cat = risk_category(RISK_MEDIUM_MAX + 0.01)
        assert "High" in cat

    def test_zero_probability(self):
        cat = risk_category(0.0)
        assert "Low" in cat

    def test_one_probability(self):
        cat = risk_category(1.0)
        assert "High" in cat

    def test_returns_string(self):
        assert isinstance(risk_category(0.5), str)


# ─────────────────────────────────────────────
# Recommendation tests
# ─────────────────────────────────────────────
class TestGetRecommendations:

    def test_returns_list(self):
        recs = get_recommendations(0.5, {})
        assert isinstance(recs, list)

    def test_non_empty_recommendations(self):
        recs = get_recommendations(0.8, {})
        assert len(recs) > 0, "High-risk should produce at least one recommendation"

    def test_high_risk_recommendation_text(self):
        recs = get_recommendations(0.9, {})
        combined = " ".join(recs).lower()
        assert "risk" in combined or "supplier" in combined or "inspection" in combined

    def test_low_risk_recommendation_text(self):
        recs = get_recommendations(0.1, {})
        combined = " ".join(recs).lower()
        assert "low" in combined or "standard" in combined or "monitoring" in combined

    def test_high_defect_rate_triggers_tip(self):
        recs = get_recommendations(0.5, {"Defect rates": 4.0})
        combined = " ".join(recs).lower()
        assert "defect" in combined

    def test_long_lead_time_triggers_tip(self):
        recs = get_recommendations(0.5, {"Lead time": 25})
        combined = " ".join(recs).lower()
        assert "lead" in combined

    def test_no_extra_tips_for_clean_shipment(self):
        """A clean shipment should not trigger defect/lead-time tips."""
        recs = get_recommendations(0.1, {"Defect rates": 0.5, "Lead time": 5})
        combined = " ".join(recs).lower()
        # The defect / lead-time-specific tips should NOT be present
        assert "defect rate" not in combined
        assert "lead time" not in combined


# ─────────────────────────────────────────────
# predict_single input validation tests
# (uses a mock pipeline to avoid disk dependency)
# ─────────────────────────────────────────────
class MockPipeline:
    """Minimal mock that mimics sklearn Pipeline predict interface."""

    def predict(self, df):
        return np.array([1] * len(df))

    def predict_proba(self, df):
        return np.array([[0.2, 0.8]] * len(df))


from predict import predict_single


class TestPredictSingle:

    def setup_method(self):
        self.pipeline = MockPipeline()
        self.valid_input = {
            "Product type"          : "haircare",
            "Price"                 : 50.0,
            "Availability"          : 60,
            "Customer demographics" : "Female",
            "Stock levels"          : 30,
            "Lead times"            : 15,
            "Order quantities"      : 50,
            "Shipping times"        : 5,
            "Shipping carriers"     : "Carrier A",
            "Shipping costs"        : 10.0,
            "Supplier name"         : "Supplier 1",
            "Location"              : "Mumbai",
            "Lead time"             : 15,
            "Production volumes"    : 300,
            "Manufacturing lead time": 20,
            "Manufacturing costs"   : 40.0,
            "Defect rates"          : 1.0,
            "Transportation modes"  : "Road",
            "Routes"                : "Route A",
        }

    def test_returns_dict(self):
        result = predict_single(self.pipeline, self.valid_input)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        result = predict_single(self.pipeline, self.valid_input)
        for key in ["prediction", "probability", "risk_category", "recommendations"]:
            assert key in result, f"Missing key: {key}"

    def test_probability_between_0_and_1(self):
        result = predict_single(self.pipeline, self.valid_input)
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_binary(self):
        result = predict_single(self.pipeline, self.valid_input)
        assert result["prediction"] in [0, 1]

    def test_prediction_label_correct(self):
        result = predict_single(self.pipeline, self.valid_input)
        expected = "Disruption Likely" if result["prediction"] == 1 else "No Disruption"
        assert result["prediction_label"] == expected

    def test_recommendations_non_empty(self):
        result = predict_single(self.pipeline, self.valid_input)
        assert len(result["recommendations"]) > 0

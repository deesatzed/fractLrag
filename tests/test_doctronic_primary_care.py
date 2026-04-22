"""
Tests for doctronic_primary_care_ownership.py
Covers: PatientProfile, DoctronicPrimaryCareEngine
Includes regression test for known bug:
  - BUG: churn risk always calculated with days_since=0 (line 106)
"""
import sys
import os
import numpy as np
import pytest
from datetime import datetime, timedelta

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from doctronic_primary_care_ownership import PatientProfile, DoctronicPrimaryCareEngine


# ============================================================
# PatientProfile UNIT TESTS
# ============================================================
class TestPatientProfile:
    """Test the PatientProfile dataclass."""

    def test_creation(self):
        p = PatientProfile(patient_id="P-001", age=55)
        assert p.patient_id == "P-001"
        assert p.age == 55
        assert p.conditions == []
        assert p.churn_risk_score == 0.0
        assert p.relationship_length_days == 0

    def test_update_churn_risk_zero_days(self):
        p = PatientProfile(patient_id="P-001", age=50)
        p.update_churn_risk(0, 0.8)
        # base_risk = 0/45 = 0, value_factor = 1-0.8 = 0.2
        # churn = (0 * 0.6) + (0.2 * 0.4) = 0.08
        assert p.churn_risk_score == pytest.approx(0.08, abs=0.001)

    def test_update_churn_risk_high_days(self):
        p = PatientProfile(patient_id="P-001", age=50)
        p.update_churn_risk(45, 0.5)
        # base_risk = min(45/45, 1.0) = 1.0, value_factor = 1-0.5 = 0.5
        # churn = (1.0 * 0.6) + (0.5 * 0.4) = 0.6 + 0.2 = 0.8
        assert p.churn_risk_score == pytest.approx(0.8, abs=0.001)

    def test_update_churn_risk_capped_at_one(self):
        p = PatientProfile(patient_id="P-001", age=50)
        p.update_churn_risk(100, 0.0)
        # base_risk = min(100/45, 1.0) = 1.0, value_factor = 1.0
        # churn = (1.0 * 0.6) + (1.0 * 0.4) = 1.0
        assert p.churn_risk_score == pytest.approx(1.0, abs=0.001)

    def test_update_churn_risk_full_value(self):
        p = PatientProfile(patient_id="P-001", age=50)
        p.update_churn_risk(0, 1.0)
        # base_risk = 0, value_factor = max(0, 1-1.0) = 0
        # churn = 0
        assert p.churn_risk_score == pytest.approx(0.0, abs=0.001)

    def test_update_churn_risk_over_value(self):
        p = PatientProfile(patient_id="P-001", age=50)
        p.update_churn_risk(0, 1.5)
        # value_factor = max(0, 1-1.5) = 0
        # churn = 0
        assert p.churn_risk_score == pytest.approx(0.0, abs=0.001)

    def test_to_json_valid(self):
        import json
        p = PatientProfile(patient_id="P-001", age=50, conditions=["Diabetes"])
        j = p.to_json()
        data = json.loads(j)
        assert data["patient_id"] == "P-001"
        assert data["age"] == 50
        assert "Diabetes" in data["conditions"]

    def test_to_json_with_datetime_field(self):
        import json
        p = PatientProfile(patient_id="P-001", age=50)
        p.last_interaction = datetime.now().isoformat()
        j = p.to_json()
        data = json.loads(j)  # Should not raise
        assert "last_interaction" in data


# ============================================================
# DoctronicPrimaryCareEngine INTEGRATION TESTS
# ============================================================
class TestOnboardPatient:
    """Test patient onboarding."""

    def test_onboard_creates_profile(self, doctronic_engine):
        profile = doctronic_engine.onboard_patient("P-001", 55, ["Diabetes"], "Stay healthy")
        assert isinstance(profile, PatientProfile)
        assert profile.patient_id == "P-001"
        assert profile.age == 55

    def test_onboard_stores_patient(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, ["Diabetes"], "Stay healthy")
        assert "P-001" in doctronic_engine.patients

    def test_onboard_initializes_history(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        assert "P-001" in doctronic_engine.interaction_history
        assert len(doctronic_engine.interaction_history["P-001"]) == 0

    def test_onboard_logs_action(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        assert any("ONBOARD" in entry for entry in doctronic_engine.decision_log)

    def test_onboard_sets_last_interaction(self, doctronic_engine):
        profile = doctronic_engine.onboard_patient("P-001", 55, [], "")
        assert profile.last_interaction != ""


class TestRecordInteraction:
    """Test interaction recording."""

    def test_records_interaction(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.record_interaction("P-001", "check_up", 0.8, "Routine visit")
        assert len(doctronic_engine.interaction_history["P-001"]) == 1

    def test_interaction_structure(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.record_interaction("P-001", "check_up", 0.8, "Routine")
        interaction = doctronic_engine.interaction_history["P-001"][0]
        assert "timestamp" in interaction
        assert "type" in interaction
        assert "value_score" in interaction
        assert "notes" in interaction

    def test_unknown_patient_ignored(self, doctronic_engine):
        # Should not crash when patient not found
        doctronic_engine.record_interaction("UNKNOWN", "check_up", 0.5)
        # No crash is the assertion

    def test_multiple_interactions_accumulated(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        for i in range(5):
            doctronic_engine.record_interaction("P-001", f"visit_{i}", 0.7)
        assert len(doctronic_engine.interaction_history["P-001"]) == 5


class TestBugRegressionChurnRiskAlwaysZeroDays:
    """
    BUG: At doctronic_primary_care_ownership.py line 103-106:
      profile.last_interaction = now.isoformat()  # line 103
      days_since = (now - datetime.fromisoformat(profile.last_interaction)).days  # line 106

    Because last_interaction is set to 'now' on line 103 BEFORE calculating
    days_since on line 106, days_since is ALWAYS 0.
    This means update_churn_risk always gets days_since=0, so the time-based
    component of churn prediction is completely non-functional.
    """

    def test_churn_risk_never_reflects_time_gap(self, doctronic_engine):
        """
        Even after waiting (simulated), churn risk should increase with time
        gaps between interactions. Due to the bug, it does NOT.
        """
        doctronic_engine.onboard_patient("P-001", 55, ["Diabetes"], "Stay healthy")

        # First interaction
        doctronic_engine.record_interaction("P-001", "initial", 0.5)
        risk_after_first = doctronic_engine.patients["P-001"].churn_risk_score

        # Second interaction (immediately, same session)
        doctronic_engine.record_interaction("P-001", "followup", 0.5)
        risk_after_second = doctronic_engine.patients["P-001"].churn_risk_score

        # BUG PROOF: Both interactions compute days_since=0, so risk is identical
        # In a correct implementation, even a small time gap would change the risk.
        assert risk_after_first == risk_after_second, (
            "BUG REGRESSION: Churn risk is now different between interactions, "
            "which means the days_since=0 bug may have been fixed. Update this test."
        )

    def test_days_since_always_zero_proof(self, doctronic_engine):
        """
        Directly test that the churn risk for any interaction equals
        what you'd get with days_since=0, proving the bug.
        """
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        value_score = 0.6

        doctronic_engine.record_interaction("P-001", "test", value_score)
        actual_risk = doctronic_engine.patients["P-001"].churn_risk_score

        # Calculate expected risk with days_since=0
        base_risk = min(0 / 45.0, 1.0)  # 0
        value_factor = max(0, 1.0 - value_score)  # 0.4
        expected_risk = round((base_risk * 0.6 + value_factor * 0.4), 3)  # 0.16

        assert actual_risk == pytest.approx(expected_risk, abs=0.001), (
            f"Expected risk={expected_risk} (days_since=0), got {actual_risk}"
        )

    def test_relationship_length_is_zero_for_rapid_interactions(self, doctronic_engine):
        """
        The relationship_length_days is also computed using last_interaction,
        but at least it is computed BEFORE last_interaction is updated (line 92-93).
        With rapid successive calls, it will still be 0 days.
        """
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.record_interaction("P-001", "first", 0.5)
        doctronic_engine.record_interaction("P-001", "second", 0.5)
        # Both interactions happen in same second, so days = 0
        assert doctronic_engine.patients["P-001"].relationship_length_days == 0


class TestPredictTrajectory:
    """Test trajectory prediction."""

    def test_empty_history_returns_stable(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        trajectory = doctronic_engine.predict_trajectory("P-001")
        assert trajectory["current_state"] == "stable"
        assert trajectory["velocity"] == 0

    def test_improving_trajectory(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        # Record interactions with increasing value
        for v in [0.3, 0.4, 0.5, 0.6, 0.8]:
            doctronic_engine.record_interaction("P-001", "visit", v)
        trajectory = doctronic_engine.predict_trajectory("P-001")
        assert trajectory["velocity"] > 0
        assert trajectory["current_state"] == "improving"

    def test_declining_trajectory(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        for v in [0.8, 0.6, 0.4, 0.3, 0.2]:
            doctronic_engine.record_interaction("P-001", "visit", v)
        trajectory = doctronic_engine.predict_trajectory("P-001")
        assert trajectory["velocity"] < 0
        assert trajectory["current_state"] == "declining"

    def test_unknown_patient_returns_empty(self, doctronic_engine):
        assert doctronic_engine.predict_trajectory("UNKNOWN") == {}

    def test_predicted_churn_bounded(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.record_interaction("P-001", "visit", 0.5)
        trajectory = doctronic_engine.predict_trajectory("P-001")
        assert 0.0 <= trajectory["predicted_churn_30_days"] <= 1.0


class TestRecommendAction:
    """Test action recommendation logic."""

    def test_high_churn_risk_action(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.patients["P-001"].churn_risk_score = 0.8
        action = doctronic_engine._recommend_action(
            doctronic_engine.patients["P-001"], 0.0, 0.0
        )
        assert "outreach" in action.lower() or "churn" in action.lower()

    def test_negative_acceleration_action(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.patients["P-001"].churn_risk_score = 0.3
        action = doctronic_engine._recommend_action(
            doctronic_engine.patients["P-001"], 0.0, -0.2
        )
        assert "review" in action.lower() or "intervention" in action.lower()

    def test_proactive_positive_velocity(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.patients["P-001"].churn_risk_score = 0.3
        doctronic_engine.patients["P-001"].engagement_preference = "proactive"
        action = doctronic_engine._recommend_action(
            doctronic_engine.patients["P-001"], 0.2, 0.0
        )
        assert "proactive" in action.lower()

    def test_default_action(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, [], "")
        doctronic_engine.patients["P-001"].churn_risk_score = 0.3
        doctronic_engine.patients["P-001"].engagement_preference = "reactive"
        action = doctronic_engine._recommend_action(
            doctronic_engine.patients["P-001"], 0.0, 0.0
        )
        assert "check-in" in action.lower() or "standard" in action.lower()


class TestCorrelateResearch:
    """Test research correlation."""

    def test_known_condition(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, ["Type 2 Diabetes"], "")
        research = doctronic_engine.correlate_research("P-001", "Type 2 Diabetes")
        assert len(research) > 0
        assert any("ADA" in r or "diabetes" in r.lower() for r in research)

    def test_hypertension_condition(self, doctronic_engine):
        research = doctronic_engine.correlate_research("P-001", "Hypertension")
        assert len(research) > 0

    def test_unknown_condition_fallback(self, doctronic_engine):
        research = doctronic_engine.correlate_research("P-001", "Rare Disease XYZ")
        assert len(research) > 0
        assert "personalized" in research[0].lower()


class TestGenerateProactiveInsight:
    """Test insight generation."""

    def test_generates_insight_for_known_patient(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, ["Type 2 Diabetes"], "Stay healthy")
        doctronic_engine.record_interaction("P-001", "visit", 0.7)
        insight = doctronic_engine.generate_proactive_insight("P-001")
        assert "P-001" in insight
        assert "Trajectory" in insight or "trajectory" in insight.lower()

    def test_unknown_patient_returns_empty(self, doctronic_engine):
        assert doctronic_engine.generate_proactive_insight("UNKNOWN") == ""

    def test_insight_includes_research(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, ["Hypertension"], "")
        doctronic_engine.record_interaction("P-001", "visit", 0.5)
        insight = doctronic_engine.generate_proactive_insight("P-001")
        assert "Research" in insight or "research" in insight.lower()

    def test_insight_logs_action(self, doctronic_engine):
        doctronic_engine.onboard_patient("P-001", 55, ["Diabetes"], "")
        doctronic_engine.record_interaction("P-001", "visit", 0.5)
        log_before = len(doctronic_engine.decision_log)
        doctronic_engine.generate_proactive_insight("P-001")
        assert len(doctronic_engine.decision_log) > log_before

"""
Tests for doctronic_b2b_outsourced_intelligence.py
Covers: PatientProfile (B2B version), DoctronicB2BIntelligence class
Includes regression test for discovered bug:
  - BUG: nurse_decision_support and post_visit_primary_care access
    profile.conditions[0] without checking if conditions is empty,
    causing IndexError. (lines 101, 116)
"""
import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from doctronic_b2b_outsourced_intelligence import PatientProfile, DoctronicB2BIntelligence


# ============================================================
# PatientProfile (B2B) UNIT TESTS
# ============================================================
class TestPatientProfileB2B:
    """Test the B2B version of PatientProfile."""

    def test_creation(self):
        p = PatientProfile("P-001", 60, ["Diabetes"], "Stay healthy")
        assert p.patient_id == "P-001"
        assert p.age == 60
        assert p.conditions == ["Diabetes"]
        assert p.goals == "Stay healthy"

    def test_default_values(self):
        p = PatientProfile("P-001", 40, [], "")
        assert p.risk_tolerance == "medium"
        assert p.total_visits == 0
        assert p.churn_risk == 0.0
        assert p.primary_care_ownership_score == 0.0

    def test_to_dict(self):
        p = PatientProfile("P-001", 55, ["HF"], "Avoid ER")
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["patient_id"] == "P-001"
        assert d["age"] == 55
        assert "HF" in d["conditions"]
        assert d["goals"] == "Avoid ER"


# ============================================================
# DoctronicB2BIntelligence INTEGRATION TESTS
# ============================================================
class TestB2BOnboardPatient:
    """Test B2B patient onboarding."""

    def test_onboard_creates_profile(self, b2b_engine):
        p = b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "Stay healthy", "SITE-A")
        assert isinstance(p, PatientProfile)
        assert p.patient_id == "P-001"

    def test_onboard_stores_patient(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        assert "P-001" in b2b_engine.patients

    def test_onboard_initializes_site_metrics(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        assert "SITE-A" in b2b_engine.site_metrics
        assert b2b_engine.site_metrics["SITE-A"]["total_patients"] == 1

    def test_onboard_increments_site_patient_count(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        b2b_engine.onboard_patient("P-002", 45, [], "", "SITE-A")
        assert b2b_engine.site_metrics["SITE-A"]["total_patients"] == 2

    def test_multiple_sites(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        b2b_engine.onboard_patient("P-002", 45, [], "", "SITE-B")
        assert "SITE-A" in b2b_engine.site_metrics
        assert "SITE-B" in b2b_engine.site_metrics

    def test_logs_onboarding(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        assert any("ONBOARD" in entry for entry in b2b_engine.decision_log)


class TestPreTriage:
    """Test AI-powered pre-triage."""

    def test_known_patient_returns_risk(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes", "Hypertension"], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "mild fatigue", "SITE-A")
        assert "risk_score" in result
        assert "recommendation" in result
        assert "suggested_pathway" in result

    def test_unknown_patient_returns_default(self, b2b_engine):
        result = b2b_engine.pre_triage("UNKNOWN", "chest pain", "SITE-A")
        assert result["risk_level"] == "medium"
        assert "Full assessment" in result["recommendation"]

    def test_chest_pain_high_risk(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 30, [], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "severe chest pain", "SITE-A")
        assert result["risk_score"] >= 0.9

    def test_shortness_of_breath_high_risk(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 30, [], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "shortness of breath", "SITE-A")
        assert result["risk_score"] >= 0.9

    def test_low_risk_nurse_led(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 25, [], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "mild headache", "SITE-A")
        assert result["suggested_pathway"] == "Nurse-led"

    def test_high_risk_physician_review(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 80, ["HF", "CKD", "Diabetes", "COPD", "Stroke"], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "chest pain and shortness of breath", "SITE-A")
        assert result["suggested_pathway"] == "Physician review"

    def test_age_contributes_to_risk(self, b2b_engine):
        b2b_engine.onboard_patient("P-YOUNG", 25, [], "", "SITE-A")
        b2b_engine.onboard_patient("P-OLD", 85, [], "", "SITE-A")
        result_young = b2b_engine.pre_triage("P-YOUNG", "mild headache", "SITE-A")
        result_old = b2b_engine.pre_triage("P-OLD", "mild headache", "SITE-A")
        assert result_old["risk_score"] > result_young["risk_score"]

    def test_conditions_contribute_to_risk(self, b2b_engine):
        b2b_engine.onboard_patient("P-HEALTHY", 50, [], "", "SITE-A")
        b2b_engine.onboard_patient("P-COMORBID", 50, ["HF", "CKD", "Diabetes"], "", "SITE-A")
        r_healthy = b2b_engine.pre_triage("P-HEALTHY", "mild headache", "SITE-A")
        r_comorbid = b2b_engine.pre_triage("P-COMORBID", "mild headache", "SITE-A")
        assert r_comorbid["risk_score"] > r_healthy["risk_score"]

    def test_risk_score_capped(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 99, ["A", "B", "C", "D", "E", "F", "G"], "", "SITE-A")
        result = b2b_engine.pre_triage("P-001", "mild symptom", "SITE-A")
        assert result["risk_score"] <= 0.95

    def test_logs_triage(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        b2b_engine.pre_triage("P-001", "symptom", "SITE-A")
        assert any("PRE-TRIAGE" in entry for entry in b2b_engine.decision_log)


class TestNurseDecisionSupport:
    """Test real-time nurse decision support."""

    def test_returns_support_for_known_patient(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        result = b2b_engine.nurse_decision_support("P-001", "Blood pressure normal")
        assert "recommended_action" in result
        assert "updated_risk" in result
        assert "primary_care_plan" in result

    def test_unknown_patient_returns_empty(self, b2b_engine):
        result = b2b_engine.nurse_decision_support("UNKNOWN", "findings")
        assert result == {}

    def test_abnormal_findings_escalate(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["HF"], "", "SITE-A")
        result = b2b_engine.nurse_decision_support("P-001", "abnormal lung sounds detected")
        assert "Escalate" in result["recommended_action"]

    def test_normal_findings_continue(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        result = b2b_engine.nurse_decision_support("P-001", "All vitals normal")
        assert "Proceed" in result["recommended_action"] or "nurse" in result["recommended_action"].lower()

    def test_risk_increases_with_support_call(self, b2b_engine):
        """Churn risk should increase after a nurse decision support call."""
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        initial_risk = b2b_engine.patients["P-001"].churn_risk
        b2b_engine.nurse_decision_support("P-001", "findings")
        assert b2b_engine.patients["P-001"].churn_risk > initial_risk

    def test_empty_conditions_crashes_nurse_support(self, b2b_engine):
        """
        BUG: nurse_decision_support accesses profile.conditions[0] without
        checking if conditions is empty. This causes IndexError.
        At doctronic_b2b_outsourced_intelligence.py line 101.
        """
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        with pytest.raises(IndexError):
            b2b_engine.nurse_decision_support("P-001", "findings")

    def test_primary_care_plan_mentions_condition(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Hypertension"], "", "SITE-A")
        result = b2b_engine.nurse_decision_support("P-001", "Normal")
        assert "Hypertension" in result["primary_care_plan"]


class TestPostVisitPrimaryCare:
    """Test post-visit primary care ownership."""

    def test_returns_plan_for_known_patient(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        plan = b2b_engine.post_visit_primary_care("P-001", "Stable condition")
        assert "P-001" in plan
        assert "follow-up" in plan.lower() or "Follow-up" in plan

    def test_unknown_patient_returns_empty(self, b2b_engine):
        assert b2b_engine.post_visit_primary_care("UNKNOWN", "discharge") == ""

    def test_increments_total_visits(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        assert b2b_engine.patients["P-001"].total_visits == 0
        b2b_engine.post_visit_primary_care("P-001", "discharge")
        assert b2b_engine.patients["P-001"].total_visits == 1

    def test_increases_ownership_score(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        initial = b2b_engine.patients["P-001"].primary_care_ownership_score
        b2b_engine.post_visit_primary_care("P-001", "discharge")
        assert b2b_engine.patients["P-001"].primary_care_ownership_score > initial

    def test_ownership_score_capped_at_one(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        for _ in range(10):
            b2b_engine.post_visit_primary_care("P-001", "discharge")
        assert b2b_engine.patients["P-001"].primary_care_ownership_score <= 1.0

    def test_logs_post_visit(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, ["Diabetes"], "", "SITE-A")
        b2b_engine.post_visit_primary_care("P-001", "discharge")
        assert any("POST-VISIT" in entry for entry in b2b_engine.decision_log)

    def test_empty_conditions_crashes_post_visit(self, b2b_engine):
        """
        BUG: post_visit_primary_care accesses profile.conditions[0] without
        checking if conditions is empty. This causes IndexError.
        At doctronic_b2b_outsourced_intelligence.py line 116.
        """
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        with pytest.raises(IndexError):
            b2b_engine.post_visit_primary_care("P-001", "discharge")


class TestCalculateSiteROI:
    """Test ROI calculation."""

    def test_returns_required_fields(self, b2b_engine):
        b2b_engine.onboard_patient("P-001", 55, [], "", "SITE-A")
        roi = b2b_engine.calculate_site_roi("SITE-A", monthly_visits=1000)
        assert "monthly_savings" in roi
        assert "annual_savings" in roi
        assert "doctronic_fee" in roi
        assert "net_annual_benefit" in roi
        assert "roi" in roi

    def test_monthly_savings_calculation(self, b2b_engine):
        roi = b2b_engine.calculate_site_roi("NEW-SITE", monthly_visits=1000, avg_physician_cost=180)
        # patients=1000, reduction=0.28, cost=180
        expected_monthly = round(1000 * 0.28 * 180)
        assert roi["monthly_savings"] == expected_monthly

    def test_annual_savings_is_twelve_times_monthly(self, b2b_engine):
        roi = b2b_engine.calculate_site_roi("SITE", monthly_visits=500)
        assert roi["annual_savings"] == roi["monthly_savings"] * 12

    def test_doctronic_fee_calculation(self, b2b_engine):
        roi = b2b_engine.calculate_site_roi("SITE", monthly_visits=1000)
        # $12 per patient per month
        assert roi["doctronic_fee"] == round(1000 * 12)

    def test_net_benefit_positive(self, b2b_engine):
        roi = b2b_engine.calculate_site_roi("SITE", monthly_visits=1000)
        # Savings should exceed fees for reasonable volumes
        assert roi["net_annual_benefit"] > 0

    def test_roi_calculation(self, b2b_engine):
        roi = b2b_engine.calculate_site_roi("SITE", monthly_visits=1000)
        expected_roi = round(
            ((roi["annual_savings"] - (roi["doctronic_fee"] * 12)) / (roi["doctronic_fee"] * 12)) * 100, 1
        )
        assert roi["roi"] == expected_roi

    def test_different_physician_cost(self, b2b_engine):
        roi_low = b2b_engine.calculate_site_roi("SITE", monthly_visits=1000, avg_physician_cost=100)
        roi_high = b2b_engine.calculate_site_roi("SITE", monthly_visits=1000, avg_physician_cost=300)
        assert roi_high["monthly_savings"] > roi_low["monthly_savings"]

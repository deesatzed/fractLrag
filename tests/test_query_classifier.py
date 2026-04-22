"""
Tests for the multi-signal query type classifier.

Validates classify_query_type() against all 40 benchmark queries (ground truth)
and structural edge cases. No mocks.
"""

import json
import pytest
from pathlib import Path
from fractrag.query import classify_query_type


# ============================================================
# Ground truth: all 40 medical benchmark queries
# ============================================================

BENCHMARK_QUERIES_PATH = Path(__file__).parent.parent / "corpus" / "medical_queries.json"


class TestBenchmarkQueryClassification:
    """Every benchmark query must be classified correctly."""

    @pytest.fixture(scope="class")
    def benchmark_queries(self):
        return json.loads(BENCHMARK_QUERIES_PATH.read_text())["queries"]

    def test_all_40_queries_classified_correctly(self, benchmark_queries):
        failures = []
        for q in benchmark_queries:
            predicted = classify_query_type(q["query_text"])
            if predicted != q["query_type"]:
                failures.append(
                    f"[{q['query_type']} -> {predicted}] {q['query_text'][:80]}"
                )
        assert failures == [], f"Misclassified {len(failures)} queries:\n" + "\n".join(failures)

    def test_specification_accuracy(self, benchmark_queries):
        spec_queries = [q for q in benchmark_queries if q["query_type"] == "specification"]
        correct = sum(1 for q in spec_queries if classify_query_type(q["query_text"]) == "specification")
        assert correct >= 8, f"Specification accuracy: {correct}/{len(spec_queries)}"

    def test_summary_accuracy(self, benchmark_queries):
        sum_queries = [q for q in benchmark_queries if q["query_type"] == "summary"]
        correct = sum(1 for q in sum_queries if classify_query_type(q["query_text"]) == "summary")
        assert correct >= 8, f"Summary accuracy: {correct}/{len(sum_queries)}"

    def test_logic_accuracy(self, benchmark_queries):
        logic_queries = [q for q in benchmark_queries if q["query_type"] == "logic"]
        correct = sum(1 for q in logic_queries if classify_query_type(q["query_text"]) == "logic")
        assert correct >= 8, f"Logic accuracy: {correct}/{len(logic_queries)}"

    def test_synthesis_accuracy(self, benchmark_queries):
        synth_queries = [q for q in benchmark_queries if q["query_type"] == "synthesis"]
        correct = sum(1 for q in synth_queries if classify_query_type(q["query_text"]) == "synthesis")
        assert correct >= 8, f"Synthesis accuracy: {correct}/{len(synth_queries)}"

    def test_overall_accuracy_above_80_percent(self, benchmark_queries):
        correct = sum(
            1 for q in benchmark_queries
            if classify_query_type(q["query_text"]) == q["query_type"]
        )
        accuracy = correct / len(benchmark_queries)
        assert accuracy >= 0.80, f"Overall accuracy {accuracy:.1%} below 80% threshold"


# ============================================================
# Structural pattern tests
# ============================================================

class TestSpecificationPatterns:
    def test_what_type_with_study_ref(self):
        assert classify_query_type("What type of therapy does the randomized trial compare?") == "specification"

    def test_how_many(self):
        assert classify_query_type("How many studies were included in the review?") == "specification"

    def test_in_what_specialty(self):
        assert classify_query_type("In what medical specialty is AI being applied?") == "specification"

    def test_what_specific(self):
        assert classify_query_type("What specific drug interactions does the review examine?") == "specification"

    def test_what_is_the_short(self):
        assert classify_query_type("What is the exact date?") == "specification"

    def test_what_is_the_difference(self):
        assert classify_query_type("What is the difference between mitosis and meiosis?") == "specification"


class TestSummaryPatterns:
    def test_summarize(self):
        assert classify_query_type("Summarize the key findings.") == "summary"

    def test_overview(self):
        assert classify_query_type("Give an overview of the field.") == "summary"

    def test_how_is_being_applied(self):
        assert classify_query_type("How is AI being applied to triage patients?") == "summary"

    def test_what_role_does(self):
        assert classify_query_type("What role does AI play in patient care?") == "summary"

    def test_what_are_recent_advances(self):
        assert classify_query_type("What are recent advances in drug safety?") == "summary"

    def test_short_what_is(self):
        assert classify_query_type("What is DNA?") == "summary"

    def test_short_explain(self):
        assert classify_query_type("Explain DNA") == "summary"


class TestLogicPatterns:
    def test_why(self):
        assert classify_query_type("Why is explainability important for trust?") == "logic"

    def test_how_does_mechanism(self):
        assert classify_query_type("How does deep learning detect retinal disease?") == "logic"

    def test_obstacles(self):
        assert classify_query_type("What are the obstacles preventing wider adoption of AI?") == "logic"

    def test_compare_short(self):
        assert classify_query_type("Compare DNA and RNA structures") == "logic"

    def test_effect_standalone(self):
        assert classify_query_type("What is the effect of gravity?") == "specification"

    def test_how_are_connected(self):
        assert classify_query_type("How are they connected?") == "logic"


class TestSynthesisPatterns:
    def test_both_and(self):
        assert classify_query_type("How does AI impact both drug safety and clinical decision support?") == "synthesis"

    def test_versus(self):
        assert classify_query_type("Compare AI effectiveness in skin conditions versus cardiac conditions.") == "synthesis"

    def test_across_with_enumeration(self):
        assert classify_query_type("What are the barriers across diagnosis, triage, and treatment?") == "synthesis"

    def test_work_together(self):
        assert classify_query_type("How do diagnostic tools and triage systems work together?") == "synthesis"

    def test_themes_across(self):
        assert classify_query_type("What are the overall themes across technology and history?") == "synthesis"

    def test_differ_between(self):
        assert classify_query_type("How do ethical concerns differ between resource allocation and diagnosis?") == "synthesis"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_string(self):
        result = classify_query_type("")
        assert result in ("specification", "summary", "logic", "synthesis")

    def test_case_insensitive(self):
        assert classify_query_type("LIST all items") == "specification"
        assert classify_query_type("HOW DOES it work") == "logic"
        assert classify_query_type("SUMMARIZE this") == "summary"

    def test_profile_fallback(self):
        from fractrag.profile import DocumentProfile
        profile = DocumentProfile(likely_question_types=["specification"])
        result = classify_query_type("Tell me about this", profile=profile)
        # Profile adds 0.5 but synthesis default is 0 — spec should win with profile boost
        assert result in ("specification", "synthesis")

    def test_returns_valid_type(self):
        for query in ["random text", "????", "a", "the quick brown fox"]:
            result = classify_query_type(query)
            assert result in ("specification", "summary", "logic", "synthesis")

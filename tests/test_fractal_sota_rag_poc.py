"""
Tests for fractal_sota_rag_poc.py (v2 FractalSOTARAG)
Covers: classify_query_type, get_type_weights, FractalSOTARAG class
"""
import sys
import os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from fractal_sota_rag_poc import (
    classify_query_type, get_type_weights, text_to_latent, normalize,
    FractalSOTARAG, DIM, BASE_DERIV_WEIGHT
)


# ============================================================
# classify_query_type UNIT TESTS
# ============================================================
class TestClassifyQueryType:
    """Comprehensive tests for the heuristic query classifier.

    The classifier checks keyword groups in this order:
      1. specification: ["list", "exact", "specific", "detail", "number",
                         "how many", "what is the", "name the"]
      2. specification (long query): >8 words + ["policy", "procedure",
                         "step", "code", "id", "version"]
      3. logic: ["how does", "why", "compare", "difference", "cause",
                 "effect", "what if", "explain the relationship"]
      4. summary: ["summarize", "overview", "briefly", "what is",
                   "main points", "high level"]
      5. synthesis: default

    Because specification is checked first, any query containing "what is the"
    as a substring will match specification even if it also contains logic
    keywords like "difference" or "effect". This is a known limitation of the
    heuristic approach.
    """

    # --- Specification type (line 49: primary keyword match) ---
    def test_specification_list(self):
        assert classify_query_type("List all the factors") == "specification"

    def test_specification_exact(self):
        assert classify_query_type("What is the exact date?") == "specification"

    def test_specification_specific(self):
        assert classify_query_type("Give me specific details") == "specification"

    def test_specification_how_many(self):
        assert classify_query_type("How many people attended?") == "specification"

    def test_specification_name_the(self):
        assert classify_query_type("Name the discoverer of DNA") == "specification"

    def test_specification_long_policy(self):
        assert classify_query_type("What is the company policy on remote work for international employees?") == "specification"

    def test_specification_precedes_logic_what_is_the(self):
        """'What is the difference...' matches 'what is the' (spec) before 'difference' (logic)."""
        assert classify_query_type("What is the difference between mitosis and meiosis?") == "specification"

    def test_specification_precedes_logic_what_is_the_effect(self):
        """'What is the effect...' matches 'what is the' (spec) before 'effect' (logic)."""
        assert classify_query_type("What is the effect of gravity?") == "specification"

    # --- Specification type (line 51-52: long query with domain keywords) ---
    def test_specification_long_procedure_without_primary_keywords(self):
        """Covers line 52: >8 words containing 'procedure' but NO line-49 keywords.
        This query has 10 words, contains 'procedure', and does not contain
        'list', 'exact', 'specific', 'detail', 'number', 'how many',
        'what is the', or 'name the'."""
        query = "Please follow the standard onboarding procedure for all new remote employees carefully"
        assert len(query.lower().split()) > 8
        assert classify_query_type(query) == "specification"

    def test_specification_long_version_without_primary_keywords(self):
        """Covers line 52: >8 words containing 'version' but NO line-49 keywords."""
        query = "Make sure to check which firmware version runs on each production gateway device"
        assert len(query.lower().split()) > 8
        assert classify_query_type(query) == "specification"

    def test_specification_long_code_without_primary_keywords(self):
        """Covers line 52: >8 words containing 'code' but NO line-49 keywords."""
        query = "We need to review the source code changes submitted by the engineering team recently"
        assert len(query.lower().split()) > 8
        assert classify_query_type(query) == "specification"

    # --- Logic type ---
    def test_logic_how_does(self):
        assert classify_query_type("How does machine learning work?") == "logic"

    def test_logic_why(self):
        assert classify_query_type("Why did Rome fall?") == "logic"

    def test_logic_compare(self):
        assert classify_query_type("Compare DNA and RNA structures") == "logic"

    def test_logic_difference_without_what_is_the(self):
        """Queries with 'difference' that do NOT contain 'what is the' go to logic."""
        assert classify_query_type("Explain the difference between mitosis and meiosis") == "logic"

    def test_logic_cause(self):
        assert classify_query_type("What was the cause of the war?") == "logic"

    def test_logic_effect_without_what_is_the(self):
        """Queries with 'effect' that do NOT contain 'what is the' go to logic."""
        assert classify_query_type("Describe the effect of gravity on orbits") == "logic"

    def test_logic_what_if(self):
        assert classify_query_type("What if Rome had survived?") == "logic"

    def test_logic_how_at_start(self):
        assert classify_query_type("How are they connected?") == "logic"

    def test_logic_why_at_start(self):
        assert classify_query_type("Why is this important?") == "logic"

    # --- Summary type ---
    def test_summary_summarize(self):
        assert classify_query_type("Summarize the paper") == "summary"

    def test_summary_overview(self):
        assert classify_query_type("Give an overview") == "summary"

    def test_summary_briefly(self):
        assert classify_query_type("Explain briefly") == "summary"

    def test_summary_what_is(self):
        assert classify_query_type("What is DNA?") == "summary"

    def test_summary_main_points(self):
        assert classify_query_type("What are the main points?") == "summary"

    def test_summary_high_level(self):
        assert classify_query_type("High level explanation please") == "summary"

    def test_summary_short_explain(self):
        assert classify_query_type("Explain DNA") == "summary"

    # --- Synthesis (default) type ---
    def test_synthesis_default(self):
        assert classify_query_type("Integrate the findings across all domains") == "synthesis"

    def test_synthesis_no_keywords(self):
        assert classify_query_type("Tell me about this topic") == "synthesis"

    def test_synthesis_empty_string(self):
        """Empty string should default to synthesis."""
        assert classify_query_type("") == "synthesis"

    # --- Case insensitivity ---
    def test_case_insensitive(self):
        assert classify_query_type("LIST all items") == "specification"
        assert classify_query_type("HOW DOES it work") == "logic"
        assert classify_query_type("SUMMARIZE this") == "summary"


# ============================================================
# get_type_weights UNIT TESTS
# ============================================================
class TestGetTypeWeights:
    """Test that weight configurations are correct per query type."""

    def test_specification_weights(self):
        w = get_type_weights("specification")
        assert len(w["levels"]) == 3
        assert sum(w["levels"]) == pytest.approx(1.0)
        assert w["levels"][2] > w["levels"][0]  # sentence > doc for specification
        assert w["deriv_mult"] > 1.0  # strong derivative emphasis

    def test_summary_weights(self):
        w = get_type_weights("summary")
        assert w["levels"][0] > w["levels"][2]  # doc > sentence for summary
        assert w["deriv_mult"] < 1.0  # weak derivative emphasis

    def test_logic_weights(self):
        w = get_type_weights("logic")
        assert w["deriv_mult"] > 1.0  # strong derivatives for reasoning

    def test_synthesis_weights(self):
        w = get_type_weights("synthesis")
        assert w["deriv_mult"] == pytest.approx(1.0)  # balanced

    def test_all_types_have_k_per_level(self):
        for qtype in ["specification", "summary", "logic", "synthesis"]:
            w = get_type_weights(qtype)
            assert "k_per_level" in w
            assert w["k_per_level"] > 0

    def test_level_weights_sum_to_one(self):
        for qtype in ["specification", "summary", "logic", "synthesis"]:
            w = get_type_weights(qtype)
            assert sum(w["levels"]) == pytest.approx(1.0)


# ============================================================
# FractalSOTARAG INTEGRATION TESTS
# ============================================================
class TestFractalSOTARAGAddDocument:
    """Test document ingestion in v2."""

    def test_adds_all_levels(self, sample_docs):
        rag = FractalSOTARAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        assert len(rag.index[2]) >= 1
        assert len(rag.index[1]) >= 1
        assert len(rag.index[0]) >= 1

    def test_adapter_always_created(self, sample_docs):
        rag = FractalSOTARAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        assert "ai" in rag.adapters

    def test_derivatives_for_all_sentences(self, sample_docs):
        rag = FractalSOTARAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        sent_ids = [item['id'] for item in rag.index[0]]
        for sid in sent_ids:
            assert sid in rag.derivatives


class TestRetrieveAdaptive:
    """Test type-aware adaptive retrieval."""

    def test_returns_results_and_type(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive("How does AI work?")
        assert isinstance(results, dict)
        assert isinstance(qtype, str)
        assert qtype in ["specification", "summary", "logic", "synthesis"]

    def test_explicit_query_type_respected(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive("random text", query_type="specification")
        assert qtype == "specification"

    def test_auto_classification_used(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive("Summarize AI advances")
        assert qtype == "summary"

    def test_results_have_all_levels(self, v2_engine_loaded):
        results, _ = v2_engine_loaded.retrieve_adaptive("test query")
        assert 0 in results
        assert 1 in results
        assert 2 in results

    def test_results_sorted_descending(self, v2_engine_loaded):
        results, _ = v2_engine_loaded.retrieve_adaptive("machine learning")
        for lvl in results:
            scores = [s for _, s in results[lvl]]
            assert scores == sorted(scores, reverse=True)

    def test_specification_returns_more_results(self, v2_engine_loaded):
        """Specification type should request k=4 results per level."""
        results_spec, _ = v2_engine_loaded.retrieve_adaptive("List all factors", query_type="specification")
        results_sum, _ = v2_engine_loaded.retrieve_adaptive("Summary please", query_type="summary")
        # Specification k=4, summary k=2
        total_spec = sum(len(results_spec[lvl]) for lvl in results_spec)
        total_sum = sum(len(results_sum[lvl]) for lvl in results_sum)
        assert total_spec >= total_sum


class TestEvaluateCorrectOutcome:
    """Test type-specific outcome measurement."""

    def test_specification_has_precision_metrics(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive(
            "What is the exact date Rome collapsed?", query_type="specification"
        )
        outcome = v2_engine_loaded.evaluate_correct_outcome(
            "What is the exact date Rome collapsed?", qtype, results
        )
        assert "exact_term_precision" in outcome
        assert "sentence_level_completeness" in outcome

    def test_summary_has_coverage_metric(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive(
            "Summarize DNA", query_type="summary"
        )
        outcome = v2_engine_loaded.evaluate_correct_outcome("Summarize DNA", qtype, results)
        assert "main_idea_coverage" in outcome

    def test_logic_has_relationship_metric(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive(
            "How does AI learn?", query_type="logic"
        )
        outcome = v2_engine_loaded.evaluate_correct_outcome("How does AI learn?", qtype, results)
        assert "relationship_curvature" in outcome

    def test_synthesis_has_balance_metric(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive(
            "Discuss overall themes", query_type="synthesis"
        )
        outcome = v2_engine_loaded.evaluate_correct_outcome("Discuss overall themes", qtype, results)
        assert "fractal_balance" in outcome

    def test_all_outcomes_have_query_type(self, v2_engine_loaded):
        for qt in ["specification", "summary", "logic", "synthesis"]:
            results, qtype = v2_engine_loaded.retrieve_adaptive("test", query_type=qt)
            outcome = v2_engine_loaded.evaluate_correct_outcome("test", qt, results)
            assert outcome["query_type"] == qt

    def test_metrics_are_numeric(self, v2_engine_loaded):
        results, _ = v2_engine_loaded.retrieve_adaptive("How does AI work?", query_type="logic")
        outcome = v2_engine_loaded.evaluate_correct_outcome("How does AI work?", "logic", results)
        for key, value in outcome.items():
            if key != "query_type":
                assert isinstance(value, (int, float)), f"{key} is not numeric: {type(value)}"


class TestGenerateTypeAware:
    """Test type-aware generation output."""

    def test_output_contains_query(self, v2_engine_loaded):
        results, qtype = v2_engine_loaded.retrieve_adaptive("test query")
        output = v2_engine_loaded.generate_type_aware("test query", qtype, results)
        assert "test query" in output

    def test_output_contains_type(self, v2_engine_loaded):
        results, _ = v2_engine_loaded.retrieve_adaptive("List items", query_type="specification")
        output = v2_engine_loaded.generate_type_aware("List items", "specification", results)
        assert "SPECIFICATION" in output

    def test_output_contains_style_instruction(self, v2_engine_loaded):
        results, _ = v2_engine_loaded.retrieve_adaptive("How does it work?", query_type="logic")
        output = v2_engine_loaded.generate_type_aware("How does it work?", "logic", results)
        assert "relationship" in output.lower() or "cause" in output.lower() or "step" in output.lower()

    def test_all_types_generate_without_error(self, v2_engine_loaded):
        for qt in ["specification", "summary", "logic", "synthesis"]:
            results, _ = v2_engine_loaded.retrieve_adaptive("query", query_type=qt)
            output = v2_engine_loaded.generate_type_aware("query", qt, results)
            assert isinstance(output, str)
            assert len(output) > 0

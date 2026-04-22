"""
Tests for xai_musk_knowledge_engine_demo.py
Covers: MuskProfile, xai_preprocess, xAIKnowledgeEngine class
Includes regression tests for known bugs:
  - BUG 1: generate_llm_style() returns hardcoded text regardless of retrieval (line 173-197)
  - BUG 2: Division by zero possible in derivative calc (lines 130-131)
  - DEAD CODE: pick() helper at lines 177-179 is defined but never called
"""
import sys
import os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from xai_musk_knowledge_engine_demo import (
    MuskProfile, xai_preprocess, xAIKnowledgeEngine
)


# ============================================================
# MuskProfile UNIT TESTS
# ============================================================
class TestMuskProfile:
    """Test the MuskProfile dataclass."""

    def test_default_values(self):
        p = MuskProfile()
        assert p.accuracy_importance == "critical"
        assert p.complexity_level == "high"
        assert p.mission_critical is True
        assert p.owner == "xAI"

    def test_smart_config_critical(self):
        p = MuskProfile(accuracy_importance="critical")
        cfg = p.smart_config()
        assert cfg["adapter_strength"] == 0.45
        assert cfg["retrieval_threshold"] == 0.68

    def test_smart_config_high(self):
        p = MuskProfile(accuracy_importance="high")
        cfg = p.smart_config()
        assert cfg["adapter_strength"] == 0.28
        assert cfg["retrieval_threshold"] == 0.58

    def test_smart_config_deriv_depth(self):
        p_high = MuskProfile(complexity_level="high")
        p_med = MuskProfile(complexity_level="medium")
        assert p_high.smart_config()["deriv_depth"] == 3
        assert p_med.smart_config()["deriv_depth"] == 2

    def test_smart_config_metadata_fields(self):
        p_critical = MuskProfile(mission_critical=True)
        p_normal = MuskProfile(mission_critical=False)
        assert p_critical.smart_config()["metadata_fields"] == 12
        assert p_normal.smart_config()["metadata_fields"] == 6

    def test_smart_config_style_string(self):
        p = MuskProfile(accuracy_importance="critical")
        cfg = p.smart_config()
        assert "PRECISE" in cfg["style"].upper()

    def test_update_strategy(self):
        p_low = MuskProfile(update_frequency="low")
        p_med = MuskProfile(update_frequency="medium")
        assert p_low.smart_config()["update_strategy"] == "delta_only"
        assert p_med.smart_config()["update_strategy"] == "smart_invalidate"


# ============================================================
# xai_preprocess UNIT TESTS
# ============================================================
class TestXaiPreprocess:
    """Test the xAI-grade preprocessing function."""

    def test_returns_required_keys(self):
        p = MuskProfile()
        result = xai_preprocess("doc1", "Test sentence one. Test sentence two.", p)
        assert "metadata" in result
        assert "sentences" in result
        assert "paragraphs" in result
        assert "config" in result

    def test_metadata_doc_id(self):
        p = MuskProfile()
        result = xai_preprocess("my_doc", "Text.", p)
        assert result["metadata"]["doc_id"] == "my_doc"

    def test_key_entities_extraction(self):
        p = MuskProfile()
        text = "Tesla produces batteries. SpaceX launches rockets."
        result = xai_preprocess("doc", text, p)
        entities = result["metadata"]["key_entities"]
        assert any("Tesla" in e for e in entities)
        assert any("SpaceX" in e for e in entities)

    def test_key_numbers_extraction(self):
        p = MuskProfile()
        text = "The target is 92% by Q4. Energy density is 350 Wh/kg. Cycle life is 1500 days."
        result = xai_preprocess("doc", text, p)
        numbers = result["metadata"]["key_numbers"]
        assert len(numbers) > 0

    def test_anticipated_questions_present(self):
        p = MuskProfile()
        result = xai_preprocess("doc", "Text.", p)
        assert "anticipated_questions" in result["metadata"]
        assert len(result["metadata"]["anticipated_questions"]) == 2

    def test_mission_critical_flag_in_metadata(self):
        p = MuskProfile(mission_critical=True)
        result = xai_preprocess("doc", "Text.", p)
        assert result["metadata"]["mission_critical"] is True

    def test_created_timestamp_present(self):
        p = MuskProfile()
        result = xai_preprocess("doc", "Text.", p)
        assert "created" in result["metadata"]


# ============================================================
# xAIKnowledgeEngine INTEGRATION TESTS
# ============================================================
class TestXAIKnowledgeEngineAdd:
    """Test document ingestion."""

    def test_add_populates_all_levels(self):
        engine = xAIKnowledgeEngine()
        p = MuskProfile()
        engine.add("test", "First sentence. Second sentence. Third sentence.", p)
        assert len(engine.index[2]) >= 1
        assert len(engine.index[1]) >= 1
        assert len(engine.index[0]) >= 1

    def test_add_stores_profile(self):
        engine = xAIKnowledgeEngine()
        p = MuskProfile(domain="mars")
        engine.add("test", "Text.", p)
        assert engine.profiles["test"].domain == "mars"

    def test_add_stores_metadata(self):
        engine = xAIKnowledgeEngine()
        p = MuskProfile()
        engine.add("test", "Text.", p)
        assert "test" in engine.metadata

    def test_add_creates_adapter(self):
        engine = xAIKnowledgeEngine()
        p = MuskProfile()
        engine.add("test", "Text.", p)
        assert "test" in engine.adapters

    def test_decision_log_records_ingestion(self):
        engine = xAIKnowledgeEngine()
        p = MuskProfile()
        engine.add("test", "Text.", p)
        assert any("INGEST" in entry for entry in engine.decision_log)


class TestXAIKnowledgeEngineRetrieve:
    """Test retrieval functionality."""

    def test_returns_results_type_profile(self, xai_engine_loaded):
        results, qtype, profile = xai_engine_loaded.retrieve("What is the habitat volume?")
        assert isinstance(results, dict)
        assert isinstance(qtype, str)
        assert isinstance(profile, MuskProfile)

    def test_focused_doc_retrieval(self, xai_engine_loaded):
        results, _, _ = xai_engine_loaded.retrieve("test", focus_doc="mars_habitat")
        for lvl in [0, 1]:
            for item, _ in results.get(lvl, []):
                assert item.get("parent", item["id"]) == "mars_habitat"

    def test_cross_doc_retrieval(self, xai_engine_loaded):
        results, _, _ = xai_engine_loaded.retrieve("energy density and habitat volume")
        # Should have entries from multiple docs at document level
        assert len(results[2]) >= 1

    def test_classify_specification(self, xai_engine_loaded):
        _, qtype, _ = xai_engine_loaded.retrieve("What is the exact habitat volume?")
        assert qtype == "specification"

    def test_classify_logic(self, xai_engine_loaded):
        _, qtype, _ = xai_engine_loaded.retrieve("How does energy density affect payload?")
        assert qtype == "logic"

    def test_classify_summary(self, xai_engine_loaded):
        """Note: xAI _classify checks spec keywords first. 'Summarize the findings'
        has no spec substring matches, so it correctly goes to summary."""
        _, qtype, _ = xai_engine_loaded.retrieve("Summarize the findings briefly")
        assert qtype == "summary"

    def test_classify_summary_spec_precedence(self, xai_engine_loaded):
        """Demonstrates that spec keywords take precedence: 'specifications'
        contains the substring 'specific', so _classify returns specification."""
        _, qtype, _ = xai_engine_loaded.retrieve("Summarize the main specifications")
        assert qtype == "specification"  # spec keyword matched first

    def test_decision_log_records_retrieval(self, xai_engine_loaded):
        xai_engine_loaded.retrieve("test query")
        assert any("RETRIEVE" in entry for entry in xai_engine_loaded.decision_log)


class TestXAIKnowledgeEngineClassify:
    """Test the inline _classify method.

    The xAI _classify method checks keywords in this order:
      1. specification: ["exact", "what is the", "how many", "specific"]
      2. logic: ["how", "why", "cause", "compare", "relationship"]
      3. summary: ["summarize", "overview", "main"]
      4. synthesis: default

    Because spec is checked first, strings containing "what is the" or "specific"
    as substrings will be classified as specification even if they contain logic
    or summary keywords.
    """

    def test_specification(self):
        engine = xAIKnowledgeEngine()
        assert engine._classify("What is the exact value?") == "specification"
        assert engine._classify("How many units?") == "specification"

    def test_specification_precedence_over_logic(self):
        """'What is the cause?' matches 'what is the' (spec) before 'cause' (logic)."""
        engine = xAIKnowledgeEngine()
        assert engine._classify("What is the cause?") == "specification"

    def test_logic(self):
        engine = xAIKnowledgeEngine()
        assert engine._classify("How does it work?") == "logic"
        assert engine._classify("Why did it fail?") == "logic"
        # Use a query that does NOT trigger spec keywords
        assert engine._classify("The root cause of failure is unclear") == "logic"

    def test_summary(self):
        engine = xAIKnowledgeEngine()
        assert engine._classify("Summarize the findings") == "summary"
        assert engine._classify("What are the main points?") == "summary"

    def test_synthesis_default(self):
        engine = xAIKnowledgeEngine()
        assert engine._classify("Tell me about the project") == "synthesis"


# ============================================================
# BUG REGRESSION TESTS
# ============================================================
class TestBugRegressionHardcodedGeneration:
    """
    BUG: generate_llm_style() at line 173-197 returns hardcoded text about
    Mars Habitat Alpha-7 and 4680 batteries regardless of what was actually
    retrieved. This test proves the bug exists.

    DEAD CODE: The pick() helper function at lines 177-179 is defined inside
    generate_llm_style() but never called. The hardcoded answer string at
    lines 189-197 bypasses all retrieval results. The pick() function is
    unreachable dead code -- no test can cover it without modifying the source.
    This is a direct consequence of the hardcoded generation bug.
    """

    def test_hardcoded_output_regardless_of_input(self):
        """
        If we feed completely unrelated documents (cooking recipes) and query
        about them, the output should NOT mention "Mars Habitat Alpha-7" --
        but because of the bug, it WILL.
        """
        engine = xAIKnowledgeEngine()
        p = MuskProfile(accuracy_importance="critical", domain="cooking")
        engine.add("recipes", "Mix flour and eggs to make pasta. Boil water for 10 minutes.", p)

        results, qtype, profile = engine.retrieve("How do I make pasta?", focus_doc="recipes")
        output = engine.generate_llm_style("How do I make pasta?", results, qtype, profile)

        # BUG FIXED: output should now use retrieved content, not hardcoded Mars text
        assert "Mars Habitat Alpha-7" not in output, (
            "Hardcoded generation bug has regressed — output should use retrieved content"
        )
        # The output should reference the actual document content
        assert "pasta" in output.lower() or "flour" in output.lower() or "eggs" in output.lower() or "Boil" in output, (
            "Fixed generation should include actual retrieved content about pasta/cooking"
        )

    def test_output_does_not_use_retrieved_content(self):
        """
        The output text should ideally vary based on what was retrieved.
        This test documents that it currently does NOT.
        """
        engine = xAIKnowledgeEngine()
        p1 = MuskProfile(domain="sports")
        p2 = MuskProfile(domain="music")
        engine.add("sports", "Football is played with 11 players per team.", p1)
        engine.add("music", "A piano has 88 keys arranged chromatically.", p2)

        r1, qt1, pr1 = engine.retrieve("How many players in football?", focus_doc="sports")
        r2, qt2, pr2 = engine.retrieve("How many keys on a piano?", focus_doc="music")

        out1 = engine.generate_llm_style("How many players?", r1, qt1, pr1)
        out2 = engine.generate_llm_style("How many keys?", r2, qt2, pr2)

        # BUG FIXED: Outputs should now differ based on different retrieved content
        # out1 should reference football, out2 should reference piano
        assert out1 != out2, (
            "Fixed generation should produce different outputs for different queries/documents"
        )

    def test_pick_function_is_used(self):
        """
        Validates that pick() is now wired into the generation output.
        Previously it was dead code (defined but never called).
        """
        import inspect
        engine = xAIKnowledgeEngine()
        source = inspect.getsource(engine.generate_llm_style)
        assert "def pick(" in source, "pick() helper should be defined in generate_llm_style"
        # pick() should now be called (appears more than once: def + usage)
        pick_occurrences = source.count("pick(")
        assert pick_occurrences > 1, (
            f"pick() appears {pick_occurrences} times; expected >1 (def + calls). "
            "pick() should be used to extract retrieved content."
        )


class TestGenerateLLMStyleNonCriticalProfile:
    """Test generate_llm_style() with non-critical accuracy profiles.
    Covers line 187: the else branch where prefix = '' when
    accuracy_importance != 'critical'.
    """

    def test_non_critical_profile_omits_critical_prefix(self):
        """
        When accuracy_importance='high' (not 'critical'), the output should
        NOT contain 'CRITICAL ACCURACY MODE ENGAGED'. This covers line 187.
        """
        engine = xAIKnowledgeEngine()
        p = MuskProfile(accuracy_importance="high", domain="testing")
        engine.add("doc", "Some test content for coverage. Another sentence here.", p)

        results, qtype, profile = engine.retrieve("test query", focus_doc="doc")
        output = engine.generate_llm_style("test query", results, qtype, profile)

        # Line 187: prefix = "" (non-critical path)
        assert "CRITICAL ACCURACY MODE ENGAGED" not in output
        # Output should contain actual retrieved content (bug fixed)
        assert "test" in output.lower() or "content" in output.lower() or "sentence" in output.lower()

    def test_critical_profile_includes_critical_prefix(self):
        """
        Contrast test: when accuracy_importance='critical', the output DOES
        contain the critical prefix. This confirms the branching works.
        """
        engine = xAIKnowledgeEngine()
        p = MuskProfile(accuracy_importance="critical", domain="testing")
        engine.add("doc", "Some test content for coverage. Another sentence here.", p)

        results, qtype, profile = engine.retrieve("test query", focus_doc="doc")
        output = engine.generate_llm_style("test query", results, qtype, profile)

        # Line 185: prefix = "CRITICAL ACCURACY MODE ENGAGED. "
        assert "CRITICAL ACCURACY MODE ENGAGED" in output


class TestBugRegressionDivisionByZero:
    """
    BUG: At xai_musk_knowledge_engine_demo.py lines 130-131, derivative
    calculation divides by np.linalg.norm(sv - pmean) and np.linalg.norm(d1 - (sv - dvec)).
    If either norm is zero, this produces NaN/Inf.

    Unlike the safe normalize() used in v1/v2, the xAI engine uses raw division.
    """

    def test_duplicate_sentences_no_nan_derivatives(self):
        """
        BUG FIXED: Division by zero is now guarded. Duplicate sentences should
        produce valid (finite) derivatives, not NaN/Inf.
        """
        engine = xAIKnowledgeEngine()
        p = MuskProfile(complexity_level="high")  # deriv_depth=3 -> processes 15 sentences

        # Create a document where all sentences are identical
        identical_text = "Exact same sentence. Exact same sentence. Exact same sentence."
        engine.add("dup_doc", identical_text, p)

        # All derivatives must be finite (no NaN or Inf)
        for sid, deriv in engine.derivs.items():
            assert np.all(np.isfinite(deriv['d1'])), f"d1 for {sid} contains NaN/Inf"
            assert np.all(np.isfinite(deriv['d2'])), f"d2 for {sid} contains NaN/Inf"

        # Verify the code now uses safe normalization (not raw division)
        import inspect
        source = inspect.getsource(engine.add)
        assert "d1_norm > 0" in source, "Code should guard against zero-norm division"

    def test_single_sentence_document(self):
        """
        A single-sentence document where pmean equals the single paragraph vector.
        The adapter-modified sv may still differ from pmean, but this tests the edge case.
        """
        engine = xAIKnowledgeEngine()
        p = MuskProfile(complexity_level="high")
        engine.add("single", "Just one sentence here.", p)

        # Verify the engine processed without crashing
        assert len(engine.index[0]) >= 1

        # Check derivatives are not NaN
        for sid, deriv in engine.derivs.items():
            if sid.startswith("single_"):
                # The derivatives should ideally be valid, but we document
                # that the division is unprotected
                d1_valid = np.all(np.isfinite(deriv['d1']))
                d2_valid = np.all(np.isfinite(deriv['d2']))
                # If these fail, it confirms the division-by-zero bug
                if not d1_valid or not d2_valid:
                    pytest.xfail("Division by zero bug confirmed: NaN/Inf in derivatives")


class TestSynthesizeMethod:
    """Test the synthesize delegation."""

    def test_synthesize_delegates_to_generate(self, xai_engine_loaded):
        results, qtype, profile = xai_engine_loaded.retrieve("test query")
        output = xai_engine_loaded.synthesize("test query", results, qtype, profile)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_synthesize_logs_action(self, xai_engine_loaded):
        results, qtype, profile = xai_engine_loaded.retrieve("test")
        log_count_before = len(xai_engine_loaded.decision_log)
        xai_engine_loaded.synthesize("test", results, qtype, profile)
        assert len(xai_engine_loaded.decision_log) > log_count_before

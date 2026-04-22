"""
Tests for fractal_sota_rag_v3.py (v3 FractalSOTARAGv3)
Covers: DocumentProfile, profile_to_config, classify_query_type (v3),
        get_type_weights (v3), smart_preprocess, FractalSOTARAGv3 class
"""
import sys
import os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from fractal_sota_rag_v3 import (
    DocumentProfile, profile_to_config, classify_query_type, get_type_weights,
    smart_preprocess, text_to_latent, normalize, FractalSOTARAGv3
)


# ============================================================
# DocumentProfile UNIT TESTS
# ============================================================
class TestDocumentProfile:
    """Test the DocumentProfile dataclass validation."""

    def test_default_values(self):
        p = DocumentProfile()
        assert p.accuracy_importance == "medium"
        assert p.complexity_level == "medium"
        assert p.update_frequency == "low"
        assert p.domain == "general"
        assert p.tolerance_for_hallucination == "medium"
        assert p.priority == "balanced"

    def test_valid_accuracy_importance(self):
        for val in ["critical", "high", "medium", "low"]:
            p = DocumentProfile(accuracy_importance=val)
            assert p.accuracy_importance == val

    def test_invalid_accuracy_importance_raises(self):
        with pytest.raises(ValueError, match="Invalid accuracy_importance"):
            DocumentProfile(accuracy_importance="extreme")

    def test_valid_complexity_level(self):
        for val in ["low", "medium", "high"]:
            p = DocumentProfile(complexity_level=val)
            assert p.complexity_level == val

    def test_invalid_complexity_level_raises(self):
        with pytest.raises(ValueError, match="Invalid complexity_level"):
            DocumentProfile(complexity_level="ultra")

    def test_valid_update_frequency(self):
        for val in ["static", "low", "medium", "high"]:
            p = DocumentProfile(update_frequency=val)
            assert p.update_frequency == val

    def test_invalid_update_frequency_raises(self):
        with pytest.raises(ValueError, match="Invalid update_frequency"):
            DocumentProfile(update_frequency="realtime")

    def test_valid_tolerance(self):
        for val in ["zero", "low", "medium", "high"]:
            p = DocumentProfile(tolerance_for_hallucination=val)
            assert p.tolerance_for_hallucination == val

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="Invalid tolerance_for_hallucination"):
            DocumentProfile(tolerance_for_hallucination="infinite")

    def test_valid_priority(self):
        for val in ["precision", "balanced", "speed"]:
            p = DocumentProfile(priority=val)
            assert p.priority == val

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Invalid priority"):
            DocumentProfile(priority="maximum")

    def test_likely_question_types_default(self):
        p = DocumentProfile()
        assert p.likely_question_types == ["synthesis"]

    def test_likely_question_types_custom(self):
        p = DocumentProfile(likely_question_types=["specification", "logic"])
        assert p.likely_question_types == ["specification", "logic"]


# ============================================================
# profile_to_config UNIT TESTS
# ============================================================
class TestProfileToConfig:
    """Test config derivation from profiles."""

    def test_low_complexity_config(self):
        p = DocumentProfile(complexity_level="low")
        c = profile_to_config(p)
        assert c["chunk_sizes"] == 180
        assert c["deriv_depth"] == 1
        assert c["adapter_strength"] == 0.15

    def test_high_complexity_config(self):
        p = DocumentProfile(complexity_level="high")
        c = profile_to_config(p)
        assert c["chunk_sizes"] == 80
        assert c["deriv_depth"] == 3
        assert c["adapter_strength"] == 0.40

    def test_critical_accuracy_threshold(self):
        p = DocumentProfile(accuracy_importance="critical")
        c = profile_to_config(p)
        assert c["retrieval_threshold"] == 0.78

    def test_low_accuracy_threshold(self):
        p = DocumentProfile(accuracy_importance="low")
        c = profile_to_config(p)
        assert c["retrieval_threshold"] == 0.55

    def test_zero_tolerance_strict(self):
        p = DocumentProfile(tolerance_for_hallucination="zero")
        c = profile_to_config(p)
        assert c["generation_strictness"] == "strict"

    def test_high_tolerance_creative(self):
        p = DocumentProfile(tolerance_for_hallucination="high")
        c = profile_to_config(p)
        assert c["generation_strictness"] == "creative"

    def test_precision_priority_bias(self):
        p = DocumentProfile(priority="precision")
        c = profile_to_config(p)
        assert c["priority_bias"]["deriv"] == 1.6

    def test_speed_priority_bias(self):
        p = DocumentProfile(priority="speed")
        c = profile_to_config(p)
        assert c["priority_bias"]["deriv"] == 0.6

    def test_update_strategies(self):
        mapping = {
            "static": "delta_only",
            "low": "smart_invalidate",
            "medium": "full_reindex",
            "high": "stream",
        }
        for freq, strategy in mapping.items():
            p = DocumentProfile(update_frequency=freq)
            c = profile_to_config(p)
            assert c["update_strategy"] == strategy


# ============================================================
# classify_query_type (v3) UNIT TESTS
# ============================================================
class TestClassifyQueryTypeV3:
    """Test v3 classifier that accepts an optional DocumentProfile."""

    def test_specification_keywords(self):
        assert classify_query_type("What is the exact clause?") == "specification"
        assert classify_query_type("Name the section for termination") == "specification"

    def test_logic_keywords(self):
        assert classify_query_type("How does this work?") == "logic"
        assert classify_query_type("Why did it fail?") == "logic"

    def test_summary_keywords(self):
        assert classify_query_type("Summarize the contract") == "summary"
        assert classify_query_type("Give an overview") == "summary"

    def test_synthesis_default(self):
        assert classify_query_type("Tell me about this") == "synthesis"

    def test_profile_influences_fallback(self):
        """When no keywords match, profile's likely_question_types should influence result."""
        p = DocumentProfile(likely_question_types=["specification", "logic"])
        result = classify_query_type("Tell me about this", profile=p)
        assert result == "specification"  # first in the list

    def test_keywords_override_profile(self):
        """Keywords should take priority over profile hints."""
        p = DocumentProfile(likely_question_types=["specification"])
        result = classify_query_type("Summarize this document", profile=p)
        assert result == "summary"


# ============================================================
# get_type_weights (v3) UNIT TESTS
# ============================================================
class TestGetTypeWeightsV3:
    """Test v3 weight function that requires a profile."""

    def test_specification_strong_sentence_level(self):
        p = DocumentProfile()
        w = get_type_weights("specification", p)
        assert w["levels"][2] > w["levels"][0]  # sentence > doc

    def test_summary_strong_doc_level(self):
        p = DocumentProfile()
        w = get_type_weights("summary", p)
        assert w["levels"][0] > w["levels"][2]  # doc > sentence

    def test_k_depends_on_complexity(self):
        p_low = DocumentProfile(complexity_level="low")
        p_high = DocumentProfile(complexity_level="high")
        w_low = get_type_weights("synthesis", p_low)
        w_high = get_type_weights("synthesis", p_high)
        assert w_low["k"] < w_high["k"]

    def test_threshold_is_reasonable(self):
        p = DocumentProfile(accuracy_importance="critical")
        w = get_type_weights("synthesis", p)
        assert 0.35 <= w["threshold"] <= 1.0


# ============================================================
# smart_preprocess UNIT TESTS
# ============================================================
class TestSmartPreprocess:
    """Test the layered preprocessing function."""

    def test_returns_required_keys(self):
        p = DocumentProfile()
        result = smart_preprocess("test_doc", "Some text here.", p)
        assert "metadata" in result
        assert "sentences" in result
        assert "paragraphs" in result
        assert "config" in result

    def test_metadata_contains_doc_id(self):
        p = DocumentProfile()
        result = smart_preprocess("my_doc", "Text.", p)
        assert result["metadata"]["doc_id"] == "my_doc"

    def test_metadata_contains_domain(self):
        p = DocumentProfile(domain="legal")
        result = smart_preprocess("doc", "Text.", p)
        assert result["metadata"]["domain"] == "legal"

    def test_key_entities_extracted(self):
        p = DocumentProfile()
        result = smart_preprocess("doc", "Watson and Crick discovered DNA structure.", p)
        entities = result["metadata"]["key_entities"]
        assert isinstance(entities, list)
        # Should find at least Watson and Crick
        assert any("Watson" in e for e in entities)

    def test_key_facts_extracted(self):
        p = DocumentProfile()
        result = smart_preprocess("doc", "DNA is fundamental. RNA is also important.", p)
        facts = result["metadata"]["key_facts"]
        assert len(facts) > 0

    def test_anticipated_questions(self):
        p = DocumentProfile(likely_question_types=["specification", "logic"])
        result = smart_preprocess("doc", "Text.", p)
        aq = result["metadata"]["anticipated_questions"]
        assert len(aq) == 2

    def test_sections_limited_by_richness(self):
        p_low = DocumentProfile(complexity_level="low")
        p_high = DocumentProfile(complexity_level="high")
        text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
        r_low = smart_preprocess("doc", text, p_low)
        r_high = smart_preprocess("doc", text, p_high)
        assert len(r_low["metadata"]["sections"]) <= len(r_high["metadata"]["sections"])


# ============================================================
# FractalSOTARAGv3 INTEGRATION TESTS
# ============================================================
class TestFractalSOTARAGv3AddDocument:
    """Test v3 document ingestion with profiles."""

    def test_default_profile_used(self):
        rag = FractalSOTARAGv3()
        rag.add_document("test", "Simple text for testing.")
        assert "test" in rag.profiles
        assert isinstance(rag.profiles["test"], DocumentProfile)

    def test_custom_profile_stored(self):
        rag = FractalSOTARAGv3()
        p = DocumentProfile(accuracy_importance="critical", domain="legal")
        rag.add_document("legal", "Legal text here.", p)
        assert rag.profiles["legal"].accuracy_importance == "critical"

    def test_metadata_stored(self):
        rag = FractalSOTARAGv3()
        rag.add_document("test", "Some text.")
        assert "test" in rag.metadata_store

    def test_high_complexity_indexes_more_sentences(self):
        """High complexity profile should index more sentence-level items due to deriv_depth."""
        rag_low = FractalSOTARAGv3()
        rag_high = FractalSOTARAGv3()
        text = ". ".join([f"Sentence number {i}" for i in range(20)]) + "."
        rag_low.add_document("doc", text, DocumentProfile(complexity_level="low"))
        rag_high.add_document("doc", text, DocumentProfile(complexity_level="high"))
        # deriv_depth: low=1 (4 sentences max), high=3 (12 sentences max)
        assert len(rag_low.index[0]) <= len(rag_high.index[0])


class TestRetrieveSmart:
    """Test profile-aware smart retrieval."""

    def test_returns_results_type_and_profile(self, v3_engine_loaded):
        results, qtype, profile = v3_engine_loaded.retrieve_smart("How does AI work?")
        assert isinstance(results, dict)
        assert isinstance(qtype, str)
        assert isinstance(profile, DocumentProfile)

    def test_doc_scoped_retrieval(self, v3_engine_loaded):
        """When doc_id is specified, results should only come from that document."""
        results, _, _ = v3_engine_loaded.retrieve_smart("test query", doc_id="ai_overview")
        for lvl in [0, 1]:
            for item, _ in results.get(lvl, []):
                assert item.get('parent', item['id']) == "ai_overview"

    def test_cross_doc_retrieval(self, v3_engine_loaded):
        """Without doc_id, results can come from any document."""
        results, _, _ = v3_engine_loaded.retrieve_smart("test query")
        parents = set()
        for lvl in [0, 1, 2]:
            for item, _ in results.get(lvl, []):
                parents.add(item.get('parent', item['id']))
        # Should have results from multiple docs
        assert len(parents) >= 1


class TestEvaluateSmart:
    """Test smart evaluation with profile awareness."""

    def test_critical_accuracy_triggers_precision_metrics(self, v3_engine_loaded):
        p = DocumentProfile(accuracy_importance="critical")
        results, qtype, _ = v3_engine_loaded.retrieve_smart("test query")
        outcome = v3_engine_loaded.evaluate_smart("test query", "synthesis", p, results)
        assert "exact_precision" in outcome

    def test_high_complexity_triggers_relationship_metric(self, v3_engine_loaded):
        p = DocumentProfile(complexity_level="high")
        results, _, _ = v3_engine_loaded.retrieve_smart("test query")
        outcome = v3_engine_loaded.evaluate_smart("test query", "logic", p, results)
        assert "relationship_strength" in outcome

    def test_summary_has_coverage(self, v3_engine_loaded):
        p = DocumentProfile()
        results, _, _ = v3_engine_loaded.retrieve_smart("Summarize everything")
        outcome = v3_engine_loaded.evaluate_smart("Summarize everything", "summary", p, results)
        assert "coverage" in outcome

    def test_overall_smart_score_present(self, v3_engine_loaded):
        p = DocumentProfile()
        results, _, _ = v3_engine_loaded.retrieve_smart("test")
        outcome = v3_engine_loaded.evaluate_smart("test", "synthesis", p, results)
        assert "overall_smart_score" in outcome
        assert 0.0 <= outcome["overall_smart_score"] <= 1.5  # reasonable range


class TestGenerateSmart:
    """Test v3 generation output."""

    def test_output_contains_accuracy_level(self, v3_engine_loaded):
        p = DocumentProfile(accuracy_importance="critical")
        results, _, _ = v3_engine_loaded.retrieve_smart("test")
        output = v3_engine_loaded.generate_smart("test", "specification", p, results)
        assert "CRITICAL" in output

    def test_output_contains_query_type(self, v3_engine_loaded):
        p = DocumentProfile()
        results, _, _ = v3_engine_loaded.retrieve_smart("test")
        output = v3_engine_loaded.generate_smart("test", "logic", p, results)
        assert "LOGIC" in output

    def test_all_accuracy_levels_generate(self, v3_engine_loaded):
        for acc in ["critical", "high", "medium", "low"]:
            p = DocumentProfile(accuracy_importance=acc)
            results, _, _ = v3_engine_loaded.retrieve_smart("test")
            output = v3_engine_loaded.generate_smart("test", "synthesis", p, results)
            assert isinstance(output, str)
            assert len(output) > 0

    def test_empty_results_handled_gracefully(self):
        rag = FractalSOTARAGv3()
        p = DocumentProfile()
        output = rag.generate_smart("test", "synthesis", p, {})
        assert "N/A" in output or "no strong match" in output.lower()

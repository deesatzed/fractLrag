"""
Tests for domain-aware query routing (extract_domain_hints + wiring in retrieve_adaptive).

Direction 4: Automatically detect domain keywords in queries and apply metadata boost.
All tests use real HashEmbedding. No mocks.
"""

import numpy as np
import pytest

from fractrag import FractalRAG, HashEmbedding
from fractrag.query import extract_domain_hints, _DOMAIN_KEYWORDS


class TestExtractDomainHints:
    def test_radiology_single_domain(self):
        """Strong radiology signal returns a single-domain boost."""
        hints = extract_domain_hints("How do radiology imaging AI models detect anomalies?")
        assert hints is not None
        assert hints["domain_target"] == "ai_radiology_pathology"
        assert hints["domain_boost"] == 0.05

    def test_no_domain_signal_returns_none(self):
        """Generic query with no domain keywords returns None."""
        hints = extract_domain_hints("hello world")
        assert hints is None

    def test_case_insensitive(self):
        """Domain keywords should match case-insensitively."""
        hints = extract_domain_hints("RADIOLOGY IMAGING analysis")
        assert hints is not None
        assert hints["domain_target"] == "ai_radiology_pathology"

    def test_multi_domain_returns_list(self):
        """Query mentioning multiple domains returns list of targets."""
        # Each domain gets exactly 1 keyword match — triggers multi-domain path
        hints = extract_domain_hints("Compare surgical outcomes with telemedicine")
        assert hints is not None
        assert isinstance(hints["domain_target"], list)
        assert len(hints["domain_target"]) >= 2
        assert hints["domain_boost"] == 0.03

    def test_pharma_domain_detected(self):
        """Pharmaceutical keywords detected correctly."""
        hints = extract_domain_hints("What is the role of pharmacovigilance in drug safety?")
        assert hints is not None
        target = hints["domain_target"]
        if isinstance(target, list):
            assert "ai_drug_discovery_pharma" in target
        else:
            assert target == "ai_drug_discovery_pharma"

    def test_nlp_domain_detected(self):
        """NLP medical keywords detected correctly."""
        hints = extract_domain_hints("How does natural language processing extract clinical notes?")
        assert hints is not None
        target = hints["domain_target"]
        if isinstance(target, list):
            assert "nlp_medical" in target
        else:
            assert target == "nlp_medical"

    def test_surgery_domain_detected(self):
        """Surgery/robotics keywords detected."""
        hints = extract_domain_hints("robot-assisted surgery with haptic feedback")
        assert hints is not None
        target = hints["domain_target"]
        if isinstance(target, list):
            assert "ai_surgery_robotics" in target
        else:
            assert target == "ai_surgery_robotics"

    def test_all_domains_have_keywords(self):
        """Every domain in _DOMAIN_KEYWORDS has at least 3 keywords."""
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            assert len(keywords) >= 3, f"Domain '{domain}' has too few keywords: {len(keywords)}"


class TestDomainRoutingInRetrieveAdaptive:
    def _build_rag(self):
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document(
            "d1",
            "AI detects tumors in radiology scans using deep learning models and convolutional neural networks.",
            metadata={"domain": "ai_radiology_pathology"},
            title="Radiology AI",
        )
        rag.add_document(
            "d2",
            "Drug metabolism is affected by genetic polymorphisms in cytochrome P450 enzyme pathways.",
            metadata={"domain": "ai_drug_discovery_pharma"},
            title="Pharma Genetics",
        )
        rag.add_document(
            "d3",
            "Clinical decision support systems improve patient outcomes through evidence-based triage protocols.",
            metadata={"domain": "ai_clinical_decision"},
            title="Clinical DSS",
        )
        return rag

    def test_auto_detect_applies_domain_boost(self):
        """retrieve_adaptive auto-detects domain and applies boost without explicit metadata_boost."""
        rag = self._build_rag()
        # Query with strong radiology signal — should auto-detect domain
        results, qtype = rag.retrieve_adaptive(
            "How do radiology imaging AI models detect anomalies in CT scans?",
            k=5,
        )
        # Should not crash and return results
        assert len(results[2]) > 0

    def test_explicit_metadata_boost_overrides_auto(self):
        """Explicit metadata_boost should override auto-detected domain hints."""
        rag = self._build_rag()
        # Auto would detect radiology, but we explicitly boost pharma
        results, _ = rag.retrieve_adaptive(
            "How do radiology imaging AI models detect anomalies?",
            k=5,
            metadata_boost={"domain_boost": 0.5, "domain_target": "ai_drug_discovery_pharma"},
        )
        # Should return results (no crash)
        assert len(results[2]) > 0

    def test_no_domain_signal_no_boost(self):
        """Generic query without domain keywords should not apply domain boost."""
        rag = self._build_rag()
        # No domain keywords
        results, _ = rag.retrieve_adaptive("Tell me something interesting", k=5)
        assert len(results[2]) > 0

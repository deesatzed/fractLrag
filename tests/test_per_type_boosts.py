"""
Tests for per-type adaptive boosts in FractalRAG.

Direction 2: Each query type gets different boost parameters, including
diversity_boost for synthesis queries. All tests use real HashEmbedding. No mocks.
"""

import numpy as np
import pytest

from fractrag import FractalRAG, HashEmbedding
from fractrag.engine import FractalRAG as EngineRAG
from fractrag.benchmarks.cross_validation import (
    tune_boost_params_per_type,
    PARA_GRID,
    SENT_GRID,
    DERIV_GRID,
)


def _make_rag(dim=64):
    return FractalRAG(backend=HashEmbedding(dim=dim))


class TestTypeBoostsConfig:
    def test_all_four_types_present(self):
        """_TYPE_BOOSTS has entries for all 4 query types."""
        boosts = EngineRAG._TYPE_BOOSTS
        assert "specification" in boosts
        assert "summary" in boosts
        assert "logic" in boosts
        assert "synthesis" in boosts

    def test_types_have_different_values(self):
        """At least some types have different boost values (not all identical)."""
        boosts = EngineRAG._TYPE_BOOSTS
        values = [tuple(sorted(v.items())) for v in boosts.values()]
        assert len(set(values)) > 1, "All types have identical boosts"

    def test_specification_is_all_zero(self):
        """Specification queries use all-zero boosts (flat retrieval)."""
        boosts = EngineRAG._TYPE_BOOSTS["specification"]
        assert boosts["para_boost"] == 0.0
        assert boosts["sent_boost"] == 0.0
        assert boosts["deriv_boost"] == 0.0
        assert boosts["diversity_boost"] == 0.0

    def test_synthesis_has_diversity_boost(self):
        """Synthesis queries have diversity_boost > 0."""
        boosts = EngineRAG._TYPE_BOOSTS["synthesis"]
        assert boosts["diversity_boost"] > 0

    def test_logic_has_high_deriv_boost(self):
        """Logic queries have higher deriv_boost than summary."""
        logic = EngineRAG._TYPE_BOOSTS["logic"]
        summary = EngineRAG._TYPE_BOOSTS["summary"]
        assert logic["deriv_boost"] >= summary["deriv_boost"]

    def test_diversity_boost_only_for_synthesis(self):
        """Only synthesis should have non-zero diversity_boost."""
        boosts = EngineRAG._TYPE_BOOSTS
        for qtype in ["specification", "summary", "logic"]:
            assert boosts[qtype]["diversity_boost"] == 0.0
        assert boosts["synthesis"]["diversity_boost"] > 0.0


class TestRetrieveAdaptivePerType:
    def test_passes_correct_boosts_per_type(self):
        """retrieve_adaptive should use type-specific boosts."""
        rag = _make_rag()
        rag.add_document(
            "d1",
            "AI detects tumors in radiology scans using deep learning models. "
            "Neural networks classify medical images. "
            "Transfer learning boosts performance. "
            "Computer vision aids diagnostics.",
            metadata={"domain": "radiology"},
        )
        rag.add_document(
            "d2",
            "Drug metabolism varies with CYP450 enzyme genetics. "
            "Pharmacogenomics predicts drug responses. "
            "Adverse reactions are monitored. "
            "Clinical trials assess safety.",
            metadata={"domain": "pharma"},
        )

        # Specification query → should use flat (all-zero boosts)
        results_spec, qtype_spec = rag.retrieve_adaptive(
            "What specific retinal imaging technique does the study use?",
            k=5,
        )
        assert qtype_spec == "specification"

        # Synthesis query → should use reranked with diversity
        results_synth, qtype_synth = rag.retrieve_adaptive(
            "How do both radiology AI and pharmacovigilance work together across different clinical settings?",
            k=5,
        )
        assert qtype_synth == "synthesis"


class TestPerTypeTuning:
    def _make_queries(self, n_per_type=4):
        types = ["specification", "summary", "logic", "synthesis"]
        queries = []
        for qtype in types:
            for i in range(n_per_type):
                queries.append({
                    "query_text": f"Test {qtype} query {i} about medical AI topic",
                    "query_type": qtype,
                    "relevant_pmids": [f"{1000 + len(types) * i + types.index(qtype)}"],
                })
        return queries

    def _make_rag_with_docs(self, n_docs=20, dim=64):
        rag = FractalRAG(backend=HashEmbedding(dim=dim))
        for i in range(n_docs):
            doc_id = f"pmid_{1000 + i}"
            text = (
                f"Document {i} about topic {i % 5}. "
                f"Medical AI in domain {i % 3}. "
                f"Deep learning methods compared. "
                f"Results show significant improvements."
            )
            rag.add_document(doc_id, text)
        return rag

    def test_per_type_tuning_returns_all_types(self):
        """tune_boost_params_per_type returns params for each query type."""
        rag = self._make_rag_with_docs(10)
        queries = self._make_queries(4)
        result = tune_boost_params_per_type(rag, queries, k=5)
        assert "specification" in result
        assert "summary" in result
        assert "logic" in result
        assert "synthesis" in result

    def test_per_type_params_are_valid_grid_values(self):
        """Per-type tuned params must be from the grid."""
        rag = self._make_rag_with_docs(10)
        queries = self._make_queries(4)
        result = tune_boost_params_per_type(rag, queries, k=5)
        for qtype, params in result.items():
            assert params["para_boost"] in PARA_GRID, f"{qtype} para_boost not in grid"
            assert params["sent_boost"] in SENT_GRID, f"{qtype} sent_boost not in grid"
            assert params["deriv_boost"] in DERIV_GRID, f"{qtype} deriv_boost not in grid"

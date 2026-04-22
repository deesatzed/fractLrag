"""
Tests for fractal_latent_rag_poc.py (v1 FractalLatentRAG)
Covers: text_to_latent, normalize, FractalLatentRAG class
"""
import sys
import os
import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))

from fractal_latent_rag_poc import text_to_latent, normalize, FractalLatentRAG, DIM


# ============================================================
# text_to_latent UNIT TESTS
# ============================================================
class TestTextToLatent:
    """Unit tests for the deterministic embedding function."""

    def test_determinism_same_input_same_output(self):
        """Same text must always produce identical vectors."""
        v1 = text_to_latent("hello world")
        v2 = text_to_latent("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_different_input_different_output(self):
        """Different texts must produce different vectors."""
        v1 = text_to_latent("hello world")
        v2 = text_to_latent("goodbye world")
        assert not np.allclose(v1, v2), "Different inputs produced identical vectors"

    def test_output_dimension_default(self):
        """Default dimension should match DIM constant."""
        vec = text_to_latent("test")
        assert vec.shape == (DIM,)

    def test_output_dimension_custom(self):
        """Custom dimension parameter should be respected."""
        for dim in [16, 32, 128, 256]:
            vec = text_to_latent("test", dim=dim)
            assert vec.shape == (dim,), f"Expected dim {dim}, got {vec.shape}"

    def test_output_is_unit_norm(self):
        """Output vector should be normalized to unit length."""
        vec = text_to_latent("any text at all")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Norm is {norm}, expected 1.0"

    def test_output_dtype_float32(self):
        """Output should be float32."""
        vec = text_to_latent("type check")
        assert vec.dtype == np.float32

    def test_empty_string(self):
        """Empty string should still produce a valid unit vector."""
        vec = text_to_latent("")
        assert vec.shape == (DIM,)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_very_long_string(self):
        """Long string should produce a valid unit vector."""
        long_text = "word " * 10000
        vec = text_to_latent(long_text)
        assert vec.shape == (DIM,)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_unicode_text(self):
        """Unicode text should produce a valid unit vector."""
        vec = text_to_latent("Schrodinger's Katze")
        assert vec.shape == (DIM,)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_whitespace_sensitivity(self):
        """Different whitespace should produce different vectors."""
        v1 = text_to_latent("hello world")
        v2 = text_to_latent("hello  world")
        assert not np.allclose(v1, v2)


# ============================================================
# normalize UNIT TESTS
# ============================================================
class TestNormalize:
    """Unit tests for the normalize utility."""

    def test_normalizes_to_unit_length(self):
        vec = np.array([3.0, 4.0])
        result = normalize(vec)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-7

    def test_zero_vector_returns_zero(self):
        """Zero vector should return zero vector (not NaN or error)."""
        vec = np.zeros(64)
        result = normalize(vec)
        np.testing.assert_array_equal(result, vec)

    def test_already_normalized(self):
        vec = np.array([1.0, 0.0, 0.0])
        result = normalize(vec)
        np.testing.assert_allclose(result, vec)

    def test_negative_values(self):
        vec = np.array([-3.0, -4.0])
        result = normalize(vec)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-7

    def test_large_values(self):
        vec = np.array([1e10, 1e10])
        result = normalize(vec)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-7


# ============================================================
# FractalLatentRAG UNIT + INTEGRATION TESTS
# ============================================================
class TestFractalLatentRAGInit:
    """Test initialization state."""

    def test_default_dim(self):
        rag = FractalLatentRAG()
        assert rag.dim == DIM

    def test_custom_dim(self):
        rag = FractalLatentRAG(dim=128)
        assert rag.dim == 128

    def test_empty_index_on_init(self):
        rag = FractalLatentRAG()
        assert len(rag.index[0]) == 0
        assert len(rag.index[1]) == 0
        assert len(rag.index[2]) == 0

    def test_empty_docs_on_init(self):
        rag = FractalLatentRAG()
        assert len(rag.docs) == 0
        assert len(rag.adapters) == 0
        assert len(rag.derivatives) == 0


class TestChunkFractal:
    """Test the fractal chunking logic."""

    def test_sentence_splitting(self):
        rag = FractalLatentRAG()
        sentences, paragraphs, full = rag._chunk_fractal("Hello world. Goodbye world.")
        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "Goodbye world."

    def test_paragraph_splitting(self):
        rag = FractalLatentRAG()
        text = "Para one.\n\nPara two."
        sentences, paragraphs, full = rag._chunk_fractal(text)
        assert len(paragraphs) == 2

    def test_no_paragraph_breaks_fallback(self):
        rag = FractalLatentRAG()
        text = "Single paragraph with no breaks."
        sentences, paragraphs, full = rag._chunk_fractal(text)
        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_full_text_returned(self):
        rag = FractalLatentRAG()
        text = "Original text."
        _, _, full = rag._chunk_fractal(text)
        assert full == text

    def test_empty_text(self):
        rag = FractalLatentRAG()
        sentences, paragraphs, full = rag._chunk_fractal("")
        assert len(sentences) == 0
        assert full == ""


class TestAddDocument:
    """Test document indexing pipeline."""

    def test_adds_to_all_levels(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        assert len(rag.index[2]) == 1  # 1 document
        assert len(rag.index[1]) >= 1  # at least 1 paragraph
        assert len(rag.index[0]) >= 1  # at least 1 sentence

    def test_doc_stored_in_docs_dict(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        assert "ai" in rag.docs
        assert rag.docs["ai"] == sample_docs["ai_overview"]

    def test_adapter_created(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        assert "ai" in rag.adapters
        assert rag.adapters["ai"].shape == (DIM,)

    def test_adapter_not_created_when_disabled(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"], learn_adapter=False)
        assert "ai" not in rag.adapters

    def test_derivatives_computed_for_sentences(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        # Derivatives should exist for each sentence
        sentence_ids = [item['id'] for item in rag.index[0]]
        for sid in sentence_ids:
            assert sid in rag.derivatives
            assert 'd1' in rag.derivatives[sid]
            assert 'd2' in rag.derivatives[sid]

    def test_derivative_vectors_have_correct_dim(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        for sid, deriv in rag.derivatives.items():
            assert deriv['d1'].shape == (DIM,)
            assert deriv['d2'].shape == (DIM,)

    def test_index_item_structure(self, sample_docs):
        rag = FractalLatentRAG()
        rag.add_document("ai", sample_docs["ai_overview"])
        doc_item = rag.index[2][0]
        assert 'id' in doc_item
        assert 'level' in doc_item
        assert 'vec' in doc_item
        assert 'text' in doc_item
        assert doc_item['level'] == 2

    def test_multiple_documents_independent(self, sample_docs):
        rag = FractalLatentRAG()
        for doc_id, text in sample_docs.items():
            rag.add_document(doc_id, text)
        assert len(rag.index[2]) == 3
        assert len(rag.docs) == 3
        assert len(rag.adapters) == 3


class TestScoreWithDerivatives:
    """Test the derivative-enhanced scoring function."""

    def test_base_similarity_component(self, v1_engine_loaded):
        """Score without derivatives should be base cosine similarity."""
        rag = v1_engine_loaded
        qvec = text_to_latent("test query")
        # Doc-level items have no derivatives, so score should be base dot product
        doc_item = rag.index[2][0]
        score = rag._score_with_derivatives(qvec, doc_item, 2)
        expected_base = np.dot(qvec, doc_item['vec'])
        assert abs(score - expected_base) < 1e-6

    def test_derivative_bonus_applied_for_sentences(self, v1_engine_loaded):
        """Sentence-level items should have derivative bonus applied."""
        rag = v1_engine_loaded
        qvec = text_to_latent("machine learning patterns")
        sent_item = rag.index[0][0]
        score_with_deriv = rag._score_with_derivatives(qvec, sent_item, 0)
        base_score = np.dot(qvec, sent_item['vec'])
        # With derivatives, score should differ from base
        # (It could be higher or lower depending on derivative alignment)
        assert score_with_deriv != pytest.approx(base_score, abs=1e-10)


class TestRetrieve:
    """Test the multi-scale retrieval function."""

    def test_returns_all_levels_by_default(self, v1_engine_loaded):
        results = v1_engine_loaded.retrieve("machine learning")
        assert 0 in results
        assert 1 in results
        assert 2 in results

    def test_respects_k_parameter(self, v1_engine_loaded):
        results = v1_engine_loaded.retrieve("machine learning", k=1)
        for lvl in [0, 1, 2]:
            assert len(results[lvl]) <= 1

    def test_respects_levels_parameter(self, v1_engine_loaded):
        results = v1_engine_loaded.retrieve("machine learning", levels=[0])
        assert 0 in results
        assert 1 not in results
        assert 2 not in results

    def test_results_are_sorted_descending(self, v1_engine_loaded):
        results = v1_engine_loaded.retrieve("machine learning", k=5)
        for lvl in results:
            scores = [score for _, score in results[lvl]]
            assert scores == sorted(scores, reverse=True)

    def test_results_contain_item_and_score(self, v1_engine_loaded):
        results = v1_engine_loaded.retrieve("test query")
        for lvl in results:
            for item, score in results[lvl]:
                assert isinstance(item, dict)
                assert isinstance(score, (float, np.floating))

    def test_empty_index_returns_empty(self):
        rag = FractalLatentRAG()
        results = rag.retrieve("anything")
        for lvl in [0, 1, 2]:
            assert len(results[lvl]) == 0

    def test_deterministic_retrieval(self, v1_engine_loaded):
        """Same query should always return same results."""
        r1 = v1_engine_loaded.retrieve("machine learning", k=2)
        r2 = v1_engine_loaded.retrieve("machine learning", k=2)
        for lvl in [0, 1, 2]:
            for (item1, s1), (item2, s2) in zip(r1[lvl], r2[lvl]):
                assert item1['id'] == item2['id']
                assert abs(s1 - s2) < 1e-10


class TestGenerate:
    """Test the formatted generation output."""

    def test_output_contains_query(self, v1_engine_loaded):
        retrieved = v1_engine_loaded.retrieve("test query")
        output = v1_engine_loaded.generate("test query", retrieved)
        assert "test query" in output

    def test_output_contains_all_levels(self, v1_engine_loaded):
        retrieved = v1_engine_loaded.retrieve("test query")
        output = v1_engine_loaded.generate("test query", retrieved)
        assert "DOC-LEVEL" in output
        assert "PARA-LEVEL" in output
        assert "SENT-LEVEL" in output

    def test_output_with_empty_retrieval(self, v1_engine_loaded):
        """Should handle empty retrieval results gracefully."""
        output = v1_engine_loaded.generate("test", {})
        assert "N/A" in output

    def test_output_is_string(self, v1_engine_loaded):
        retrieved = v1_engine_loaded.retrieve("test")
        output = v1_engine_loaded.generate("test", retrieved)
        assert isinstance(output, str)


class TestCompareFlatVsFractal:
    """Test the comparison method outputs correctly."""

    def test_comparison_runs_without_error(self, v1_engine_loaded, capsys):
        v1_engine_loaded.compare_flat_vs_fractal("machine learning")
        captured = capsys.readouterr()
        assert "NAIVE FLAT" in captured.out
        assert "FULL FRACTAL" in captured.out

    def test_comparison_shows_scores(self, v1_engine_loaded, capsys):
        v1_engine_loaded.compare_flat_vs_fractal("machine learning")
        captured = capsys.readouterr()
        # Should contain numeric scores
        assert "Score:" in captured.out or "score" in captured.out.lower()

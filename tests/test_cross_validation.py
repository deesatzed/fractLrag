"""
Tests for fractrag.benchmarks.cross_validation — stratified CV + significance testing.

All tests use real HashEmbedding computations. No mocks.
"""

import numpy as np
import pytest

from fractrag import FractalRAG, HashEmbedding
from fractrag.benchmarks.cross_validation import (
    create_stratified_folds,
    tune_boost_params,
    evaluate_per_query,
    paired_t_test,
    bootstrap_ci,
    run_cross_validation,
    FoldResult,
    PARA_GRID,
    SENT_GRID,
    DERIV_GRID,
)


# ============================================================
# Helpers
# ============================================================

def _make_queries(n_per_type: int = 10) -> list:
    """Build synthetic queries with known relevant PMIDs for HashEmbedding testing."""
    types = ["specification", "summary", "logic", "synthesis"]
    queries = []
    for qtype in types:
        for i in range(n_per_type):
            queries.append({
                "query_text": f"Test {qtype} query number {i} about medical AI",
                "query_type": qtype,
                "relevant_pmids": [f"{1000 + len(types) * i + types.index(qtype)}"],
            })
    return queries


def _make_rag_with_docs(n_docs: int = 20, dim: int = 64) -> FractalRAG:
    """Build a FractalRAG with n_docs real HashEmbedding-indexed documents."""
    rag = FractalRAG(backend=HashEmbedding(dim=dim))
    for i in range(n_docs):
        # Use pmid_ prefix to match the compute_mrr convention
        doc_id = f"pmid_{1000 + i}"
        text = (
            f"This is document {i} about topic {i % 5}. "
            f"It contains multiple sentences for fractal indexing. "
            f"Medical AI applications in domain {i % 3} are discussed. "
            f"Machine learning and deep learning methods are compared."
        )
        rag.add_document(doc_id, text)
    return rag


# ============================================================
# Fold creation tests
# ============================================================

class TestCreateStratifiedFolds:
    def test_correct_fold_count(self):
        queries = _make_queries(10)
        folds = create_stratified_folds(queries, n_folds=5)
        assert len(folds) == 5

    def test_stratification_per_type(self):
        """Each fold's test set has exactly 2 queries per type (10/5 = 2)."""
        queries = _make_queries(10)
        folds = create_stratified_folds(queries, n_folds=5)
        for train, test in folds:
            type_counts = {}
            for q in test:
                t = q["query_type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            for qtype in ["specification", "summary", "logic", "synthesis"]:
                assert type_counts.get(qtype, 0) == 2, f"Expected 2 {qtype} in test, got {type_counts.get(qtype, 0)}"

    def test_all_queries_appear_once_in_test(self):
        """Every query appears in exactly one test fold."""
        queries = _make_queries(10)
        folds = create_stratified_folds(queries, n_folds=5)
        all_test_texts = []
        for _, test in folds:
            all_test_texts.extend(q["query_text"] for q in test)
        assert len(all_test_texts) == len(queries)
        assert len(set(all_test_texts)) == len(queries)

    def test_train_test_disjoint(self):
        queries = _make_queries(10)
        folds = create_stratified_folds(queries, n_folds=5)
        for train, test in folds:
            train_texts = {q["query_text"] for q in train}
            test_texts = {q["query_text"] for q in test}
            assert train_texts.isdisjoint(test_texts)

    def test_fold_sizes(self):
        """With 40 queries and 5 folds: 32 train, 8 test per fold."""
        queries = _make_queries(10)
        folds = create_stratified_folds(queries, n_folds=5)
        for train, test in folds:
            assert len(train) == 32
            assert len(test) == 8

    def test_non_divisible_raises(self):
        """Queries not evenly divisible by folds should raise ValueError."""
        queries = _make_queries(10)
        queries.append({
            "query_text": "Extra query",
            "query_type": "specification",
            "relevant_pmids": ["9999"],
        })
        with pytest.raises(ValueError, match="not divisible"):
            create_stratified_folds(queries, n_folds=5)


# ============================================================
# Tuning tests
# ============================================================

class TestTuneBoostParams:
    def test_returns_valid_grid_params(self):
        rag = _make_rag_with_docs(10)
        queries = _make_queries(4)  # Small for speed
        params, mrr = tune_boost_params(rag, queries, k=5)
        assert params["para_boost"] in PARA_GRID
        assert params["sent_boost"] in SENT_GRID
        assert params["deriv_boost"] in DERIV_GRID

    def test_best_mrr_is_max(self):
        """The returned MRR should be the maximum across the grid."""
        rag = _make_rag_with_docs(10)
        queries = _make_queries(4)
        best_params, best_mrr = tune_boost_params(rag, queries, k=5)
        # Verify by spot-checking a few other combos
        for pb in [0.0, 0.15]:
            for sb in [0.0, 0.30]:
                scores = evaluate_per_query(
                    rag, queries, use_flat=False,
                    para_boost=pb, sent_boost=sb, deriv_boost=0.0, k=5,
                )
                assert float(np.mean(scores)) <= best_mrr + 1e-9

    def test_grid_size(self):
        expected = len(PARA_GRID) * len(SENT_GRID) * len(DERIV_GRID)
        assert expected == 36


# ============================================================
# Per-query eval tests
# ============================================================

class TestEvaluatePerQuery:
    def test_correct_length(self):
        rag = _make_rag_with_docs(10)
        queries = _make_queries(4)
        scores = evaluate_per_query(rag, queries, use_flat=True, k=5)
        assert len(scores) == len(queries)

    def test_scores_in_range(self):
        rag = _make_rag_with_docs(10)
        queries = _make_queries(4)
        for use_flat in [True, False]:
            scores = evaluate_per_query(rag, queries, use_flat=use_flat, k=5)
            for s in scores:
                assert 0.0 <= s <= 1.0, f"Score {s} out of range"

    def test_flat_ignores_boost_params(self):
        """Flat results should be identical regardless of boost params."""
        rag = _make_rag_with_docs(10)
        queries = _make_queries(4)
        scores_a = evaluate_per_query(rag, queries, use_flat=True, para_boost=0.5, sent_boost=0.5, k=5)
        scores_b = evaluate_per_query(rag, queries, use_flat=True, para_boost=0.0, sent_boost=0.0, k=5)
        np.testing.assert_array_almost_equal(scores_a, scores_b)


# ============================================================
# T-test tests
# ============================================================

class TestPairedTTest:
    def test_identical_scores(self):
        scores = [0.5, 0.7, 0.3, 0.9, 0.1]
        t, p, diff = paired_t_test(scores, scores)
        assert abs(t) < 1e-10
        assert abs(p - 1.0) < 1e-10
        assert abs(diff) < 1e-10

    def test_different_scores_significant(self):
        a = [0.9, 0.8, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.86, 0.93]
        b = [0.3, 0.2, 0.25, 0.35, 0.28, 0.32, 0.27, 0.31, 0.26, 0.33]
        t, p, diff = paired_t_test(a, b)
        assert p < 0.001  # Clearly different distributions
        assert diff > 0    # a > b

    def test_correct_direction(self):
        a = [0.6, 0.7, 0.8]
        b = [0.4, 0.5, 0.6]
        _, _, diff = paired_t_test(a, b)
        assert diff > 0  # a > b

        _, _, diff2 = paired_t_test(b, a)
        assert diff2 < 0  # b < a


# ============================================================
# Bootstrap tests
# ============================================================

class TestBootstrapCI:
    def test_ci_contains_mean(self):
        a = [0.8, 0.7, 0.9, 0.85, 0.75]
        b = [0.5, 0.4, 0.6, 0.55, 0.45]
        lower, upper, mean_diff = bootstrap_ci(a, b, seed=42)
        assert lower <= mean_diff <= upper

    def test_zero_diff_ci_contains_zero(self):
        # Same distribution, slight noise
        rng = np.random.default_rng(123)
        scores = rng.uniform(0.3, 0.8, 20).tolist()
        lower, upper, _ = bootstrap_ci(scores, scores, seed=42)
        assert lower <= 0.0 <= upper

    def test_reproducible_with_seed(self):
        a = [0.8, 0.7, 0.9]
        b = [0.5, 0.4, 0.6]
        r1 = bootstrap_ci(a, b, seed=99)
        r2 = bootstrap_ci(a, b, seed=99)
        assert r1 == r2


# ============================================================
# Integration test
# ============================================================

class TestCrossValidationIntegration:
    def test_full_cv_returns_fold_results(self):
        """Full CV with HashEmbedding runs and returns n FoldResults."""
        rag = _make_rag_with_docs(20)
        queries = _make_queries(10)
        fold_results = run_cross_validation(rag, queries, n_folds=5, k=5)

        assert len(fold_results) == 5
        for fr in fold_results:
            assert isinstance(fr, FoldResult)
            assert fr.train_size == 32
            assert fr.test_size == 8
            assert len(fr.flat_scores) == 8
            assert len(fr.fractal_scores) == 8
            assert len(fr.test_query_types) == 8
            assert fr.best_params["para_boost"] in PARA_GRID
            assert all(0.0 <= s <= 1.0 for s in fr.flat_scores)
            assert all(0.0 <= s <= 1.0 for s in fr.fractal_scores)

    def test_all_queries_evaluated_once(self):
        """Every query is evaluated in exactly one test fold."""
        rag = _make_rag_with_docs(20)
        queries = _make_queries(10)
        fold_results = run_cross_validation(rag, queries, n_folds=5, k=5)

        total_test = sum(fr.test_size for fr in fold_results)
        assert total_test == 40  # All 40 queries evaluated once

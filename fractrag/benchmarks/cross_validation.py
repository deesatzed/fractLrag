#!/usr/bin/env python3
"""
Cross-validation + statistical significance testing for Fractal vs Flat retrieval.

Addresses the key validation gap: boost parameters were tuned AND evaluated on the
same 40 queries. This script uses stratified k-fold CV so that tuning and evaluation
happen on disjoint query sets, giving an honest improvement estimate.

Statistical tests:
  - Paired t-test (per-query MRR as pairing variable)
  - Bootstrap 95% CI for mean MRR improvement
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fractrag import FractalRAG, SentenceTransformerEmbedding, HashEmbedding
from fractrag.core import EmbeddingBackend
from fractrag.benchmarks.flat_vs_fractal import (
    load_corpus,
    build_index,
    compute_mrr,
)


RESULTS_PATH = Path(__file__).resolve().parents[2] / "corpus" / "cv_results.json"

# Expanded grid for CV tuning (5x6x5 = 150 combos, covers post-3A landscape)
PARA_GRID = [0.0, 0.02, 0.05, 0.10, 0.15]
SENT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
DERIV_GRID = [0.0, 0.03, 0.05, 0.10, 0.15]


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold_idx: int
    train_size: int
    test_size: int
    best_params: Dict[str, float]
    best_train_mrr: float
    flat_scores: List[float]       # per-query MRR on test set (flat)
    fractal_scores: List[float]    # per-query MRR on test set (fractal with tuned params)
    test_query_types: List[str]    # query type for each test query
    per_type_params: Optional[Dict[str, Dict[str, float]]] = None  # per-type tuned params
    fractal_per_type_scores: Optional[List[float]] = None  # per-query MRR with per-type tuning


def create_stratified_folds(queries: List[Dict], n_folds: int = 5) -> List[Tuple[List[Dict], List[Dict]]]:
    """Split queries into n stratified folds.

    Each fold's test set has exactly (count_per_type / n_folds) queries of each type.
    With 40 queries (10 per type) and 5 folds, that's 2 per type per fold = 8 test queries.

    Returns:
        List of (train_queries, test_queries) tuples, one per fold.
    """
    # Group queries by type
    by_type: Dict[str, List[Dict]] = {}
    for q in queries:
        qtype = q["query_type"]
        by_type.setdefault(qtype, []).append(q)

    # Validate: each type must have a count divisible by n_folds
    for qtype, qs in by_type.items():
        if len(qs) % n_folds != 0:
            raise ValueError(
                f"Type '{qtype}' has {len(qs)} queries, not divisible by {n_folds} folds"
            )

    # Build fold assignments
    folds = []
    for fold_idx in range(n_folds):
        test_queries = []
        train_queries = []
        for qtype, qs in by_type.items():
            per_fold = len(qs) // n_folds
            start = fold_idx * per_fold
            end = start + per_fold
            test_queries.extend(qs[start:end])
            train_queries.extend(qs[:start] + qs[end:])
        folds.append((train_queries, test_queries))

    return folds


def _evaluate_query_mrr(
    rag: FractalRAG,
    query: Dict,
    use_flat: bool,
    para_boost: float = 0.0,
    sent_boost: float = 0.0,
    deriv_boost: float = 0.0,
    k: int = 10,
) -> float:
    """Evaluate a single query's MRR using either flat or reranked retrieval."""
    qtext = query["query_text"]
    relevant = query["relevant_pmids"]

    if use_flat:
        results, _ = rag.retrieve_flat(qtext, k=k)
    else:
        results, _ = rag.retrieve_reranked(
            qtext, k=k,
            use_derivatives=(deriv_boost > 0),
            para_boost=para_boost,
            sent_boost=sent_boost,
            deriv_boost=deriv_boost,
        )

    return compute_mrr(results, relevant, k=k)


def evaluate_per_query(
    rag: FractalRAG,
    queries: List[Dict],
    use_flat: bool,
    para_boost: float = 0.0,
    sent_boost: float = 0.0,
    deriv_boost: float = 0.0,
    k: int = 10,
) -> List[float]:
    """Returns list of per-query MRR floats."""
    return [
        _evaluate_query_mrr(rag, q, use_flat, para_boost, sent_boost, deriv_boost, k)
        for q in queries
    ]


def tune_boost_params(
    rag: FractalRAG,
    train_queries: List[Dict],
    k: int = 10,
) -> Tuple[Dict[str, float], float]:
    """Reduced grid sweep on training queries only.

    Returns:
        (best_params_dict, best_mean_mrr)
    """
    best_mrr = -1.0
    best_params = {"para_boost": 0.0, "sent_boost": 0.0, "deriv_boost": 0.0}

    for pb in PARA_GRID:
        for sb in SENT_GRID:
            for db in DERIV_GRID:
                scores = evaluate_per_query(
                    rag, train_queries, use_flat=False,
                    para_boost=pb, sent_boost=sb, deriv_boost=db, k=k,
                )
                mean_mrr = float(np.mean(scores))
                if mean_mrr > best_mrr:
                    best_mrr = mean_mrr
                    best_params = {"para_boost": pb, "sent_boost": sb, "deriv_boost": db}

    return best_params, best_mrr


def tune_boost_params_per_type(
    rag: FractalRAG,
    train_queries: List[Dict],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Tune boost params separately per query type on training queries."""
    type_queries: Dict[str, List[Dict]] = {}
    for q in train_queries:
        type_queries.setdefault(q["query_type"], []).append(q)

    type_params: Dict[str, Dict[str, float]] = {}
    for qtype, qs in type_queries.items():
        best_mrr = -1.0
        best: Dict[str, float] = {"para_boost": 0.0, "sent_boost": 0.0, "deriv_boost": 0.0}
        for pb in PARA_GRID:
            for sb in SENT_GRID:
                for db in DERIV_GRID:
                    scores = evaluate_per_query(
                        rag, qs, use_flat=False,
                        para_boost=pb, sent_boost=sb, deriv_boost=db, k=k,
                    )
                    mean = float(np.mean(scores))
                    if mean > best_mrr:
                        best_mrr = mean
                        best = {"para_boost": pb, "sent_boost": sb, "deriv_boost": db}
        type_params[qtype] = best
    return type_params


def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float, float]:
    """Paired t-test on per-query scores.

    Args:
        scores_a: Per-query MRR for method A (e.g., fractal).
        scores_b: Per-query MRR for method B (e.g., flat baseline).

    Returns:
        (t_statistic, p_value, mean_difference) where mean_diff = mean(a) - mean(b).
    """
    from scipy.stats import ttest_rel

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Score arrays must be same length: {len(a)} vs {len(b)}")

    # Handle degenerate case: identical scores
    if np.allclose(a, b):
        return 0.0, 1.0, 0.0

    t_stat, p_val = ttest_rel(a, b)
    mean_diff = float(np.mean(a - b))
    return float(t_stat), float(p_val), mean_diff


def bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mean MRR difference (a - b).

    Returns:
        (lower_bound, upper_bound, mean_diff)
    """
    rng = np.random.default_rng(seed)
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diffs = a - b
    n = len(diffs)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[idx])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    return lower, upper, mean_diff


def run_cross_validation(
    rag: FractalRAG,
    queries: List[Dict],
    n_folds: int = 5,
    k: int = 10,
) -> List[FoldResult]:
    """Full cross-validation pipeline.

    For each fold:
      1. Split queries into train/test (stratified)
      2. Tune boost params on train queries
      3. Evaluate flat and fractal on test queries
      4. Collect per-query MRR for paired tests
    """
    folds = create_stratified_folds(queries, n_folds)
    results = []

    for fold_idx, (train_qs, test_qs) in enumerate(folds):
        # Tune on training set (uniform params)
        best_params, best_train_mrr = tune_boost_params(rag, train_qs, k=k)

        # Tune on training set (per-type params)
        per_type_params = tune_boost_params_per_type(rag, train_qs, k=k)

        # Evaluate on test set
        flat_scores = evaluate_per_query(rag, test_qs, use_flat=True, k=k)
        fractal_scores = evaluate_per_query(
            rag, test_qs, use_flat=False,
            para_boost=best_params["para_boost"],
            sent_boost=best_params["sent_boost"],
            deriv_boost=best_params["deriv_boost"],
            k=k,
        )

        # Evaluate per-type tuned: each test query uses its type's optimal params
        per_type_scores = []
        for q in test_qs:
            qtype = q["query_type"]
            tp = per_type_params.get(qtype, best_params)
            score = _evaluate_query_mrr(
                rag, q, use_flat=False,
                para_boost=tp["para_boost"],
                sent_boost=tp["sent_boost"],
                deriv_boost=tp["deriv_boost"],
                k=k,
            )
            per_type_scores.append(score)

        test_types = [q["query_type"] for q in test_qs]

        results.append(FoldResult(
            fold_idx=fold_idx,
            train_size=len(train_qs),
            test_size=len(test_qs),
            best_params=best_params,
            best_train_mrr=best_train_mrr,
            flat_scores=flat_scores,
            fractal_scores=fractal_scores,
            test_query_types=test_types,
            per_type_params=per_type_params,
            fractal_per_type_scores=per_type_scores,
        ))

    return results


def print_cv_report(fold_results: List[FoldResult]) -> Dict:
    """Print a formatted cross-validation report and return summary dict."""
    # Collect all per-query scores across folds
    all_flat = []
    all_fractal = []
    all_types = []
    for fr in fold_results:
        all_flat.extend(fr.flat_scores)
        all_fractal.extend(fr.fractal_scores)
        all_types.extend(fr.test_query_types)

    # Overall metrics
    mean_flat = float(np.mean(all_flat))
    mean_fractal = float(np.mean(all_fractal))
    improvement_pct = ((mean_fractal - mean_flat) / mean_flat * 100) if mean_flat > 0 else 0.0

    # Overall significance
    t_stat, p_val, mean_diff = paired_t_test(all_fractal, all_flat)
    ci_lower, ci_upper, _ = bootstrap_ci(all_fractal, all_flat, seed=42)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION REPORT: FRACTAL vs FLAT")
    print("=" * 80)

    # Per-fold summary
    print(f"\n{'FOLD':<6} {'TRAIN':>6} {'TEST':>6} {'FLAT MRR':>10} {'FRACT MRR':>11} {'DELTA':>8} {'PARAMS'}")
    print("-" * 80)
    for fr in fold_results:
        flat_mrr = float(np.mean(fr.flat_scores))
        frac_mrr = float(np.mean(fr.fractal_scores))
        delta = frac_mrr - flat_mrr
        params_str = f"p={fr.best_params['para_boost']:.2f} s={fr.best_params['sent_boost']:.2f} d={fr.best_params['deriv_boost']:.2f}"
        print(f"  {fr.fold_idx:<4} {fr.train_size:>6} {fr.test_size:>6} {flat_mrr:>10.4f} {frac_mrr:>11.4f} {delta:>+8.4f} {params_str}")

    # Overall
    print(f"\n{'OVERALL':}")
    print(f"  Flat MRR (CV):     {mean_flat:.4f}")
    print(f"  Fractal MRR (CV):  {mean_fractal:.4f}")
    print(f"  Improvement:       {improvement_pct:+.1f}%")
    print(f"  Mean diff:         {mean_diff:+.4f}")

    # Significance
    print(f"\nSTATISTICAL SIGNIFICANCE (paired t-test, n={len(all_flat)} queries):")
    print(f"  t-statistic:       {t_stat:.3f}")
    print(f"  p-value:           {p_val:.4f}")
    sig = "YES" if p_val < 0.05 else "NO"
    print(f"  Significant (p<0.05): {sig}")

    print(f"\nBOOTSTRAP 95% CI for mean MRR improvement:")
    print(f"  [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    contains_zero = ci_lower <= 0 <= ci_upper
    print(f"  Contains zero: {'YES (inconclusive)' if contains_zero else 'NO (significant)'}")

    # Per-type significance
    print(f"\nPER-TYPE ANALYSIS:")
    print(f"{'TYPE':<18} {'N':>4} {'FLAT':>8} {'FRACT':>8} {'DELTA':>8} {'p-val':>8} {'SIG':>5}")
    print("-" * 60)

    type_results = {}
    for qtype in ["specification", "summary", "logic", "synthesis"]:
        type_flat = [all_flat[i] for i in range(len(all_types)) if all_types[i] == qtype]
        type_frac = [all_fractal[i] for i in range(len(all_types)) if all_types[i] == qtype]
        n = len(type_flat)
        if n > 0:
            tf = float(np.mean(type_flat))
            ff = float(np.mean(type_frac))
            delta = ff - tf
            try:
                _, tp, _ = paired_t_test(type_frac, type_flat)
            except Exception:
                tp = 1.0
            sig = "*" if tp < 0.05 else ""
            print(f"{qtype:<18} {n:>4} {tf:>8.4f} {ff:>8.4f} {delta:>+8.4f} {tp:>8.4f} {sig:>5}")
            type_results[qtype] = {
                "n": n, "flat_mrr": round(tf, 4), "fractal_mrr": round(ff, 4),
                "delta": round(delta, 4), "p_value": round(tp, 4),
            }

    # Per-type tuned results (if available)
    all_per_type_fractal = []
    has_per_type = all(fr.fractal_per_type_scores is not None for fr in fold_results)
    per_type_tuned_summary = {}
    if has_per_type:
        for fr in fold_results:
            all_per_type_fractal.extend(fr.fractal_per_type_scores)
        mean_per_type = float(np.mean(all_per_type_fractal))
        pt_improvement = ((mean_per_type - mean_flat) / mean_flat * 100) if mean_flat > 0 else 0.0
        pt_t, pt_p, pt_diff = paired_t_test(all_per_type_fractal, all_flat)
        pt_ci_lower, pt_ci_upper, _ = bootstrap_ci(all_per_type_fractal, all_flat, seed=42)

        print(f"\nPER-TYPE TUNED FRACTAL (each type uses its own optimal params):")
        print(f"  MRR:          {mean_per_type:.4f} ({pt_improvement:+.1f}% vs flat)")
        print(f"  p-value:      {pt_p:.4f}")
        print(f"  95% CI:       [{pt_ci_lower:+.4f}, {pt_ci_upper:+.4f}]")

        # Show per-type params from each fold
        print(f"\n  Per-type params (per fold):")
        for fr in fold_results:
            if fr.per_type_params:
                params_str = ", ".join(
                    f"{t}: p={p['para_boost']:.2f} s={p['sent_boost']:.2f} d={p['deriv_boost']:.2f}"
                    for t, p in sorted(fr.per_type_params.items())
                )
                print(f"    Fold {fr.fold_idx}: {params_str}")

        per_type_tuned_summary = {
            "mrr": round(mean_per_type, 4),
            "improvement_pct": round(pt_improvement, 2),
            "p_value": round(pt_p, 4),
            "ci_lower": round(pt_ci_lower, 4),
            "ci_upper": round(pt_ci_upper, 4),
        }

    # Verdict
    print(f"\n{'='*80}")
    if p_val < 0.05 and improvement_pct >= 5:
        verdict = "VALIDATED"
        print(f"VERDICT: {verdict} — Statistically significant improvement of {improvement_pct:+.1f}%")
    elif p_val < 0.05:
        verdict = "SIGNIFICANT_BUT_SMALL"
        print(f"VERDICT: {verdict} — Significant (p={p_val:.4f}) but small improvement ({improvement_pct:+.1f}%)")
    elif improvement_pct >= 5:
        verdict = "PROMISING_NOT_SIGNIFICANT"
        print(f"VERDICT: {verdict} — {improvement_pct:+.1f}% improvement but p={p_val:.4f} (not significant)")
    else:
        verdict = "NOT_VALIDATED"
        print(f"VERDICT: {verdict} — No significant improvement ({improvement_pct:+.1f}%, p={p_val:.4f})")
    print(f"{'='*80}")

    return {
        "overall": {
            "flat_mrr": round(mean_flat, 4),
            "fractal_mrr": round(mean_fractal, 4),
            "improvement_pct": round(improvement_pct, 2),
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "significant": p_val < 0.05,
            "verdict": verdict,
        },
        "per_type": type_results,
        "per_type_tuned": per_type_tuned_summary,
        "folds": [
            {
                "fold_idx": fr.fold_idx,
                "best_params": fr.best_params,
                "per_type_params": fr.per_type_params,
                "best_train_mrr": round(fr.best_train_mrr, 4),
                "test_flat_mrr": round(float(np.mean(fr.flat_scores)), 4),
                "test_fractal_mrr": round(float(np.mean(fr.fractal_scores)), 4),
                "test_per_type_mrr": round(float(np.mean(fr.fractal_per_type_scores)), 4) if fr.fractal_per_type_scores else None,
            }
            for fr in fold_results
        ],
    }


def main():
    print("Loading corpus...")
    documents, queries = load_corpus()
    print(f"  {len(documents)} documents, {len(queries)} queries")

    print("\nLoading BGE-M3 embedding model...")
    t0 = time.time()
    backend = SentenceTransformerEmbedding("BAAI/bge-m3")
    print(f"  Model loaded in {time.time()-t0:.1f}s (dim={backend.dim})")

    print(f"\nBuilding fractal index ({len(documents)} documents)...")
    t0 = time.time()
    rag = build_index(documents, backend)
    stats = rag.stats()
    print(f"  Indexed in {time.time()-t0:.1f}s")
    print(f"  Sentences: {stats['entries_level_0']}, Paragraphs: {stats['entries_level_1']}, Documents: {stats['entries_level_2']}")

    grid_size = len(PARA_GRID) * len(SENT_GRID) * len(DERIV_GRID)
    print(f"\nRunning 5-fold cross-validation ({grid_size} param combos per fold)...")
    t0 = time.time()
    fold_results = run_cross_validation(rag, queries, n_folds=5, k=10)
    cv_time = time.time() - t0
    print(f"  CV completed in {cv_time:.1f}s")

    summary = print_cv_report(fold_results)

    # Save results
    output = {
        "metadata": {
            "corpus_size": len(documents),
            "query_count": len(queries),
            "n_folds": 5,
            "grid_size": len(PARA_GRID) * len(SENT_GRID) * len(DERIV_GRID),
            "embedding_model": "BAAI/bge-m3",
            "cv_time_seconds": round(cv_time, 1),
        },
        "summary": summary,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

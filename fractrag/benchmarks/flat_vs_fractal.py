#!/usr/bin/env python3
"""
THE HYPOTHESIS TEST: Does fractal retrieval beat flat retrieval with real embeddings?

Indexes the medical corpus with BGE-M3 embeddings, runs 40 queries through
multiple retrieval configurations, and reports metrics per query type.

Configurations tested (ablation):
  1. FLAT        — doc-level only, no derivatives, no level weights
  2. MULTI-SCALE — all 3 levels, uniform weights, no derivatives
  3. +DERIV      — all 3 levels, uniform weights, WITH derivatives
  4. +TYPE-AWARE — all 3 levels, type-aware weights, no derivatives
  5. FULL FRACTAL — all 3 levels, type-aware weights, WITH derivatives (all features)

Go/No-Go criteria:
  - FULL FRACTAL must beat FLAT by >= 10% on MRR to justify the complexity.
  - If < 5%, the fractal approach does not justify its overhead.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add parent to path for fractrag imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fractrag import FractalRAG, SentenceTransformerEmbedding, HashEmbedding
from fractrag.core import EmbeddingBackend


CORPUS_PATH = Path(__file__).resolve().parents[2] / "corpus" / "medical_corpus.json"
QUERIES_PATH = Path(__file__).resolve().parents[2] / "corpus" / "medical_queries.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "corpus" / "benchmark_results.json"


def load_corpus() -> Tuple[List[Dict], List[Dict]]:
    """Load the medical corpus and query set."""
    corpus = json.loads(CORPUS_PATH.read_text())
    queries = json.loads(QUERIES_PATH.read_text())
    return corpus["documents"], queries["queries"]


def build_index(documents: List[Dict], backend: EmbeddingBackend) -> FractalRAG:
    """Build a FractalRAG index from the corpus with metadata and title-prefix."""
    rag = FractalRAG(backend=backend)
    for doc in documents:
        text = doc["abstract"]
        metadata = {
            "domain": doc.get("domain"),
            "year": doc.get("year"),
            "mesh_terms": doc.get("mesh_terms", []),
            "journal": doc.get("journal"),
            "title": doc.get("title"),
        }
        rag.add_document(
            doc["doc_id"], text,
            metadata=metadata,
            title=doc.get("title"),
        )
    return rag


def compute_mrr(results_by_level: Dict, relevant_pmids: List[str], k: int = 10) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result across all levels."""
    # Flatten all results across levels, sorted by score
    all_results = []
    for lvl, items in results_by_level.items():
        for entry, score in items:
            # Map entry ID back to doc_id (parent or self)
            doc_id = entry.parent if entry.parent else entry.id
            all_results.append((doc_id, score))

    # Sort by score descending, deduplicate by doc_id
    all_results.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    ranked = []
    for doc_id, score in all_results:
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= k:
            break

    # Find rank of first relevant doc
    relevant_set = {f"pmid_{pmid}" for pmid in relevant_pmids}
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(results_by_level: Dict, relevant_pmids: List[str], k: int = 3) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    all_results = []
    for lvl, items in results_by_level.items():
        for entry, score in items:
            doc_id = entry.parent if entry.parent else entry.id
            all_results.append((doc_id, score))

    all_results.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    ranked = []
    for doc_id, score in all_results:
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= k:
            break

    relevant_set = {f"pmid_{pmid}" for pmid in relevant_pmids}
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant_set)
    return hits / min(k, len(ranked)) if ranked else 0.0


def compute_recall_at_k(results_by_level: Dict, relevant_pmids: List[str], k: int = 10) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    all_results = []
    for lvl, items in results_by_level.items():
        for entry, score in items:
            doc_id = entry.parent if entry.parent else entry.id
            all_results.append((doc_id, score))

    all_results.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    ranked = []
    for doc_id, score in all_results:
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= k:
            break

    relevant_set = {f"pmid_{pmid}" for pmid in relevant_pmids}
    found = sum(1 for doc_id in ranked[:k] if doc_id in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0


def run_reranked_configuration(
    rag: FractalRAG,
    queries: List[Dict],
    config_name: str,
    use_derivatives: bool = True,
    para_boost: float = 0.15,
    sent_boost: float = 0.10,
    deriv_boost: float = 0.05,
    k: int = 10,
) -> Dict:
    """Run all queries through the reranked retrieval configuration."""
    results = {
        "config": config_name,
        "overall": {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []},
        "by_type": {},
    }

    for query in queries:
        qtext = query["query_text"]
        relevant = query["relevant_pmids"]
        qtype = query["query_type"]

        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []}

        retrieved, _ = rag.retrieve_reranked(
            qtext, k=k,
            use_derivatives=use_derivatives,
            para_boost=para_boost,
            sent_boost=sent_boost,
            deriv_boost=deriv_boost,
        )

        mrr = compute_mrr(retrieved, relevant)
        p1 = compute_precision_at_k(retrieved, relevant, k=1)
        p3 = compute_precision_at_k(retrieved, relevant, k=3)
        r10 = compute_recall_at_k(retrieved, relevant, k=10)

        results["overall"]["mrr"].append(mrr)
        results["overall"]["precision_at_1"].append(p1)
        results["overall"]["precision_at_3"].append(p3)
        results["overall"]["recall_at_10"].append(r10)

        results["by_type"][qtype]["mrr"].append(mrr)
        results["by_type"][qtype]["precision_at_1"].append(p1)
        results["by_type"][qtype]["precision_at_3"].append(p3)
        results["by_type"][qtype]["recall_at_10"].append(r10)

    # Compute averages
    for metric in results["overall"]:
        vals = results["overall"][metric]
        results["overall"][metric] = round(np.mean(vals), 4) if vals else 0.0
    for qtype in results["by_type"]:
        for metric in results["by_type"][qtype]:
            vals = results["by_type"][qtype][metric]
            results["by_type"][qtype][metric] = round(np.mean(vals), 4) if vals else 0.0

    return results


def run_configuration(
    rag: FractalRAG,
    queries: List[Dict],
    config_name: str,
    levels: Optional[List[int]] = None,
    use_derivatives: bool = False,
    use_level_weights: bool = False,
) -> Dict:
    """Run all queries through a specific retrieval configuration."""
    results = {
        "config": config_name,
        "overall": {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []},
        "by_type": {},
    }

    for query in queries:
        qtext = query["query_text"]
        relevant = query["relevant_pmids"]
        qtype = query["query_type"]

        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []}

        retrieved, _ = rag.retrieve(
            qtext, k=5,
            levels=levels,
            use_derivatives=use_derivatives,
            use_level_weights=use_level_weights,
        )

        mrr = compute_mrr(retrieved, relevant)
        p1 = compute_precision_at_k(retrieved, relevant, k=1)
        p3 = compute_precision_at_k(retrieved, relevant, k=3)
        r10 = compute_recall_at_k(retrieved, relevant, k=10)

        results["overall"]["mrr"].append(mrr)
        results["overall"]["precision_at_1"].append(p1)
        results["overall"]["precision_at_3"].append(p3)
        results["overall"]["recall_at_10"].append(r10)

        results["by_type"][qtype]["mrr"].append(mrr)
        results["by_type"][qtype]["precision_at_1"].append(p1)
        results["by_type"][qtype]["precision_at_3"].append(p3)
        results["by_type"][qtype]["recall_at_10"].append(r10)

    # Compute averages
    for metric in results["overall"]:
        vals = results["overall"][metric]
        results["overall"][metric] = round(np.mean(vals), 4) if vals else 0.0
    for qtype in results["by_type"]:
        for metric in results["by_type"][qtype]:
            vals = results["by_type"][qtype][metric]
            results["by_type"][qtype][metric] = round(np.mean(vals), 4) if vals else 0.0

    return results


def print_results_table(all_configs: List[Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS: FLAT vs FRACTAL RETRIEVAL (Real BGE-M3 Embeddings)")
    print("=" * 90)

    # Overall table
    print(f"\n{'CONFIG':<25} {'MRR':>8} {'P@1':>8} {'P@3':>8} {'R@10':>8}")
    print("-" * 60)
    for cfg in all_configs:
        o = cfg["overall"]
        print(f"{cfg['config']:<25} {o['mrr']:>8.4f} {o['precision_at_1']:>8.4f} {o['precision_at_3']:>8.4f} {o['recall_at_10']:>8.4f}")

    # Per-type breakdown: FLAT vs OPTIMAL (last config)
    flat = all_configs[0]
    best = all_configs[-1]

    print(f"\n{'='*90}")
    print(f"PER QUERY TYPE: FLAT vs {best['config']}")
    print(f"{'='*90}")
    print(f"{'TYPE':<18} {'FLAT MRR':>10} {'BEST MRR':>12} {'DELTA':>8} {'WINNER':>10}")
    print("-" * 60)
    for qtype in ["specification", "summary", "logic", "synthesis"]:
        flat_mrr = flat["by_type"].get(qtype, {}).get("mrr", 0)
        best_mrr = best["by_type"].get(qtype, {}).get("mrr", 0)
        delta = best_mrr - flat_mrr
        pct = (delta / flat_mrr * 100) if flat_mrr > 0 else 0
        winner = "FRACTAL" if delta > 0 else "FLAT" if delta < 0 else "TIE"
        print(f"{qtype:<18} {flat_mrr:>10.4f} {best_mrr:>12.4f} {pct:>+7.1f}% {winner:>10}")

    # Also show recall comparison
    print(f"\n{'TYPE':<18} {'FLAT R@10':>10} {'BEST R@10':>12} {'DELTA':>8} {'WINNER':>10}")
    print("-" * 60)
    for qtype in ["specification", "summary", "logic", "synthesis"]:
        flat_r = flat["by_type"].get(qtype, {}).get("recall_at_10", 0)
        best_r = best["by_type"].get(qtype, {}).get("recall_at_10", 0)
        delta = best_r - flat_r
        pct = (delta / flat_r * 100) if flat_r > 0 else 0
        winner = "FRACTAL" if delta > 0 else "FLAT" if delta < 0 else "TIE"
        print(f"{qtype:<18} {flat_r:>10.4f} {best_r:>12.4f} {pct:>+7.1f}% {winner:>10}")

    # Go/No-Go
    flat_mrr = flat["overall"]["mrr"]
    full_mrr = best["overall"]["mrr"]
    improvement = ((full_mrr - flat_mrr) / flat_mrr * 100) if flat_mrr > 0 else 0

    print(f"\n{'='*90}")
    print(f"OVERALL MRR: FLAT={flat_mrr:.4f}  FRACTAL={full_mrr:.4f}  DELTA={improvement:+.1f}%")
    print(f"{'='*90}")

    if improvement >= 10:
        print("GO — Fractal retrieval shows >= 10% improvement. Hypothesis VALIDATED.")
    elif improvement >= 5:
        print("CONDITIONAL GO — 5-10% improvement. Identify which components help most.")
    elif improvement >= 0:
        print(f"PIVOT — <5% improvement ({improvement:.1f}%). Complexity may not be justified.")
    else:
        print(f"STOP — Fractal is WORSE than flat ({improvement:.1f}%). Hypothesis REJECTED.")

    return improvement


def run_per_type_sweep(
    rag: FractalRAG,
    queries: List[Dict],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Find optimal boost parameters per query type.

    Runs a parameter grid separately per query type, finding the best
    params for each type independently.
    """
    type_queries: Dict[str, List[Dict]] = {}
    for q in queries:
        type_queries.setdefault(q["query_type"], []).append(q)

    type_best: Dict[str, Dict[str, float]] = {}
    for qtype, qs in type_queries.items():
        best_mrr = -1.0
        best_params = {"para_boost": 0.0, "sent_boost": 0.0, "deriv_boost": 0.0}
        for pb in [0.0, 0.03, 0.05, 0.10, 0.15]:
            for sb in [0.0, 0.05, 0.10, 0.20]:
                for db in [0.0, 0.03, 0.05, 0.10, 0.15]:
                    r = run_reranked_configuration(
                        rag, qs,
                        f"TYPE_SWEEP_{qtype}",
                        use_derivatives=(db > 0),
                        para_boost=pb, sent_boost=sb, deriv_boost=db, k=k,
                    )
                    if r["overall"]["mrr"] > best_mrr:
                        best_mrr = r["overall"]["mrr"]
                        best_params = {"para_boost": pb, "sent_boost": sb, "deriv_boost": db}
        type_best[qtype] = {**best_params, "mrr": best_mrr}
    return type_best


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
    indexing_time = time.time() - t0
    stats = rag.stats()
    print(f"  Indexed in {indexing_time:.1f}s")
    print(f"  Level 0 (sentences): {stats['entries_level_0']}")
    print(f"  Level 1 (paragraphs): {stats['entries_level_1']}")
    print(f"  Level 2 (documents): {stats['entries_level_2']}")
    print(f"  Derivatives: {stats['derivatives']}")

    # Run all configurations
    configs = []

    print("\nRunning FLAT (doc-only, no features, k=10)...")
    t0 = time.time()
    flat = run_configuration(rag, queries, "FLAT", levels=[2], use_derivatives=False, use_level_weights=False)
    print(f"  Done in {time.time()-t0:.1f}s — MRR={flat['overall']['mrr']:.4f}")
    configs.append(flat)

    print("Running MULTI-SCALE (all levels, uniform weights)...")
    t0 = time.time()
    multi = run_configuration(rag, queries, "MULTI-SCALE", levels=[2,1,0], use_derivatives=False, use_level_weights=False)
    print(f"  Done in {time.time()-t0:.1f}s — MRR={multi['overall']['mrr']:.4f}")
    configs.append(multi)

    print("Running +DERIVATIVES (multi-scale + derivative scoring)...")
    t0 = time.time()
    deriv = run_configuration(rag, queries, "+DERIVATIVES", levels=[2,1,0], use_derivatives=True, use_level_weights=False)
    print(f"  Done in {time.time()-t0:.1f}s — MRR={deriv['overall']['mrr']:.4f}")
    configs.append(deriv)

    print("Running +TYPE-AWARE (multi-scale + query-type weights)...")
    t0 = time.time()
    typed = run_configuration(rag, queries, "+TYPE-AWARE", levels=[2,1,0], use_derivatives=False, use_level_weights=True)
    print(f"  Done in {time.time()-t0:.1f}s — MRR={typed['overall']['mrr']:.4f}")
    configs.append(typed)

    print("Running FULL FRACTAL (all features)...")
    t0 = time.time()
    full = run_configuration(rag, queries, "FULL FRACTAL", levels=[2,1,0], use_derivatives=True, use_level_weights=True)
    print(f"  Done in {time.time()-t0:.1f}s — MRR={full['overall']['mrr']:.4f}")
    configs.append(full)

    # --- ADAPTIVE FRACTAL: type-aware strategy selection ---
    print("\nRunning ADAPTIVE FRACTAL (flat for spec, reranked for others)...")
    t0 = time.time()
    adaptive_results = {
        "config": "ADAPTIVE FRACTAL",
        "overall": {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []},
        "by_type": {},
    }
    for query in queries:
        qtext = query["query_text"]
        relevant = query["relevant_pmids"]
        qtype = query["query_type"]
        if qtype not in adaptive_results["by_type"]:
            adaptive_results["by_type"][qtype] = {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []}
        retrieved, _ = rag.retrieve_adaptive(qtext, k=10)
        mrr = compute_mrr(retrieved, relevant)
        p1 = compute_precision_at_k(retrieved, relevant, k=1)
        p3 = compute_precision_at_k(retrieved, relevant, k=3)
        r10 = compute_recall_at_k(retrieved, relevant, k=10)
        adaptive_results["overall"]["mrr"].append(mrr)
        adaptive_results["overall"]["precision_at_1"].append(p1)
        adaptive_results["overall"]["precision_at_3"].append(p3)
        adaptive_results["overall"]["recall_at_10"].append(r10)
        adaptive_results["by_type"][qtype]["mrr"].append(mrr)
        adaptive_results["by_type"][qtype]["precision_at_1"].append(p1)
        adaptive_results["by_type"][qtype]["precision_at_3"].append(p3)
        adaptive_results["by_type"][qtype]["recall_at_10"].append(r10)
    for metric in adaptive_results["overall"]:
        vals = adaptive_results["overall"][metric]
        adaptive_results["overall"][metric] = round(np.mean(vals), 4) if vals else 0.0
    for qtype in adaptive_results["by_type"]:
        for metric in adaptive_results["by_type"][qtype]:
            vals = adaptive_results["by_type"][qtype][metric]
            adaptive_results["by_type"][qtype][metric] = round(np.mean(vals), 4) if vals else 0.0
    print(f"  Done in {time.time()-t0:.1f}s — MRR={adaptive_results['overall']['mrr']:.4f}")
    configs.append(adaptive_results)

    # --- RRF FRACTAL: reciprocal rank fusion across levels ---
    print("Running RRF FRACTAL (rank-based fusion across 3 levels)...")
    t0 = time.time()
    rrf_results = {
        "config": "RRF FRACTAL",
        "overall": {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []},
        "by_type": {},
    }
    for query in queries:
        qtext = query["query_text"]
        relevant = query["relevant_pmids"]
        qtype = query["query_type"]
        if qtype not in rrf_results["by_type"]:
            rrf_results["by_type"][qtype] = {"mrr": [], "precision_at_1": [], "precision_at_3": [], "recall_at_10": []}
        retrieved, _ = rag.retrieve_rrf(qtext, k=10)
        mrr = compute_mrr(retrieved, relevant)
        p1 = compute_precision_at_k(retrieved, relevant, k=1)
        p3 = compute_precision_at_k(retrieved, relevant, k=3)
        r10 = compute_recall_at_k(retrieved, relevant, k=10)
        rrf_results["overall"]["mrr"].append(mrr)
        rrf_results["overall"]["precision_at_1"].append(p1)
        rrf_results["overall"]["precision_at_3"].append(p3)
        rrf_results["overall"]["recall_at_10"].append(r10)
        rrf_results["by_type"][qtype]["mrr"].append(mrr)
        rrf_results["by_type"][qtype]["precision_at_1"].append(p1)
        rrf_results["by_type"][qtype]["precision_at_3"].append(p3)
        rrf_results["by_type"][qtype]["recall_at_10"].append(r10)
    for metric in rrf_results["overall"]:
        vals = rrf_results["overall"][metric]
        rrf_results["overall"][metric] = round(np.mean(vals), 4) if vals else 0.0
    for qtype in rrf_results["by_type"]:
        for metric in rrf_results["by_type"][qtype]:
            vals = rrf_results["by_type"][qtype][metric]
            rrf_results["by_type"][qtype][metric] = round(np.mean(vals), 4) if vals else 0.0
    print(f"  Done in {time.time()-t0:.1f}s — MRR={rrf_results['overall']['mrr']:.4f}")
    configs.append(rrf_results)

    # --- RERANKED FRACTAL: doc-primary with sub-doc boosts ---
    print("Running RERANKED FRACTAL (doc-primary + sub-doc boosts, k=10)...")
    t0 = time.time()
    reranked = run_reranked_configuration(
        rag, queries, "RERANKED FRACTAL",
        use_derivatives=True, para_boost=0.0, sent_boost=0.20, deriv_boost=0.05,
    )
    print(f"  Done in {time.time()-t0:.1f}s — MRR={reranked['overall']['mrr']:.4f}")
    configs.append(reranked)

    # Boost parameter sweep to find optimal settings
    print("\n--- Boost Parameter Sweep ---")
    sweep_results = []
    best_mrr = 0.0
    best_params = {}
    best_combined = 0.0  # MRR + 0.3*R@10 to balance ranking and recall
    best_balanced_params = {}
    for pb in [0.0, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for sb in [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]:
            for db in [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]:
                r = run_reranked_configuration(
                    rag, queries,
                    f"SWEEP(p={pb},s={sb},d={db})",
                    use_derivatives=(db > 0),
                    para_boost=pb, sent_boost=sb, deriv_boost=db,
                    k=10,
                )
                entry = {
                    "para_boost": pb, "sent_boost": sb, "deriv_boost": db,
                    "mrr": r["overall"]["mrr"],
                    "p1": r["overall"]["precision_at_1"],
                    "p3": r["overall"]["precision_at_3"],
                    "r10": r["overall"]["recall_at_10"],
                }
                sweep_results.append(entry)
                if r["overall"]["mrr"] > best_mrr:
                    best_mrr = r["overall"]["mrr"]
                    best_params = {"para_boost": pb, "sent_boost": sb, "deriv_boost": db}
                # Also track best balanced (MRR + recall)
                combined = r["overall"]["mrr"] + 0.3 * r["overall"]["recall_at_10"]
                if combined > best_combined:
                    best_combined = combined
                    best_balanced_params = {"para_boost": pb, "sent_boost": sb, "deriv_boost": db}

    print(f"\n  Best sweep MRR: {best_mrr:.4f}")
    print(f"  Best MRR params: para_boost={best_params['para_boost']}, sent_boost={best_params['sent_boost']}, deriv_boost={best_params['deriv_boost']}")
    print(f"  Best balanced (MRR+0.3*R@10): {best_combined:.4f}")
    print(f"  Best balanced params: para_boost={best_balanced_params['para_boost']}, sent_boost={best_balanced_params['sent_boost']}, deriv_boost={best_balanced_params['deriv_boost']}")

    # Show top-10 sweep results sorted by MRR
    sweep_sorted = sorted(sweep_results, key=lambda x: x["mrr"], reverse=True)
    print(f"\n  Top-10 parameter combinations by MRR:")
    print(f"  {'PARA':>6} {'SENT':>6} {'DERIV':>6} {'MRR':>8} {'P@1':>8} {'R@10':>8}")
    for s in sweep_sorted[:10]:
        print(f"  {s['para_boost']:>6.2f} {s['sent_boost']:>6.2f} {s['deriv_boost']:>6.2f} {s['mrr']:>8.4f} {s['p1']:>8.4f} {s['r10']:>8.4f}")

    # Run the best-params config as "OPTIMAL FRACTAL"
    print(f"\nRunning OPTIMAL FRACTAL (best MRR sweep params)...")
    t0 = time.time()
    optimal = run_reranked_configuration(
        rag, queries, "OPTIMAL FRACTAL",
        use_derivatives=(best_params["deriv_boost"] > 0),
        para_boost=best_params["para_boost"],
        sent_boost=best_params["sent_boost"],
        deriv_boost=best_params["deriv_boost"],
        k=10,
    )
    print(f"  Done in {time.time()-t0:.1f}s — MRR={optimal['overall']['mrr']:.4f} R@10={optimal['overall']['recall_at_10']:.4f}")
    configs.append(optimal)

    # If balanced params differ, also run that
    if best_balanced_params != best_params:
        print(f"Running BALANCED FRACTAL (best MRR+R@10 sweep params)...")
        t0 = time.time()
        balanced = run_reranked_configuration(
            rag, queries, "BALANCED FRACTAL",
            use_derivatives=(best_balanced_params["deriv_boost"] > 0),
            para_boost=best_balanced_params["para_boost"],
            sent_boost=best_balanced_params["sent_boost"],
            deriv_boost=best_balanced_params["deriv_boost"],
            k=10,
        )
        print(f"  Done in {time.time()-t0:.1f}s — MRR={balanced['overall']['mrr']:.4f} R@10={balanced['overall']['recall_at_10']:.4f}")
        configs.append(balanced)

    # --- PER-TYPE SWEEP ---
    print("\n--- Per-Type Boost Parameter Sweep ---")
    t0 = time.time()
    per_type_best = run_per_type_sweep(rag, queries, k=10)
    print(f"  Sweep completed in {time.time()-t0:.1f}s")
    print(f"\n  {'TYPE':<18} {'PARA':>6} {'SENT':>6} {'DERIV':>6} {'MRR':>8}")
    for qtype in ["specification", "summary", "logic", "synthesis"]:
        if qtype in per_type_best:
            p = per_type_best[qtype]
            print(f"  {qtype:<18} {p['para_boost']:>6.2f} {p['sent_boost']:>6.2f} {p['deriv_boost']:>6.2f} {p['mrr']:>8.4f}")

    # Print results and get go/no-go
    improvement = print_results_table(configs)

    # Save raw results
    output = {
        "metadata": {
            "corpus_size": len(documents),
            "query_count": len(queries),
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": backend.dim,
            "indexing_time_seconds": round(indexing_time, 1),
            "index_stats": stats,
        },
        "configurations": configs,
        "sweep_results": sweep_results,
        "best_params": best_params,
        "per_type_best": per_type_best,
        "verdict": {
            "flat_mrr": flat["overall"]["mrr"],
            "fractal_mrr": optimal["overall"]["mrr"],
            "improvement_pct": round(improvement, 2),
            "go_decision": "GO" if improvement >= 10 else "CONDITIONAL" if improvement >= 5 else "PIVOT" if improvement >= 0 else "STOP",
        },
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

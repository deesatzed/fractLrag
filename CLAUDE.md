# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Fractal Latent RAG** — a novel RAG architecture that uses multi-scale (fractal) indexing, latent space alignment, document-specific adapters, and deterministic derivative shorthands for richer retrieval and reasoning. Contains the validated `fractrag` package, benchmark infrastructure, and legacy demo scripts.

## Project Structure

```
fractrag/                    # Core package
├── __init__.py             # Public API exports
├── core.py                 # EmbeddingBackend protocol, HashEmbedding, SentenceTransformerEmbedding, normalize, chunk_fractal
├── engine.py               # FractalRAG class: retrieve(), retrieve_reranked(), retrieve_adaptive(), retrieve_rrf(), retrieve_flat()
├── profile.py              # DocumentProfile dataclass
├── query.py                # classify_query_type(), get_type_weights()
├── storage.py              # SQLite persistence (schema v2, doc_metadata + doc_titles tables)
└── benchmarks/
    ├── flat_vs_fractal.py  # Full ablation benchmark (7 configs + parameter sweep)
    └── cross_validation.py # 5-fold stratified CV + significance testing

corpus/                     # Real medical corpus (78 PubMed papers on AI in medicine)
├── medical_corpus.json     # 78 papers, 6 domains, real abstracts
├── medical_queries.json    # 40 content-grounded queries (4 types x 10)
├── cv_results.json         # Latest cross-validation results
├── benchmark_results.json  # Latest full benchmark results
├── fetch_pubmed.py         # PubMed E-utilities fetcher
└── build_grounded_queries.py  # Content-grounded query generator

tests/                      # 433 tests
legacy/                     # Historical PoC scripts (v1-v3, doctronic, xAI demos)
```

## Architecture: Fractal Latent RAG

### Core Concepts

- **Fractal multi-scale indexing**: Documents chunked at 3 self-similar levels — sentence (L0), paragraph (L1), document (L2). Retrieval operates identically at each scale.
- **Virtual paragraph chunking**: Single-paragraph docs with 4+ sentences get virtual L1 entries (groups of 3), preventing L1 degeneracy.
- **Title-prefix embeddings**: Optional document title prepended to L0/L1 text before embedding, enriching context without changing L2.
- **Document-specific adapters**: Per-document offset vectors injected at different strengths per level (0.6 sentence, 0.8 paragraph, full document).
- **Derivative shorthands**: 1st derivative = delta from paragraph mean (knowledge velocity). 2nd derivative = curvature (acceleration). Used as scoring bonuses.
- **Reranked retrieval**: Doc-level as primary ranking signal, sentence-level as boost. Preserves recall advantage while fixing ranking.
- **Reciprocal Rank Fusion (RRF)**: Rank-based fusion across levels (1/(k+rank)), more robust than linear boost combination.
- **Metadata filtering/boosting**: Domain, year, MeSH terms, journal filters narrow candidates; domain/mesh/recency boosts adjust scores.

### Retrieval Methods (engine.py)

| Method | Use Case | Strategy |
|--------|----------|----------|
| `retrieve()` | Raw multi-scale search | Scores all levels independently, configurable weights/derivatives |
| `retrieve_reranked()` | Improved ranking | Doc-level primary + sub-doc boost signals + derivative bonus + metadata boost |
| `retrieve_rrf()` | Robust fusion | Reciprocal Rank Fusion across levels, rank-based instead of score-based |
| `retrieve_adaptive()` | **Recommended** | Auto-classifies query type, routes spec→flat, others→reranked (or RRF with `use_rrf=True`) |
| `retrieve_flat()` | Baseline | Doc-level only, no fractal features |

### Embedding Backends

- `HashEmbedding(dim=64)` — Deterministic MD5-seeded random (testing only)
- `SentenceTransformerEmbedding("BAAI/bge-m3")` — Real 1024-dim semantic embeddings

## Benchmark Results

78 PubMed papers, 40 content-grounded queries, BGE-M3 embeddings.

### Cross-Validated (5-fold stratified CV, Phase 3A, 2026-04-22)

Parameters tuned on training folds only, evaluated on held-out folds.

| Metric | Flat | Fractal (CV) | Delta |
|--------|------|-------------|-------|
| MRR | 0.7517 | 0.8100 | +7.8% |
| p-value | — | 0.1395 | not significant |
| 95% CI | — | [-0.011, +0.139] | contains zero |

Per-type: spec +14.2%, logic +7.5%, summary +2.2%, synthesis -0.5%

Pre-3A baseline was +4.1% (p=0.060). Phase 3A nearly doubled the improvement.

### Non-CV (same-set, for reference only)

| Config | MRR | P@1 | R@10 |
|--------|-----|-----|------|
| FLAT | 0.7517 | 0.6500 | 0.7083 |
| RERANKED FRACTAL | 0.8067 | 0.7250 | 0.7833 |
| RRF FRACTAL | 0.7705 | 0.6750 | 0.7833 |
| OPTIMAL FRACTAL | 0.8412 | 0.7750 | 0.7917 |

The +11.9% non-CV number is inflated due to tuning on the evaluation set. The honest CV number is +7.8%.

## Running

```bash
# Install
pip install numpy sentence-transformers

# Run tests (403 tests)
pytest tests/ -v

# Run CV benchmark (BGE-M3, ~2 min)
python -m fractrag.benchmarks.cross_validation

# Run full ablation benchmark (~5 min)
python -m fractrag.benchmarks.flat_vs_fractal
```

## Dependencies

- **Required**: `numpy`
- **Embeddings**: `sentence-transformers` (for BGE-M3)
- **CV benchmark**: `scipy` (for paired t-test)
- **Testing**: `pytest`, `pytest-cov`
- Python 3.8+

## Key Design Decisions

- Virtual paragraph chunking restored non-degenerate L1 for 54% of corpus (single-paragraph docs)
- Title-prefix embeddings absorbed the sentence-boost signal (sent_boost dropped from 0.20 to 0.00 in CV)
- Derivative boost increased in importance post-3A (optimal shifted to deriv_boost=0.10)
- Adapter strength=0.25 adds diversity that the reranker can correct — acts as regularization
- Derivative bonus weight BASE_DERIV_WEIGHT=0.12 with type-aware multipliers
- Query classifier uses multi-signal scoring (100% accuracy on 40 benchmark queries)
- Content-grounded queries avoid circular evaluation (embedding model != relevance oracle)
- Raw multi-scale retrieval HURTS ranking — reranking is essential
- RRF underperformed reranked retrieval (0.7705 vs 0.8067) — linear boosts may be better tuned to this corpus

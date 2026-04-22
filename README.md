# FractRAG

**Fractal Latent RAG** — Multi-scale retrieval with derivative signals for richer document search.

FractRAG indexes documents at three self-similar levels (sentence, paragraph, document), computes derivative signals between levels, and uses a reranking strategy that outperforms flat single-scale retrieval.

## Quick Start

```bash
pip install numpy
pip install sentence-transformers  # optional: for real semantic embeddings
```

```python
from fractrag import FractalRAG, HashEmbedding

# Build index
rag = FractalRAG(backend=HashEmbedding(dim=64))
rag.add_document("doc1", "AI helps diagnose diseases using deep learning models.")
rag.add_document("doc2", "Rome fell in 476 AD after centuries of decline.")

# Retrieve (adaptive is recommended — auto-selects strategy per query type)
results, query_type = rag.retrieve_adaptive("How does AI help with diagnosis?")

# Results are keyed by level: {2: doc-level, 1: paragraph, 0: sentence}
for entry, score in results[2]:
    print(f"  {entry.id}: {score:.4f} — {entry.text[:80]}")
```

### With Real Embeddings (BGE-M3)

```python
from fractrag import FractalRAG, SentenceTransformerEmbedding

backend = SentenceTransformerEmbedding("BAAI/bge-m3")  # 1024-dim
rag = FractalRAG(backend=backend)
rag.add_document("doc1", "Your text here...")
results, qtype = rag.retrieve_adaptive("Your query here")
```

### With Metadata Filtering

```python
rag.add_document("doc1", "AI diagnosis paper...", metadata={
    "domain": "ai_diagnosis",
    "year": 2023,
    "mesh_terms": ["Artificial Intelligence", "Diagnosis"],
    "journal": "Nature Medicine",
})

# Filter by metadata
results, _ = rag.retrieve_reranked(
    "AI diagnosis methods",
    metadata_filters={"domain": "ai_diagnosis", "year_min": 2020},
    metadata_boost={"recency_boost": 0.3},
)
```

### Persistence

```python
from fractrag import save, load

save("my_index.db", rag)
rag2 = load("my_index.db", backend=backend)
```

## How It Works

### Fractal Multi-Scale Indexing

Documents are chunked at three self-similar levels:

| Level | Granularity | Adapter Strength | Purpose |
|-------|------------|-----------------|---------|
| L0 | Sentence | 0.6x | Fine-grained fact retrieval |
| L1 | Paragraph | 0.8x | Topical context |
| L2 | Document | 1.0x | Broad relevance |

Each level gets a per-document adapter vector (a learned offset that separates documents in embedding space).

### Derivative Signals

For each sentence, two derivative signals are computed:
- **1st derivative (d1)**: Delta from paragraph mean — measures "knowledge velocity" (how much a sentence deviates from its context)
- **2nd derivative (d2)**: Curvature — measures acceleration of concept shifts

These act as scoring bonuses during retrieval, not primary signals.

### Retrieval Methods

| Method | Strategy | When to Use |
|--------|----------|-------------|
| `retrieve_adaptive()` | Auto-classifies query type, routes to optimal strategy | **Recommended for all use** |
| `retrieve_reranked()` | Doc-level primary + sentence/paragraph boosts | When you want explicit control over boost params |
| `retrieve_flat()` | Doc-level only, no fractal features | Baseline comparison |
| `retrieve()` | Raw multi-scale with configurable weights | Research / ablation |

### Query Classification

Queries are automatically classified into four types, each with optimized retrieval:
- **Specification**: "What year did X happen?" — uses flat (precise, no noise)
- **Summary**: "Summarize the evidence on X" — uses fractal with sentence boost
- **Logic**: "Why does X cause Y?" — uses fractal with derivative emphasis
- **Synthesis**: "Compare X and Y across domains" — uses fractal with balanced boosts

### Metadata Filtering and Boosting

Documents can carry arbitrary metadata (domain, year, MeSH terms, journal, etc.). At retrieval time:
- **Filters** narrow candidates before scoring (domain, year_range, mesh_terms, journal)
- **Boosts** add score bonuses to matching documents (domain_boost, mesh_boost, recency_boost)

## Benchmark Results

78 PubMed papers (6 medical domains), 40 content-grounded queries, BGE-M3 embeddings (1024-dim).

### Cross-Validated (Honest Numbers)

5-fold stratified CV with grid-tuned boost parameters. Parameters tuned on training folds only, evaluated on held-out test folds.

| Metric | Flat | Fractal (CV) | Delta |
|--------|------|-------------|-------|
| MRR | 0.7517 | 0.7827 | +4.1% |
| p-value (paired t-test) | — | 0.060 | borderline |
| 95% Bootstrap CI | — | [+0.004, +0.066] | excludes zero |

Per query type (CV):

| Type | Flat MRR | Fractal MRR | Delta |
|------|----------|-------------|-------|
| Specification | 0.758 | 0.775 | +1.7% |
| Summary | 0.703 | 0.783 | +8.0% |
| Logic | 0.925 | 0.933 | +0.8% |
| Synthesis | 0.620 | 0.639 | +1.9% |

**Interpretation**: The fractal approach shows consistent positive improvement across all query types. The largest gain is on summary queries (+8.0%). The overall improvement is +4.1% after cross-validation (down from +7.3% without CV, confirming mild overfitting in the original evaluation). The bootstrap CI excludes zero, suggesting a real but small effect. The paired t-test is borderline (p=0.06).

### Tuned Parameters (Most Frequently Selected)

3 of 5 folds selected: `para_boost=0.05, sent_boost=0.20, deriv_boost=0.03`

The sentence-level boost is the most impactful fractal feature.

## API Reference

### `FractalRAG(backend, adapter_strength=0.25)`

Main engine class.

- `backend`: An `EmbeddingBackend` (e.g., `HashEmbedding(dim=64)` or `SentenceTransformerEmbedding("BAAI/bge-m3")`)
- `adapter_strength`: Per-document adapter magnitude (default 0.25)

### `rag.add_document(doc_id, text, profile=None, metadata=None)`

Index a document at all fractal levels.

- `metadata`: Optional dict with any keys. Recognized filter/boost keys: `domain`, `year`, `mesh_terms`, `journal`.

### `rag.retrieve_adaptive(query, k=10, metadata_filters=None, metadata_boost=None)`

Recommended retrieval method. Auto-classifies query type and selects optimal strategy.

Returns: `(results_by_level, detected_query_type)`

### `rag.retrieve_reranked(query, k=5, para_boost=0.15, sent_boost=0.10, deriv_boost=0.05, metadata_filters=None, metadata_boost=None)`

Reranked retrieval with explicit boost parameters.

### `rag.retrieve_flat(query, k=3, metadata_filters=None)`

Flat doc-level-only retrieval (baseline).

### `rag.stats()`

Returns dict with index statistics (document counts, entries per level, metadata count).

### `save(path, rag)` / `load(path, backend)`

SQLite persistence. Full round-trip equivalence.

## Running Benchmarks

```bash
# Cross-validation (requires BGE-M3, ~2 min)
python -m fractrag.benchmarks.cross_validation

# Full ablation study with parameter sweep (~5 min)
python -m fractrag.benchmarks.flat_vs_fractal
```

## Running Tests

```bash
pip install pytest scipy
pytest tests/ -v           # 403 tests
pytest tests/ -v --cov=fractrag  # with coverage
```

## Project Structure

```
fractrag/                    # Core package
  __init__.py               # Public API
  core.py                   # EmbeddingBackend protocol, chunking
  engine.py                 # FractalRAG class
  profile.py                # DocumentProfile dataclass
  query.py                  # Query classifier
  storage.py                # SQLite persistence
  benchmarks/
    flat_vs_fractal.py      # Full ablation benchmark
    cross_validation.py     # 5-fold CV + significance testing

corpus/                     # Benchmark data
  medical_corpus.json       # 78 PubMed papers
  medical_queries.json      # 40 content-grounded queries
  cv_results.json           # Latest CV results

tests/                      # 403 tests
legacy/                     # Historical PoC scripts (v1-v3)
```

## License

MIT

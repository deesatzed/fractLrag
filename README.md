# FractRAG

**Fractal Latent RAG** — Multi-scale retrieval with derivative signals for richer document search.

FractRAG indexes documents at three self-similar levels (sentence, paragraph, document), computes derivative signals between levels, and uses a reranking strategy that **significantly outperforms flat single-scale retrieval** (+10.4% MRR, p=0.036).

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
rag.add_document("doc1", "Your text here...", title="Document Title")
results, qtype = rag.retrieve_adaptive("Your query here")
```

### With Metadata Filtering and Domain Routing

```python
rag.add_document("doc1", "AI diagnosis paper...", metadata={
    "domain": "ai_diagnosis",
    "year": 2023,
    "mesh_terms": ["Artificial Intelligence", "Diagnosis"],
    "journal": "Nature Medicine",
}, title="AI in Clinical Diagnosis")

# Filter by metadata
results, _ = rag.retrieve_reranked(
    "AI diagnosis methods",
    metadata_filters={"domain": "ai_diagnosis", "year_min": 2020},
    metadata_boost={"recency_boost": 0.3},
)

# Domain routing is automatic in retrieve_adaptive()
# Queries mentioning "radiology", "drug discovery", etc. get auto-boosted
results, _ = rag.retrieve_adaptive("How does AI help in radiology imaging?")
```

### With Cross-Document Diversity (Synthesis Queries)

```python
# Synthesis queries automatically get domain-based diversity reranking
# This ensures top-k results span multiple domains instead of clustering
results, qtype = rag.retrieve_adaptive(
    "Compare AI applications across radiology and drug discovery"
)
# qtype == "synthesis" → diversity_boost=0.3 applied automatically

# Or set diversity explicitly on reranked retrieval
results, _ = rag.retrieve_reranked(
    "Compare clinical and pharma AI", k=10, diversity_boost=0.4,
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

**Virtual paragraph chunking**: Single-paragraph documents with 4+ sentences get synthetic L1 entries (groups of 3 sentences), preventing L1 degeneracy.

**Title-prefix embeddings**: When a title is provided, it's prepended to L0/L1 text before embedding, enriching context.

**Weighted paragraph pooling**: Paragraph contributions to the mean vector are weighted by similarity to the document vector, producing sharper derivative signals for short documents.

### Derivative Signals

For each sentence, two derivative signals are computed:
- **1st derivative (d1)**: Delta from paragraph mean — measures "knowledge velocity" (how much a sentence deviates from its context)
- **2nd derivative (d2)**: Curvature — measures acceleration of concept shifts

These act as scoring bonuses during retrieval, not primary signals.

### Retrieval Methods

| Method | Strategy | When to Use |
|--------|----------|-------------|
| `retrieve_adaptive()` | Auto-classifies query type, routes to optimal strategy | **Recommended for all use** |
| `retrieve_reranked()` | Doc-level primary + sentence/paragraph boosts + diversity | When you want explicit control over boost params |
| `retrieve_rrf()` | Reciprocal Rank Fusion across levels | Alternative fusion strategy |
| `retrieve_flat()` | Doc-level only, no fractal features | Baseline comparison |
| `retrieve()` | Raw multi-scale with configurable weights | Research / ablation |

### Query Classification

Queries are automatically classified into four types, each with optimized retrieval:
- **Specification**: "What year did X happen?" — uses flat (precise, no noise)
- **Summary**: "Summarize the evidence on X" — uses fractal with paragraph context
- **Logic**: "Why does X cause Y?" — uses fractal with high derivative emphasis
- **Synthesis**: "Compare X and Y across domains" — uses fractal with diversity reranking

### Per-Type Adaptive Boosts

Each query type gets independently tuned boost parameters:

| Type | para_boost | sent_boost | deriv_boost | diversity_boost |
|------|-----------|-----------|------------|----------------|
| Specification | 0.0 | 0.0 | 0.0 | 0.0 |
| Summary | 0.05 | 0.0 | 0.10 | 0.0 |
| Logic | 0.0 | 0.0 | 0.15 | 0.0 |
| Synthesis | 0.05 | 0.0 | 0.10 | 0.3 |

### Domain-Aware Query Routing

`retrieve_adaptive()` automatically detects domain keywords in queries across 6 medical categories (radiology, clinical decision, drug discovery, NLP, surgery/robotics, public health) and applies targeted metadata boosts without explicit configuration.

### Cross-Document Diversity Scoring

For synthesis queries, a greedy domain-based Maximal Marginal Relevance (MMR) reranking ensures the top-k results span multiple domains rather than clustering within one. This improved synthesis recall by +46%.

### Metadata Filtering and Boosting

Documents can carry arbitrary metadata (domain, year, MeSH terms, journal, etc.). At retrieval time:
- **Filters** narrow candidates before scoring (domain, year_range, mesh_terms, journal)
- **Boosts** add score bonuses to matching documents (domain_boost, mesh_boost, recency_boost)

## Benchmark Results

78 PubMed papers (6 medical domains), 40 content-grounded queries, BGE-M3 embeddings (1024-dim).

### Cross-Validated (Honest Numbers)

5-fold stratified CV with 150-combo grid-tuned boost parameters. Parameters tuned on training folds only, evaluated on held-out test folds.

| Metric | Flat | Fractal (CV) | Delta |
|--------|------|-------------|-------|
| MRR | 0.7517 | 0.8300 | **+10.4%** |
| p-value (paired t-test) | — | **0.0362** | **significant** |
| 95% Bootstrap CI | — | [+0.017, +0.153] | **excludes zero** |

Per query type (CV):

| Type | Flat MRR | Fractal MRR | Delta |
|------|----------|-------------|-------|
| Specification | 0.758 | 0.900 | +14.2% |
| Summary | 0.703 | 0.775 | +7.2% |
| Logic | 0.925 | 1.000 | +7.5% |
| Synthesis | 0.620 | 0.645 | +2.5% |

### Non-CV Ablation (for reference only)

| Config | MRR | P@1 | R@10 |
|--------|-----|-----|------|
| FLAT | 0.7517 | 0.6500 | 0.7083 |
| RERANKED FRACTAL | 0.8067 | 0.7250 | 0.7833 |
| RRF FRACTAL | 0.7705 | 0.6750 | 0.7833 |
| ADAPTIVE FRACTAL | 0.7952 | 0.7000 | 0.7917 |
| OPTIMAL FRACTAL | 0.8425 | 0.7750 | 0.7917 |

The +12.1% non-CV number is inflated due to tuning on the evaluation set. The honest CV number is +10.4%.

### Phase Progression

| Phase | CV MRR Delta | p-value | Key Changes |
|-------|-------------|---------|-------------|
| Phase 0 | +4.1% | 0.060 | Cross-validation baseline |
| Phase 3A | +7.8% | 0.139 | Virtual paragraphs, title-prefix, RRF |
| **Phase 3B** | **+10.4%** | **0.036** | Grid expansion, per-type boosts, diversity, domain routing, weighted pooling |

### Tuned Parameters (Most Frequently Selected in CV)

4 of 5 folds selected: `para_boost=0.00, sent_boost=0.05, deriv_boost=0.05`

Title-prefix embeddings absorbed the sentence-level signal (sent_boost dropped from 0.20 to 0.05).

## API Reference

### `FractalRAG(backend, adapter_strength=0.25)`

Main engine class.

- `backend`: An `EmbeddingBackend` (e.g., `HashEmbedding(dim=64)` or `SentenceTransformerEmbedding("BAAI/bge-m3")`)
- `adapter_strength`: Per-document adapter magnitude (default 0.25)

### `rag.add_document(doc_id, text, profile=None, metadata=None, title=None)`

Index a document at all fractal levels.

- `metadata`: Optional dict with any keys. Recognized filter/boost keys: `domain`, `year`, `mesh_terms`, `journal`.
- `title`: Optional document title. When provided, prepended to L0/L1 embeddings for contextual enrichment.

### `rag.retrieve_adaptive(query, k=10, metadata_filters=None, metadata_boost=None, use_rrf=False)`

Recommended retrieval method. Auto-classifies query type, selects optimal strategy, auto-detects domain hints, and applies per-type boosts including diversity for synthesis queries.

Returns: `(results_by_level, detected_query_type)`

### `rag.retrieve_reranked(query, k=5, para_boost=0.15, sent_boost=0.10, deriv_boost=0.05, diversity_boost=0.0, metadata_filters=None, metadata_boost=None)`

Reranked retrieval with explicit boost parameters. Set `diversity_boost > 0` for domain-based MMR reranking.

### `rag.retrieve_rrf(query, k=10, rrf_k=60, metadata_filters=None, metadata_boost=None)`

Reciprocal Rank Fusion across fractal levels.

### `rag.retrieve_flat(query, k=3, metadata_filters=None)`

Flat doc-level-only retrieval (baseline).

### `rag.stats()`

Returns dict with index statistics (document counts, entries per level, metadata count).

### `save(path, rag)` / `load(path, backend)`

SQLite persistence. Full round-trip equivalence.

### `extract_domain_hints(query)`

Detect domain keywords in a query string. Returns a metadata boost config dict or None.

```python
from fractrag.query import extract_domain_hints

hints = extract_domain_hints("How does AI help in radiology imaging?")
# {'domain_boost': 0.05, 'domain_target': 'ai_radiology_pathology'}
```

## Running Benchmarks

```bash
# Cross-validation (requires BGE-M3, ~17 min with 150-combo grid)
python -m fractrag.benchmarks.cross_validation

# Full ablation study with parameter sweep + per-type sweep (~5 min)
python -m fractrag.benchmarks.flat_vs_fractal
```

## Running Tests

```bash
pip install pytest scipy
pytest tests/ -v           # 470 tests
pytest tests/ -v --cov=fractrag  # with coverage
```

## Project Structure

```
fractrag/                    # Core package
  __init__.py               # Public API
  core.py                   # EmbeddingBackend protocol, chunking
  engine.py                 # FractalRAG class + diversity scoring
  profile.py                # DocumentProfile dataclass
  query.py                  # Query classifier + domain routing
  storage.py                # SQLite persistence
  benchmarks/
    flat_vs_fractal.py      # Full ablation + per-type sweep
    cross_validation.py     # 5-fold CV + per-type tuning + significance

corpus/                     # Benchmark data
  medical_corpus.json       # 78 PubMed papers
  medical_queries.json      # 40 content-grounded queries
  cv_results.json           # Latest CV results
  benchmark_results.json    # Latest ablation results

tests/                      # 470 tests
legacy/                     # Historical PoC scripts (v1-v3)
```

## Dependencies

- **Required**: `numpy`
- **Embeddings**: `sentence-transformers` (for BGE-M3)
- **CV benchmark**: `scipy` (for paired t-test)
- **Testing**: `pytest`, `pytest-cov`
- Python 3.8+

## License

MIT

#!/usr/bin/env python3
"""Per-query diagnostic: why is fractal not winning?"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from fractrag import FractalRAG, SentenceTransformerEmbedding

CORPUS_PATH = Path("/Volumes/WS4TB/repofractL/corpus/medical_corpus.json")
QUERIES_PATH = Path("/Volumes/WS4TB/repofractL/corpus/medical_queries.json")

corpus = json.loads(CORPUS_PATH.read_text())
queries = json.loads(QUERIES_PATH.read_text())
documents = corpus["documents"]
qs = queries["queries"]

print("Loading model...")
backend = SentenceTransformerEmbedding("BAAI/bge-m3")
rag = FractalRAG(backend=backend)
for doc in documents:
    rag.add_document(doc["doc_id"], doc["abstract"])

print(f"\nIndex: {rag.stats()}")

# For each query, check:
# 1. Where do the relevant docs rank in flat (L2 only)?
# 2. Where do they rank in multi-scale?
# 3. Are there sub-doc matches that SHOULD boost the relevant doc?

qvec_cache = {}
def get_qvec(text):
    if text not in qvec_cache:
        qvec_cache[text] = backend.embed(text)
    return qvec_cache[text]

print("\n" + "="*100)
print("PER-QUERY DIAGNOSTIC")
print("="*100)

for qtype in ["specification", "summary", "logic", "synthesis"]:
    type_qs = [q for q in qs if q["query_type"] == qtype]
    print(f"\n{'='*80}")
    print(f"QUERY TYPE: {qtype.upper()} ({len(type_qs)} queries)")
    print(f"{'='*80}")

    for q in type_qs:
        qtext = q["query_text"]
        relevant_pmids = set(f"pmid_{p}" for p in q["relevant_pmids"])
        qvec = get_qvec(qtext)

        # Doc-level scores for ALL docs
        doc_scores = []
        for entry in rag.index[2]:
            sim = float(np.dot(qvec, entry.vec))
            doc_scores.append((entry.id, sim))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Find rank of relevant docs
        relevant_ranks = {}
        for rank, (doc_id, score) in enumerate(doc_scores, 1):
            if doc_id in relevant_pmids:
                relevant_ranks[doc_id] = (rank, score)

        # Best sentence match for each relevant doc
        sent_scores_for_relevant = {}
        for entry in rag.index[0]:
            parent = entry.parent or entry.id
            if parent in relevant_pmids:
                sim = float(np.dot(qvec, entry.vec))
                if parent not in sent_scores_for_relevant or sim > sent_scores_for_relevant[parent]:
                    sent_scores_for_relevant[parent] = sim

        # Best sentence match across ALL docs (to see if irrelevant docs are winning)
        best_sent_overall = []
        for entry in rag.index[0]:
            parent = entry.parent or entry.id
            sim = float(np.dot(qvec, entry.vec))
            best_sent_overall.append((parent, sim, entry.text[:80]))
        best_sent_overall.sort(key=lambda x: x[1], reverse=True)

        # Print diagnostic
        print(f"\n  Q: {qtext[:90]}...")
        print(f"  Relevant: {relevant_pmids}")

        for doc_id, (rank, score) in sorted(relevant_ranks.items(), key=lambda x: x[1][0]):
            sent_best = sent_scores_for_relevant.get(doc_id, 0)
            print(f"    {doc_id}: doc_rank={rank}/78, doc_score={score:.4f}, best_sent={sent_best:.4f}")

        # Show if any irrelevant docs have better sentences
        if best_sent_overall:
            top3_sent = best_sent_overall[:3]
            top3_irrelevant = [(p, s, t) for p, s, t in top3_sent if p not in relevant_pmids]
            if top3_irrelevant:
                print(f"    Top irrelevant sentence matches:")
                for parent, sim, text in top3_irrelevant[:2]:
                    doc_rank = next((r for r, (d, _) in enumerate(doc_scores, 1) if d == parent), "?")
                    print(f"      {parent} (doc_rank={doc_rank}): sent_score={sim:.4f} — {text}")

print("\n\n" + "="*100)
print("SUMMARY: DISTRIBUTION OF RELEVANT DOC RANKS (doc-level only)")
print("="*100)

all_ranks = {"specification": [], "summary": [], "logic": [], "synthesis": []}
for q in qs:
    qvec = get_qvec(q["query_text"])
    relevant_pmids = set(f"pmid_{p}" for p in q["relevant_pmids"])
    doc_scores = []
    for entry in rag.index[2]:
        doc_scores.append((entry.id, float(np.dot(qvec, entry.vec))))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (doc_id, score) in enumerate(doc_scores, 1):
        if doc_id in relevant_pmids:
            all_ranks[q["query_type"]].append(rank)
            break  # first relevant only

for qtype, ranks in all_ranks.items():
    if ranks:
        print(f"\n  {qtype}: mean_rank={np.mean(ranks):.1f}, median={np.median(ranks):.1f}, "
              f"in_top5={sum(1 for r in ranks if r<=5)}/{len(ranks)}, "
              f"in_top10={sum(1 for r in ranks if r<=10)}/{len(ranks)}")
        print(f"    All ranks: {sorted(ranks)}")

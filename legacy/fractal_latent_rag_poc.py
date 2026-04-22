#!/usr/bin/env python3
"""
Fractal Latent RAG Proof of Concept
====================================
Implements the core ideas from the discussion:
- Fractal multi-scale indexing & retrieval (self-similar at sentence/para/doc levels)
- Latent space alignment (everything in one vector space simulating LLM latent)
- Document-specific adapters (mini fine-tuning simulation via learned offsets)
- Deterministic derivative shorthands (1st/2nd order deltas for richer matching & compression)
- Doc-specific embedding flavor via per-doc adapter injection

No external LLM or heavy deps beyond numpy (torch available but not required here).
Run: python3 fractal_latent_rag_poc.py
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple, Any

# ============================================================
# CONFIG
# ============================================================
DIM = 64                    # Latent dimension (simulates LLM hidden size)
ADAPTER_STRENGTH = 0.25     # How much doc-specific "mini FT" injection
DERIV_WEIGHT = 0.15         # Weight for derivative alignment in scoring

# ============================================================
# DETERMINISTIC LATENT EMBEDDING (simulates model latent space)
# ============================================================
def text_to_latent(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic embedding into 'latent space' via seeded random.
    In real system this would be the target LLM's hidden state projection."""
    seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ============================================================
# FRACTAL RAG CLASS
# ============================================================
class FractalLatentRAG:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.docs: Dict[str, str] = {}
        self.adapters: Dict[str, np.ndarray] = {}          # doc_id -> learned offset (mini FT)
        self.index: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}  # 0=sent, 1=para, 2=doc
        self.derivatives: Dict[str, Dict] = {}             # chunk_id -> {'d1': vec, 'd2': vec}

    def _chunk_fractal(self, text: str) -> Tuple[List[str], List[str], str]:
        """Fractal chunking: sentences -> paragraphs -> full doc (self-similar pattern)"""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()] or [text]
        return sentences, paragraphs, text

    def add_document(self, doc_id: str, text: str, learn_adapter: bool = True):
        """Add doc with full fractal indexing + doc-specific adapter + derivatives"""
        self.docs[doc_id] = text
        sentences, paragraphs, full_text = self._chunk_fractal(text)

        # === Level 2: Document (macro scale) ===
        doc_vec = text_to_latent(full_text)
        if learn_adapter:
            # Simulate mini fine-tuning: learn a small adapter from doc content
            adapter = text_to_latent(f"ADAPTER_{doc_id}_{full_text[:100]}") * ADAPTER_STRENGTH
            self.adapters[doc_id] = adapter
            doc_vec = normalize(doc_vec + adapter)
        self.index[2].append({
            'id': doc_id, 'level': 2, 'vec': doc_vec,
            'text': full_text[:180] + '...' if len(full_text) > 180 else full_text
        })

        # === Level 1: Paragraphs (meso scale) ===
        para_mean = np.zeros(self.dim)
        for i, para in enumerate(paragraphs):
            pvec = text_to_latent(para)
            adapter = self.adapters.get(doc_id, np.zeros(self.dim))
            pvec = normalize(pvec + adapter * 0.8)  # strong injection at para level
            pid = f"{doc_id}_p{i}"
            self.index[1].append({
                'id': pid, 'parent': doc_id, 'level': 1, 'vec': pvec,
                'text': para[:140] + '...' if len(para) > 140 else para
            })
            para_mean += pvec
        para_mean /= max(len(paragraphs), 1)

        # === Level 0: Sentences (micro scale) + compute derivatives ===
        sent_mean = np.zeros(self.dim)
        for i, sent in enumerate(sentences):
            svec = text_to_latent(sent)
            adapter = self.adapters.get(doc_id, np.zeros(self.dim))
            svec = normalize(svec + adapter * 0.6)
            sid = f"{doc_id}_s{i}"
            self.index[0].append({
                'id': sid, 'parent': doc_id, 'level': 0, 'vec': svec,
                'text': sent
            })
            sent_mean += svec

            # === Deterministic Derivative Shorthands ===
            # 1st deriv: delta from paragraph/doc mean (velocity of knowledge)
            d1 = normalize(svec - para_mean)
            # 2nd deriv: curvature (delta of deltas) - acceleration of concepts
            d2 = normalize(d1 - (svec - doc_vec))   # simple finite difference
            self.derivatives[sid] = {'d1': d1, 'd2': d2}

        sent_mean /= max(len(sentences), 1)

    def _score_with_derivatives(self, qvec: np.ndarray, item: Dict, level: int) -> float:
        """Enhanced scoring using latent alignment + derivative shorthands"""
        base_sim = np.dot(qvec, item['vec'])
        deriv_bonus = 0.0

        if item['id'] in self.derivatives:
            d = self.derivatives[item['id']]
            # Reward items whose 1st deriv aligns with query direction (change/emphasis)
            deriv_bonus += np.dot(qvec, d['d1']) * DERIV_WEIGHT
            # 2nd deriv captures "interesting curvature" (key relationships)
            deriv_bonus += np.dot(qvec, d['d2']) * (DERIV_WEIGHT * 0.6)

        return base_sim + deriv_bonus

    def retrieve(self, query: str, k: int = 3, levels: List[int] = None) -> Dict[int, List[Tuple[Dict, float]]]:
        """Fractal multi-scale retrieval: same logic at every level"""
        if levels is None:
            levels = [2, 1, 0]
        qvec = text_to_latent(query)

        results = {}
        for lvl in levels:
            candidates = self.index[lvl]
            scored = []
            for item in candidates:
                score = self._score_with_derivatives(qvec, item, lvl)
                scored.append((item, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results[lvl] = scored[:k]
        return results

    def generate(self, query: str, retrieved: Dict) -> str:
        """Mock generation that shows fractal synthesis (in real system: LLM call with latent injection)"""
        best_doc = retrieved.get(2, [({}, 0)])[0][0].get('text', 'N/A')
        best_para = retrieved.get(1, [({}, 0)])[0][0].get('text', 'N/A')
        best_sent = retrieved.get(0, [({}, 0)])[0][0].get('text', 'N/A')

        return (
            f"╔════════════════════════════════════════════════════════════╗\n"
            f"║  FRACTAL LATENT RAG RESPONSE (Multi-Scale Synthesis)       ║\n"
            f"╚════════════════════════════════════════════════════════════╝\n\n"
            f"Query: {query}\n\n"
            f"📘 DOC-LEVEL (Macro Context):\n   {best_doc}\n\n"
            f"📄 PARA-LEVEL (Meso Relationships):\n   {best_para}\n\n"
            f"🔹 SENT-LEVEL (Micro Precision + 1st/2nd Deriv):\n   {best_sent}\n\n"
            f"🧠 Latent Injection: Doc adapter + derivative shorthands applied.\n"
            f"   This is what a real LLM would receive in its hidden state.\n\n"
            f"[Simulated LLM Output]: The query is best answered by integrating\n"
            f"the high-level concept from the document, the specific mechanism\n"
            f"described in the paragraph, and the precise detail at sentence level\n"
            f"(boosted by derivative alignment for emphasis on change/relationships).\n"
        )

    def compare_flat_vs_fractal(self, query: str):
        """Show why fractal beats naive flat retrieval"""
        print("\n" + "="*70)
        print("COMPARISON: NAIVE FLAT (Doc-only) vs FULL FRACTAL LATENT RAG")
        print("="*70)
        qvec = text_to_latent(query)

        # Flat: only level 2
        flat = sorted(
            [(item, np.dot(qvec, item['vec'])) for item in self.index[2]],
            key=lambda x: x[1], reverse=True
        )[:2]

        print("\n🔸 NAIVE FLAT RETRIEVAL (only document level):")
        for item, score in flat:
            print(f"   Score: {score:.3f} | {item['text'][:80]}...")

        print("\n🔹 FULL FRACTAL (all scales + derivatives):")
        fractal = self.retrieve(query, k=2)
        for lvl in [2,1,0]:
            print(f"\n   Level {lvl}:")
            for item, score in fractal[lvl]:
                print(f"      {score:.3f} | {item['text'][:70]}...")

# ============================================================
# DEMO / INTERACTIVE
# ============================================================
if __name__ == "__main__":
    print("🚀 Building Fractal Latent RAG Proof of Concept...\n")

    rag = FractalLatentRAG(dim=DIM)

    # === Sample Corpus (diverse domains) ===
    docs = {
        "ai_overview": """Artificial intelligence is transforming every industry. 
Machine learning allows systems to learn patterns from data without explicit programming. 
Neural networks, inspired by the human brain, power modern deep learning breakthroughs. 
Large language models like GPT and Claude represent the current frontier.""",

        "biology_dna": """DNA serves as the fundamental blueprint of all known life forms. 
The double helix structure, discovered by Watson and Crick, encodes genetic information. 
During cell division, mitosis ensures accurate replication of chromosomes. 
Evolution acts through natural selection on genetic variation over generations.""",

        "history_rome": """The Western Roman Empire collapsed in 476 AD after centuries of decline. 
Economic troubles, barbarian invasions, and political instability were key factors. 
The Renaissance later revived classical art, science, and humanism in Europe. 
World War II, ending in 1945, reshaped global power structures forever."""
    }

    for doc_id, text in docs.items():
        rag.add_document(doc_id, text)
        print(f"✅ Indexed: {doc_id} (with doc-specific adapter + derivatives)")

    print("\n" + "="*70)
    print("INTERACTIVE DEMO - Enter queries (or 'quit' to exit)")
    print("="*70)

    # Pre-run one comparison
    example_query = "How does machine learning enable AI systems to learn?"
    rag.compare_flat_vs_fractal(example_query)

    print("\n" + "-"*70)
    print("Now try your own queries to see fractal advantages!")
    print("-"*70)

    while True:
        try:
            user_query = input("\n🔍 Enter query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            if not user_query:
                continue

            retrieved = rag.retrieve(user_query, k=2)
            print(rag.generate(user_query, retrieved))

            # Bonus: show derivative insight
            print("📊 Derivative Insight (2nd order curvature on top result):")
            top_sent = retrieved[0][0][0] if retrieved[0] else None
            if top_sent and top_sent['id'] in rag.derivatives:
                d2_norm = np.linalg.norm(rag.derivatives[top_sent['id']]['d2'])
                print(f"   2nd derivative magnitude: {d2_norm:.3f} (higher = richer relationships)")

        except KeyboardInterrupt:
            break

    print("\n✅ PoC complete. This demonstrates the fractal + latent + derivative approach in ~150 lines.")
    print("   In production: replace text_to_latent with real LLM hidden states,")
    print("   add real LoRA adapters, and call actual LLM for final generation.")
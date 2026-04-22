#!/usr/bin/env python3
"""
Build content-grounded queries from actual abstract content.

Strategy: For each document, extract a distinctive sentence/fact from the abstract,
then create a query that would require finding THAT specific content.
This ensures relevance labels are grounded in real content, not embedding distance.

Query types:
  - specification: asks about a specific detail mentioned in one paper
  - summary: asks for an overview that maps to 2-3 papers in a domain
  - logic: asks WHY/HOW about a specific mechanism described in one paper
  - synthesis: asks to connect findings across 2-3 papers from different domains

The key design principle: the query should be SEMANTICALLY RELATED but NOT
a verbatim copy of the abstract text. This tests whether the retrieval
system can find relevant content through semantic understanding, not just
keyword matching.
"""

import json
import re
from pathlib import Path


def extract_key_sentences(abstract: str) -> list[str]:
    """Split abstract into sentences and return non-trivial ones."""
    # Split on period followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)
    # Filter out very short sentences and section headers
    return [s.strip() for s in sentences
            if len(s.strip()) > 40
            and not s.strip().startswith(('METHODS:', 'RESULTS:', 'CONCLUSIONS:',
                                          'BACKGROUND:', 'OBJECTIVE:', 'PURPOSE:',
                                          'STUDY OBJECTIVES:', 'INTRODUCTION:',
                                          'IMPORTANCE:', 'DESIGN:', 'SETTING:'))]


def build_queries(corpus_path: Path) -> dict:
    corpus = json.loads(corpus_path.read_text())
    docs = corpus["documents"]

    # Group by domain
    by_domain = {}
    for doc in docs:
        by_domain.setdefault(doc["domain"], []).append(doc)

    queries = []

    # === SPECIFICATION QUERIES (10) ===
    # Ask about specific details from individual papers
    spec_targets = [
        # (pmid, query_text) — crafted by reading the abstract
        ("36246896", "What retinal condition does deep learning help screen for in diabetes patients?"),
        ("36856067", "How many studies were included in the systematic review of machine learning for sleep disorders?"),
        ("36930153", "What is the systematic review about regarding AI algorithms for hip fracture detection?"),
        ("35731103", "What types of fractures does the overview of AI diagnostic products focus on?"),
        ("38663302", "What clinical setting does the paper discuss AI transformation for?"),
        ("35939288", "What type of therapy does the randomized trial compare using AI and mobile health?"),
        ("37458400", "What specific type of drug interactions does the AI prediction review examine?"),
        ("36732152", "What data mining approach is used for pharmacovigilance of drug interactions?"),
        ("33421167", "In what medical specialty is artificial intelligence being applied for diagnosis and research?"),
        ("38486342", "What brain tumor type does the AI-based MRI radiomics study focus on?"),
    ]

    for pmid, qtext in spec_targets:
        queries.append({
            "query_text": qtext,
            "query_type": "specification",
            "relevant_pmids": [pmid],
            "grounding": "content-based"
        })

    # === SUMMARY QUERIES (10) ===
    # Ask for domain overviews — relevant docs are same-domain papers
    summary_targets = [
        # Domain ai_diagnosis
        ("What approaches have been used for AI-based clinical diagnosis in medical imaging and pathology?",
         ["36246896", "34957756", "36930153"]),
        ("What is the current evidence on machine learning for cardiovascular and respiratory disease detection?",
         ["34957756", "36856067", "38114983"]),
        # Domain ai_physician
        ("How are AI clinical decision support systems being developed for physician practice?",
         ["36315554", "37543707", "40036494"]),
        ("What are the perspectives on ethical implications of AI in healthcare decision making?",
         ["39707327", "40775927", "38920162"]),
        # Domain ai_triage
        ("What is the evidence on AI applications in emergency department triage?",
         ["39944197", "39262027", "39965433"]),
        ("How is AI being applied to triage patients with acute conditions in emergencies?",
         ["41795351", "33243898", "39558305"]),
        # Domain ai_care
        ("What role does AI play in improving patient care delivery and treatment management?",
         ["38589940", "32325045", "34272911"]),
        ("How is AI being used for medication management and clinical pharmacy applications?",
         ["38049066", "32572267", "35939288"]),
        # Domain ai_drug_safety
        ("What are recent advances in using AI for drug safety and pharmacovigilance?",
         ["35904529", "31383376", "39961738"]),
        # Domain ai_radiology_pathology
        ("How is AI being applied in radiology and pathology for cancer diagnosis?",
         ["36220072", "37493248", "37099398"]),
    ]

    for qtext, pmids in summary_targets:
        queries.append({
            "query_text": qtext,
            "query_type": "summary",
            "relevant_pmids": pmids,
            "grounding": "content-based"
        })

    # === LOGIC QUERIES (10) ===
    # WHY/HOW questions grounded in specific paper content
    logic_targets = [
        # Each query asks about a mechanism or finding described in one specific paper
        ("36246896", "Why is deep learning considered effective for large-scale retinal disease screening?"),
        ("36199569", "Why is explainability important for trust in AI clinical decision support?"),
        ("38589940", "What are the obstacles preventing wider adoption of AI in critical care settings?"),
        ("32572267", "How does human-computer collaboration improve skin cancer recognition?"),
        ("37458400", "How do AI models predict drug-drug interactions in clinical pharmacy?"),
        ("39944197", "How are machine learning methods applied to improve emergency medicine outcomes?"),
        ("36220072", "How does multimodal data integration enhance AI performance in oncology?"),
        ("37493248", "Why is interpretability crucial for AI models used in radiology and radiation oncology?"),
        ("39965433", "What are the benefits and challenges of AI-driven triage in emergency departments?"),
        ("31383376", "How does artificial intelligence assist in detecting drug toxicity and ensuring safety?"),
    ]

    for pmid, qtext in logic_targets:
        queries.append({
            "query_text": qtext,
            "query_type": "logic",
            "relevant_pmids": [pmid],
            "grounding": "content-based"
        })

    # === SYNTHESIS QUERIES (10) ===
    # Cross-domain queries requiring info from 2-3 papers in different domains
    synthesis_targets = [
        ("How do AI diagnostic tools and AI-driven triage systems work together to improve patient outcomes?",
         ["34957756", "39944197", "38589940"]),  # diagnosis + triage + care
        ("Compare the challenges of implementing AI in radiology versus emergency triage settings.",
         ["37099398", "39262027", "37493248"]),  # radiology + triage + interpretability
        ("What are the common barriers to clinical adoption of AI across diagnosis, triage, and treatment?",
         ["38920162", "39707327", "38589940"]),  # ethics + physician + care
        ("How does AI impact both drug safety monitoring and clinical decision support for physicians?",
         ["35904529", "37543707", "38049066"]),  # drug_safety + physician + care
        ("What evidence exists for AI improving accuracy in both medical imaging and emergency triage?",
         ["32941736", "39965433", "36930153"]),  # radiology + triage + diagnosis
        ("How are deep learning and machine learning being applied across different medical specialties?",
         ["36246896", "36220072", "33243898"]),  # diagnosis + radiology + triage
        ("What role does data integration play in advancing AI for both oncology and pharmacovigilance?",
         ["36220072", "36732152", "39961738"]),  # radiology + drug_safety x2
        ("Compare AI's effectiveness in detecting skin conditions versus cardiac conditions.",
         ["32572267", "34957756", "33421167"]),  # care + diagnosis x2
        ("How do ethical concerns about AI differ between resource allocation and clinical diagnosis?",
         ["39707327", "38920162", "36315554"]),  # physician x3 (different focus)
        ("What can emergency medicine learn from AI advances in both radiology and pharmacovigilance?",
         ["35933269", "35904529", "39944197"]),  # radiology + drug_safety + triage
    ]

    for qtext, pmids in synthesis_targets:
        queries.append({
            "query_text": qtext,
            "query_type": "synthesis",
            "relevant_pmids": pmids,
            "grounding": "content-based"
        })

    return {
        "metadata": {
            "created": "2026-04-21",
            "total_queries": len(queries),
            "annotation_method": "content_grounded",
            "annotation_note": "Queries crafted from actual abstract content. Relevance based on topical match to specific paper content, not embedding distance.",
            "types": {
                "specification": 10,
                "summary": 10,
                "logic": 10,
                "synthesis": 10,
            }
        },
        "queries": queries,
    }


if __name__ == "__main__":
    corpus_path = Path(__file__).parent / "medical_corpus.json"
    result = build_queries(corpus_path)
    out_path = Path(__file__).parent / "medical_queries.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Generated {result['metadata']['total_queries']} content-grounded queries")
    print(f"Saved to {out_path}")

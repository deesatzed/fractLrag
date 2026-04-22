#!/usr/bin/env python3
"""
Fetch real PubMed papers on AI in Medicine for Fractal RAG validation.

Topics serve dual purpose:
1. Benchmark corpus for fractal vs flat RAG hypothesis testing
2. Domain knowledge for Doctronic (AI physician, AI diagnosis, AI triage, AI care)

Uses PubMed E-utilities API (public, no key required at <=3 req/sec).
"""

import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CORPUS_DIR = Path(__file__).parent
OUTPUT_FILE = CORPUS_DIR / "medical_corpus.json"
QUERIES_FILE = CORPUS_DIR / "medical_queries.json"

# AI-in-medicine search topics — 6 domains, ~13 papers each = ~78 target
SEARCH_TOPICS = {
    "ai_diagnosis": [
        "artificial intelligence clinical diagnosis",
        "machine learning diagnostic accuracy",
        "deep learning medical imaging diagnosis",
    ],
    "ai_physician": [
        "AI clinical decision support physician",
        "large language model physician assistant",
        "artificial intelligence primary care",
    ],
    "ai_triage": [
        "artificial intelligence emergency triage",
        "machine learning patient triage emergency department",
        "AI-powered clinical triage acuity",
    ],
    "ai_care": [
        "artificial intelligence patient care management",
        "AI remote patient monitoring",
        "machine learning treatment recommendation",
    ],
    "ai_drug_safety": [
        "artificial intelligence adverse drug reaction",
        "machine learning pharmacovigilance",
        "AI antibiotic stewardship",
    ],
    "ai_radiology_pathology": [
        "artificial intelligence radiology interpretation",
        "deep learning pathology slide analysis",
        "AI medical image analysis clinical",
    ],
}

PAPERS_PER_DOMAIN = 13


def esearch(query: str, retmax: int = 30) -> list:
    """Search PubMed and return PMIDs."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{BASE}/esearch.fcgi?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_batch(pmids: list) -> str:
    """Fetch full records for a batch of PMIDs as XML."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    })
    url = f"{BASE}/efetch.fcgi?{params}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def parse_articles(xml_text: str) -> list:
    """Parse PubMed XML into structured article dicts."""
    root = ET.fromstring(xml_text)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            medline = article_elem.find("MedlineCitation")
            if medline is None:
                continue

            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                continue

            art = medline.find("Article")
            if art is None:
                continue

            # Title
            title_elem = art.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            if not title:
                continue

            # Abstract — concatenate all AbstractText elements
            abstract_parts = []
            abstract_elem = art.find("Abstract")
            if abstract_elem is not None:
                for at in abstract_elem.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = "".join(at.itertext()).strip()
                    if text:
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
            abstract = "\n\n".join(abstract_parts)
            if len(abstract) < 100:
                continue  # Skip papers without meaningful abstracts

            # Authors (first 5)
            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for auth in author_list.findall("Author")[:5]:
                    last = auth.find("LastName")
                    first = auth.find("ForeName")
                    if last is not None and last.text:
                        name = last.text
                        if first is not None and first.text:
                            name = f"{first.text} {last.text}"
                        authors.append(name)

            # Journal
            journal_elem = art.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown"

            # Year
            year = 0
            pub_date = art.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    year = int(year_elem.text)

            # MeSH terms
            mesh_terms = []
            mesh_list = medline.find("MeshHeadingList")
            if mesh_list is not None:
                for mh in mesh_list.findall("MeshHeading"):
                    desc = mh.find("DescriptorName")
                    if desc is not None and desc.text:
                        mesh_terms.append(desc.text)

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "year": year,
                "mesh_terms": mesh_terms,
            })
        except Exception as e:
            print(f"  Warning: Skipping article due to parse error: {e}")
            continue

    return articles


def fetch_domain(domain: str, queries: list, target_count: int) -> list:
    """Fetch papers for a domain using multiple search queries."""
    seen_pmids = set()
    all_articles = []

    for query in queries:
        if len(all_articles) >= target_count:
            break

        print(f"  Searching: {query}")
        time.sleep(0.4)  # Rate limit

        pmids = esearch(query, retmax=25)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        if not new_pmids:
            continue

        seen_pmids.update(new_pmids)
        time.sleep(0.4)

        xml = efetch_batch(new_pmids)
        articles = parse_articles(xml)

        for art in articles:
            if len(all_articles) >= target_count:
                break
            art["domain"] = domain
            art["doc_id"] = f"pmid_{art['pmid']}"
            all_articles.append(art)

        print(f"    Got {len(articles)} papers (total for domain: {len(all_articles)})")

    return all_articles[:target_count]


def build_queries_from_corpus(documents: list) -> list:
    """Build query set from actual fetched paper content.
    Each query references real PMIDs and expected terms from the corpus.
    """
    queries = []

    # Group docs by domain
    by_domain = {}
    for doc in documents:
        by_domain.setdefault(doc["domain"], []).append(doc)

    # --- SPECIFICATION queries (10) ---
    spec_templates = [
        ("What is the reported diagnostic accuracy of AI systems for {topic}?",
         ["accuracy", "sensitivity", "specificity", "AUC"]),
        ("What specific machine learning algorithm is used for {topic}?",
         ["algorithm", "model", "neural network", "random forest", "deep learning"]),
        ("What is the sample size used in the study on {topic}?",
         ["patients", "samples", "cohort", "participants", "dataset"]),
        ("What are the exact performance metrics reported for AI in {topic}?",
         ["precision", "recall", "F1", "accuracy", "AUROC"]),
        ("What specific imaging modality is used for AI diagnosis of {topic}?",
         ["CT", "MRI", "X-ray", "ultrasound", "mammography"]),
    ]

    for i, (template, terms) in enumerate(spec_templates):
        domain_docs = list(by_domain.values())[i % len(by_domain)]
        if not domain_docs:
            continue
        doc = domain_docs[i % len(domain_docs)]
        topic = doc["title"].split(":")[0] if ":" in doc["title"] else doc["title"][:60]
        queries.append({
            "query_text": template.format(topic=topic.lower()),
            "query_type": "specification",
            "relevant_pmids": [doc["pmid"]],
            "expected_terms": terms,
        })

    # Pad to 10 if needed
    while len([q for q in queries if q["query_type"] == "specification"]) < 10:
        doc = documents[len(queries) % len(documents)]
        queries.append({
            "query_text": f"What are the exact clinical criteria described in the study: {doc['title'][:80]}?",
            "query_type": "specification",
            "relevant_pmids": [doc["pmid"]],
            "expected_terms": ["criteria", "clinical", "diagnosis", "threshold"],
        })

    # --- SUMMARY queries (10) ---
    summary_templates = [
        "Summarize the current role of artificial intelligence in {domain_label}.",
        "What are the main findings regarding AI applications in {domain_label}?",
        "Provide an overview of machine learning approaches to {domain_label}.",
        "What are the key benefits and limitations of AI in {domain_label}?",
        "Summarize recent advances in AI-assisted {domain_label}.",
    ]

    domain_labels = {
        "ai_diagnosis": "clinical diagnosis",
        "ai_physician": "physician decision support",
        "ai_triage": "emergency triage",
        "ai_care": "patient care management",
        "ai_drug_safety": "drug safety monitoring",
        "ai_radiology_pathology": "medical imaging analysis",
    }

    for i, template in enumerate(summary_templates):
        domain = list(by_domain.keys())[i % len(by_domain)]
        domain_docs = by_domain[domain]
        queries.append({
            "query_text": template.format(domain_label=domain_labels.get(domain, domain)),
            "query_type": "summary",
            "relevant_pmids": [d["pmid"] for d in domain_docs[:3]],
            "expected_terms": ["AI", "machine learning", "clinical", "patient"],
        })

    while len([q for q in queries if q["query_type"] == "summary"]) < 10:
        domain = list(by_domain.keys())[len(queries) % len(by_domain)]
        queries.append({
            "query_text": f"What is the current state of AI research in {domain_labels.get(domain, domain)}?",
            "query_type": "summary",
            "relevant_pmids": [d["pmid"] for d in by_domain[domain][:2]],
            "expected_terms": ["AI", "clinical", "study", "results"],
        })

    # --- LOGIC queries (10) ---
    logic_templates = [
        "How does deep learning improve diagnostic accuracy compared to traditional methods in {topic}?",
        "Why do AI triage systems outperform manual triage in emergency departments?",
        "How does the mechanism of AI-assisted drug interaction detection work?",
        "Why is AI particularly effective for pattern recognition in medical imaging?",
        "How does natural language processing enable extraction of clinical insights from EHR data?",
        "Why do ensemble methods tend to outperform single classifiers in clinical prediction?",
        "How does transfer learning reduce the need for labeled medical data?",
        "Why is explainability crucial for AI adoption in clinical decision support?",
        "How does federated learning address privacy concerns in multi-site medical AI studies?",
        "Why do AI systems sometimes fail at generalizing across different patient populations?",
    ]

    for i, template in enumerate(logic_templates):
        domain_docs = list(by_domain.values())[i % len(by_domain)]
        doc = domain_docs[i % len(domain_docs)] if domain_docs else documents[i]
        topic = doc["title"][:50]
        queries.append({
            "query_text": template.format(topic=topic) if "{topic}" in template else template,
            "query_type": "logic",
            "relevant_pmids": [doc["pmid"]],
            "expected_terms": ["mechanism", "accuracy", "comparison", "performance", "clinical"],
        })

    # --- SYNTHESIS queries (10) ---
    synthesis_templates = [
        "Compare AI-based and traditional approaches to clinical diagnosis across multiple specialties.",
        "Discuss the relationship between AI triage accuracy and patient outcomes in emergency settings.",
        "Integrate findings on AI in radiology, pathology, and clinical decision support — what patterns emerge?",
        "Discuss how AI physician assistants and AI triage systems complement each other in healthcare delivery.",
        "Compare the effectiveness of AI in drug safety monitoring versus traditional pharmacovigilance methods.",
        "Discuss the trade-offs between AI diagnostic accuracy and clinical interpretability across medical domains.",
        "How do AI care management systems and AI diagnostic tools work together to improve patient outcomes?",
        "Compare the challenges of implementing AI in primary care versus emergency medicine.",
        "Discuss the evolving role of AI across the patient journey: from triage to diagnosis to treatment.",
        "Integrate evidence on AI in medicine — what are the common success factors and failure modes?",
    ]

    for i, template in enumerate(synthesis_templates):
        # Synthesis queries should span multiple domains
        relevant_docs = []
        for domain_docs in by_domain.values():
            if domain_docs:
                relevant_docs.append(domain_docs[i % len(domain_docs)])
            if len(relevant_docs) >= 3:
                break
        queries.append({
            "query_text": template,
            "query_type": "synthesis",
            "relevant_pmids": [d["pmid"] for d in relevant_docs],
            "expected_terms": ["AI", "clinical", "patient", "outcome", "diagnosis"],
        })

    return queries


def main():
    print("=" * 70)
    print("PUBMED CORPUS FETCH — AI in Medicine")
    print(f"Target: {PAPERS_PER_DOMAIN} papers x {len(SEARCH_TOPICS)} domains = {PAPERS_PER_DOMAIN * len(SEARCH_TOPICS)} papers")
    print("=" * 70)

    all_documents = []

    for domain, queries in SEARCH_TOPICS.items():
        print(f"\n--- Domain: {domain} ---")
        articles = fetch_domain(domain, queries, PAPERS_PER_DOMAIN)
        all_documents.extend(articles)
        print(f"  Total fetched for {domain}: {len(articles)}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL DOCUMENTS FETCHED: {len(all_documents)}")

    # Summary by domain
    domain_counts = {}
    for doc in all_documents:
        domain_counts[doc["domain"]] = domain_counts.get(doc["domain"], 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} papers")

    # Abstract length stats
    abstract_lens = [len(doc["abstract"]) for doc in all_documents]
    print(f"\nAbstract lengths: min={min(abstract_lens)}, max={max(abstract_lens)}, "
          f"avg={sum(abstract_lens)//len(abstract_lens)}")

    # Save corpus
    corpus = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d"),
            "source": "PubMed E-utilities API (real data, not mock)",
            "total_documents": len(all_documents),
            "domains": list(SEARCH_TOPICS.keys()),
            "purpose": "Fractal RAG validation + Doctronic AI-medicine domain knowledge",
        },
        "documents": all_documents,
    }

    OUTPUT_FILE.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
    print(f"\nCorpus saved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # Build and save queries
    query_set = build_queries_from_corpus(all_documents)
    query_data = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d"),
            "total_queries": len(query_set),
            "types": {qt: len([q for q in query_set if q["query_type"] == qt])
                      for qt in ["specification", "summary", "logic", "synthesis"]},
        },
        "queries": query_set,
    }

    QUERIES_FILE.write_text(json.dumps(query_data, indent=2, ensure_ascii=False))
    print(f"Queries saved to: {QUERIES_FILE}")
    print(f"Query breakdown: {query_data['metadata']['types']}")

    print(f"\n{'=' * 70}")
    print("FETCH COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

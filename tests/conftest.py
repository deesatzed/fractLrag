"""
Shared fixtures for Fractal Latent RAG test suite.
All fixtures use real data and real computations -- no mocks.
"""
import sys
import os
import pytest
import numpy as np

# Add the project root and legacy/ to the Python path so we can import all modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "legacy"))


# ============================================================
# SAMPLE DOCUMENTS (real text, diverse domains)
# ============================================================
SAMPLE_DOCS = {
    "ai_overview": (
        "Artificial intelligence is transforming every industry. "
        "Machine learning allows systems to learn patterns from data without explicit programming. "
        "Neural networks, inspired by the human brain, power modern deep learning breakthroughs. "
        "Large language models like GPT and Claude represent the current frontier."
    ),
    "biology_dna": (
        "DNA serves as the fundamental blueprint of all known life forms. "
        "The double helix structure, discovered by Watson and Crick, encodes genetic information. "
        "During cell division, mitosis ensures accurate replication of chromosomes. "
        "Evolution acts through natural selection on genetic variation over generations."
    ),
    "history_rome": (
        "The Western Roman Empire collapsed in 476 AD after centuries of decline. "
        "Economic troubles, barbarian invasions, and political instability were key factors. "
        "The Renaissance later revived classical art, science, and humanism in Europe. "
        "World War II, ending in 1945, reshaped global power structures forever."
    ),
}

SAMPLE_DOCS_WITH_PARAGRAPHS = {
    "multi_para": (
        "Artificial intelligence is transforming every industry.\n\n"
        "Machine learning allows systems to learn patterns from data.\n\n"
        "Neural networks power modern deep learning breakthroughs."
    ),
}


# ============================================================
# SAMPLE QUERIES BY TYPE
# ============================================================
SPECIFICATION_QUERIES = [
    "What is the exact date the Western Roman Empire collapsed?",
    "List the key factors in Rome's decline.",
    "How many years do confidentiality obligations survive?",
    "Name the specific structure of DNA.",
]

SUMMARY_QUERIES = [
    "Summarize the key points about DNA and evolution.",
    "Give a brief overview of artificial intelligence.",
    "What are the main points about the Roman Empire?",
]

LOGIC_QUERIES = [
    "How does machine learning enable AI systems to learn patterns?",
    "Why did the Western Roman Empire collapse?",
    "Compare natural selection and genetic variation.",
]

SYNTHESIS_QUERIES = [
    "Discuss the major factors that led to Rome's fall and modern parallels.",
    "Integrate AI advances with biological evolution concepts.",
    "What are the overall themes across technology and history?",
]


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def sample_docs():
    """Return the standard 3-document test corpus."""
    return SAMPLE_DOCS.copy()


@pytest.fixture
def sample_docs_with_paragraphs():
    """Return documents that contain paragraph breaks."""
    return SAMPLE_DOCS_WITH_PARAGRAPHS.copy()


@pytest.fixture
def v1_engine_loaded():
    """Return a FractalLatentRAG engine preloaded with the 3 sample docs."""
    from fractal_latent_rag_poc import FractalLatentRAG
    rag = FractalLatentRAG(dim=64)
    for doc_id, text in SAMPLE_DOCS.items():
        rag.add_document(doc_id, text)
    return rag


@pytest.fixture
def v2_engine_loaded():
    """Return a FractalSOTARAG engine preloaded with the 3 sample docs."""
    from fractal_sota_rag_poc import FractalSOTARAG
    rag = FractalSOTARAG()
    for doc_id, text in SAMPLE_DOCS.items():
        rag.add_document(doc_id, text)
    return rag


@pytest.fixture
def v3_engine_loaded():
    """Return a FractalSOTARAGv3 engine preloaded with the 3 sample docs."""
    from fractal_sota_rag_v3 import FractalSOTARAGv3, DocumentProfile
    rag = FractalSOTARAGv3()
    profiles = {
        "ai_overview": DocumentProfile(accuracy_importance="high", complexity_level="medium", domain="technology"),
        "biology_dna": DocumentProfile(accuracy_importance="high", complexity_level="high", domain="science"),
        "history_rome": DocumentProfile(accuracy_importance="medium", complexity_level="medium", domain="history"),
    }
    for doc_id, text in SAMPLE_DOCS.items():
        rag.add_document(doc_id, text, profiles[doc_id])
    return rag


@pytest.fixture
def xai_engine_loaded():
    """Return an xAIKnowledgeEngine preloaded with the demo docs."""
    from xai_musk_knowledge_engine_demo import xAIKnowledgeEngine, MuskProfile
    engine = xAIKnowledgeEngine()
    docs = {
        "mars_habitat": {
            "text": "Mars Habitat Alpha-7 contract requires 120 days notice for termination. Structural integrity must withstand 0.38g gravity and 210 K temperature swings. Life support redundancy factor shall be minimum 2.5.",
            "profile": MuskProfile(accuracy_importance="critical", complexity_level="high", domain="mars_habitat", mission_critical=True)
        },
        "tesla_4680": {
            "text": "4680 cell production yield target is 92% by Q4 2026. Energy density 350 Wh/kg at cell level. Cycle life 1500+ at 80% DoD.",
            "profile": MuskProfile(accuracy_importance="high", complexity_level="high", domain="tesla_battery", mission_critical=True)
        },
    }
    for did, data in docs.items():
        engine.add(did, data["text"], data["profile"])
    return engine


@pytest.fixture
def unified_engine_loaded():
    """Return a FractalRAG (unified) engine preloaded with the 3 sample docs."""
    from fractrag import FractalRAG, HashEmbedding
    rag = FractalRAG(backend=HashEmbedding(dim=64))
    for doc_id, text in SAMPLE_DOCS.items():
        rag.add_document(doc_id, text)
    return rag


@pytest.fixture
def doctronic_engine():
    """Return a fresh DoctronicPrimaryCareEngine."""
    from doctronic_primary_care_ownership import DoctronicPrimaryCareEngine
    return DoctronicPrimaryCareEngine()


@pytest.fixture
def b2b_engine():
    """Return a fresh DoctronicB2BIntelligence engine."""
    from doctronic_b2b_outsourced_intelligence import DoctronicB2BIntelligence
    return DoctronicB2BIntelligence()

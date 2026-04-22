"""
Tests for title-prefix embeddings (Phase 3A-2).

When a title is provided to add_document(), it should be prepended to
sentence (L0) and paragraph (L1) text before embedding, enriching context.
Title should NOT affect L2 (document-level) embeddings.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from fractrag import FractalRAG, HashEmbedding, save, load


TEXT = (
    "Artificial intelligence is transforming healthcare delivery worldwide. "
    "Machine learning algorithms predict patient outcomes accurately. "
    "Natural language processing extracts insights from medical records. "
    "Computer vision systems detect anomalies in radiology scans."
)

TITLE = "AI Applications in Clinical Medicine"


class TestTitlePrefixEmbeddings:
    def test_title_changes_l0_embeddings(self):
        """Embeddings with title should differ from without title."""
        rag_no_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_no_title.add_document("d1", TEXT)

        rag_with_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_with_title.add_document("d1", TEXT, title=TITLE)

        # L0 embeddings should differ
        vec_no_title = rag_no_title.index[0][0].vec
        vec_with_title = rag_with_title.index[0][0].vec
        assert not np.allclose(vec_no_title, vec_with_title), (
            "Title prefix should change L0 embeddings"
        )

    def test_title_changes_l1_embeddings(self):
        """L1 (paragraph) embeddings should also change with title."""
        rag_no_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_no_title.add_document("d1", TEXT)

        rag_with_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_with_title.add_document("d1", TEXT, title=TITLE)

        vec_no_title = rag_no_title.index[1][0].vec
        vec_with_title = rag_with_title.index[1][0].vec
        assert not np.allclose(vec_no_title, vec_with_title), (
            "Title prefix should change L1 embeddings"
        )

    def test_title_does_not_change_l2_embeddings(self):
        """L2 (document) embeddings should NOT change with title."""
        rag_no_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_no_title.add_document("d1", TEXT)

        rag_with_title = FractalRAG(backend=HashEmbedding(dim=64))
        rag_with_title.add_document("d1", TEXT, title=TITLE)

        vec_no_title = rag_no_title.index[2][0].vec
        vec_with_title = rag_with_title.index[2][0].vec
        assert np.allclose(vec_no_title, vec_with_title), (
            "Title prefix should NOT affect L2 embeddings"
        )

    def test_no_title_unchanged(self):
        """When no title is provided, behavior should be identical to before."""
        rag1 = FractalRAG(backend=HashEmbedding(dim=64))
        rag1.add_document("d1", TEXT)

        rag2 = FractalRAG(backend=HashEmbedding(dim=64))
        rag2.add_document("d1", TEXT, title=None)

        for lvl in [0, 1, 2]:
            for e1, e2 in zip(rag1.index[lvl], rag2.index[lvl]):
                assert np.allclose(e1.vec, e2.vec), (
                    f"No-title and title=None should produce identical L{lvl} embeddings"
                )

    def test_title_stored_in_doc_titles(self):
        """Title should be stored in rag.doc_titles dict."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", TEXT, title=TITLE)
        rag.add_document("d2", TEXT)  # No title

        assert rag.doc_titles["d1"] == TITLE
        assert rag.doc_titles["d2"] is None


class TestTitlePrefixPersistence:
    def test_save_load_roundtrip_preserves_titles(self):
        """Titles should survive save/load cycle."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", TEXT, title=TITLE)
        rag.add_document("d2", TEXT)  # No title

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            save(db_path, rag)
            rag2 = load(db_path, backend=HashEmbedding(dim=64))

            assert rag2.doc_titles.get("d1") == TITLE
            # d2 had no title, so it won't be in doc_titles after load
            assert rag2.doc_titles.get("d2") is None
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_load_old_db_without_titles_table(self):
        """Loading a DB without doc_titles table should not crash (backward compat)."""
        import sqlite3

        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", TEXT)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            save(db_path, rag)
            # Drop the doc_titles table to simulate old schema
            conn = sqlite3.connect(db_path)
            conn.execute("DROP TABLE IF EXISTS doc_titles")
            conn.commit()
            conn.close()

            # Should load without error
            rag2 = load(db_path, backend=HashEmbedding(dim=64))
            assert len(rag2.doc_titles) == 0
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestTitlePrefixRetrieval:
    def test_retrieval_works_with_titles(self):
        """Retrieval should work normally when titles are used."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", TEXT, title=TITLE)

        results, qtype = rag.retrieve("AI in healthcare", k=5)
        assert len(results[2]) > 0
        assert len(results[1]) > 0
        assert len(results[0]) > 0

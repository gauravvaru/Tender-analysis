"""
Pipeline sanity tests.

Run from the project root:
    python -m pytest tests/test_pipeline.py -v

All FAISS tests are self-contained — they populate a temporary index
in a temp directory so they never depend on the app's live index state.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pytest

# ── Ensure project root is on sys.path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ingestion import DocumentProcessor
from modules.embedding import EmbeddingService, FAISSVectorStore
from modules.extraction import LLMExtractor
from modules.scoring import TenderScorer

# ── One real PDF that exists in input_docs/ ──────────────────────────────────
PDF = "input_docs/GeM-Bidding-8778719.pdf"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_index_dir():
    """Create a fresh temp directory for a clean FAISS index per test session."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def processed(tmp_index_dir):
    """
    Process the PDF once and embed all chunks into a temp index.
    Returns (result, store, embedder) so individual tests can reuse them.
    """
    dp       = DocumentProcessor()
    embedder = EmbeddingService()
    store    = FAISSVectorStore(dimension=384, index_dir=tmp_index_dir)

    result = dp.process_file(PDF)
    assert result["status"] == "success", f"PDF processing failed: {result.get('error')}"

    chunks = result["chunks"]
    # Stamp file_hash on every chunk (mirrors what ingest_file() does in app.py)
    import hashlib
    sha256 = hashlib.sha256()
    with open(PDF, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    file_hash = sha256.hexdigest()
    for c in chunks:
        c["file_hash"]   = file_hash
        c["source_file"] = os.path.basename(PDF)

    texts      = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_vectors(embeddings.astype("float32"), chunks)

    return {"result": result, "store": store, "embedder": embedder, "chunks": chunks}


# ── 1. Ingestion tests ────────────────────────────────────────────────────────

def test_ingestion_returns_chunks(processed):
    result = processed["result"]
    assert result["status"] == "success"
    assert len(result["chunks"]) > 0
    assert all("text" in c for c in result["chunks"])


def test_chunks_have_no_duplicates(processed):
    texts = [c["text"] for c in processed["chunks"]]
    assert len(texts) == len(set(texts)), "Duplicate chunk texts found"


def test_chunk_size_in_range(processed):
    for c in processed["chunks"]:
        assert c["token_count"] <= 550, (
            f"Chunk too large: {c['token_count']} tokens (max 550)"
        )


# ── 2. Embedding / FAISS tests ────────────────────────────────────────────────

def test_faiss_index_loads(processed):
    """Index should have vectors after ingestion."""
    stats = processed["store"].get_stats()
    assert stats["total_vectors"] > 0, "FAISS index is empty after ingestion"


def test_search_returns_results(processed):
    """Semantic search should return results from the populated index."""
    results = processed["store"].search("bid deadline submission date", k=5)
    assert len(results) > 0, "Search returned no results"
    assert all("similarity" in r for r in results)
    assert all(0 < r["similarity"] <= 1.0 for r in results), (
        "Similarity scores out of (0, 1] range"
    )


def test_no_duplicate_vectors_on_reingest(processed):
    """
    Re-ingesting the same PDF must add 0 new vectors.
    This test reuses the already-populated temp store.
    """
    store  = processed["store"]
    before = store.get_stats()["total_vectors"]

    # Re-process and try to add the same chunks again
    dp      = DocumentProcessor()
    result  = dp.process_file(PDF)
    chunks  = result["chunks"]

    import hashlib
    sha256 = hashlib.sha256()
    with open(PDF, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    file_hash = sha256.hexdigest()
    for c in chunks:
        c["file_hash"]   = file_hash
        c["source_file"] = os.path.basename(PDF)

    embedder   = processed["embedder"]
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_vectors(embeddings.astype("float32"), chunks)

    after = store.get_stats()["total_vectors"]
    assert before == after, (
        f"Duplicate vectors were added: {after - before} extra vectors"
    )


# ── 3. Extraction tests ───────────────────────────────────────────────────────

def test_extraction_returns_dict():
    ex   = LLMExtractor()
    meta = ex.extract_tender_metadata(
        "Tender ID: TEST-001. Bid deadline: 10-04-2026. Organisation: Indian Navy."
    )
    assert isinstance(meta, dict)
    assert len(meta) > 0, "Extraction returned an empty dict"


def test_extraction_has_key_fields():
    ex   = LLMExtractor()
    meta = ex.extract_tender_metadata(
        "Tender ID: TEST-001. Bid deadline: 10-04-2026. Organisation: Indian Navy."
    )
    populated = [
        meta.get("tender_id"),
        meta.get("organisation"),
        meta.get("bid_deadline"),
    ]
    assert any(populated), (
        f"All key fields are empty. Got: {meta}"
    )


# ── 4. Scoring tests ──────────────────────────────────────────────────────────

def test_score_in_range():
    scorer = TenderScorer()
    result = scorer.score_tender(
        tender_data={
            "tender_id": "T1",
            "title":     "Test Tender",
            "budget":    500000,
            "industry":  "IT",
            "region":    "Maharashtra",
        },
        company_profile={
            "skills":           ["Python", "ML"],
            "experience_years": 5,
            "team_size":        15,
            "certifications":   ["ISO 9001"],
            "industries":       ["IT"],
            "served_regions":   ["Maharashtra"],
            "max_project_value": 1000000,
        },
        extracted_requirements={"mandatory_skills": ["Python"]},
    )
    assert 0 <= result["overall_score"] <= 100, (
        f"Score out of [0, 100] range: {result['overall_score']}"
    )
    assert result["recommendation"] in ("pursue", "review", "pass"), (
        f"Unexpected recommendation: {result['recommendation']}"
    )


def test_weights_sum_to_one():
    from config import SCORING_WEIGHTS
    total = sum(SCORING_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, (
        f"Scoring weights sum to {total:.4f}, expected 1.0"
    )
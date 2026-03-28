"""
Tender Intelligence Platform - Streamlit UI
"""

import re
import hashlib
import logging
from pathlib import Path

import streamlit as st
import pandas as pd

from modules.ingestion import DocumentProcessor
from modules.embedding import EmbeddingService, FAISSVectorStore
from modules.extraction import LLMExtractor
from modules.scoring import TenderScorer, TenderRanker
from modules.input_handler import InputDocsHandler
from modules.utils import load_company_profile, save_company_profile
from config import UPLOADS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PURE HELPERS  (defined FIRST — used everywhere below)
# ─────────────────────────────────────────────

def _parse_number(value: str) -> float:
    """Extract a float from strings like '₹5,00,000' or '500000'."""
    if not value:
        return 0.0
    digits = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(digits)
    except ValueError:
        return 0.0


def _split(raw: str) -> list:
    """Split a comma-separated string into a cleaned list."""
    return [x.strip() for x in raw.split(",") if x.strip()]


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tender Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


# ─────────────────────────────────────────────
# INIT SERVICES  (cached — created once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def init_services():
    return {
        "processor":    DocumentProcessor(),
        "embedder":     EmbeddingService(),
        "vector_store": FAISSVectorStore(),
        "extractor":    LLMExtractor(),
        "scorer":       TenderScorer(),
    }

services = init_services()


# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "processed_docs" not in st.session_state:
    st.session_state["processed_docs"] = []

if "company_profile" not in st.session_state:
    st.session_state["company_profile"] = load_company_profile()


# ─────────────────────────────────────────────
# CORE INGESTION HELPER
# ─────────────────────────────────────────────
def ingest_file(full_path: str, file_name: str, extract_meta: bool) -> dict | None:
    """
    Process one file end-to-end:
      1. Skip if already indexed (file-hash check against FAISS metadata)
      2. Chunk + embed in batches of 50  ← safe for 500-600 page PDFs
      3. Optionally extract LLM metadata
    Returns a summary dict, or None if the file was already indexed.
    """
    store     = services["vector_store"]
    processor = services["processor"]
    embedder  = services["embedder"]
    extractor = services["extractor"]

    # Compute SHA-256 hash of the file
    sha256 = hashlib.sha256()
    with open(full_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    file_hash = sha256.hexdigest()

    # Skip if this exact file is already in the index
    already_indexed = any(
        m.get("file_hash") == file_hash
        for m in store.chunk_metadata
    )
    if already_indexed:
        return None

    # Process document (extract text, tables, chunks)
    result = processor.process_file(full_path)
    if result["status"] != "success":
        st.error(f"Processing failed for {file_name}: {result.get('error')}")
        return None

    chunks = result["chunks"]
    # Stamp every chunk with the file hash and filename for future dedup checks
    for c in chunks:
        c["file_hash"]   = file_hash
        c["source_file"] = file_name

    # Embed in batches of 50 — prevents memory spikes on large PDFs
    BATCH = 50
    for batch_start in range(0, len(chunks), BATCH):
        batch_chunks = chunks[batch_start: batch_start + BATCH]
        batch_texts  = [c["text"] for c in batch_chunks]
        embeddings   = embedder.embed_texts(batch_texts)
        store.add_vectors(embeddings, batch_chunks)

    # Optional LLM metadata extraction
    metadata: dict = {}
    if extract_meta:
        try:
            texts    = [c["text"] for c in chunks]
            metadata = extractor.extract_metadata(texts[:5])
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

    return {
        "name":     file_name,
        "path":     full_path,
        "chunks":   len(chunks),
        "pages":    result.get("page_count", "?"),
        "metadata": metadata,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Tender Intelligence")

    page = st.radio(
        "Navigate",
        [
            "📤 Upload Tenders",
            "📊 Analyze & Score",
            "🏆 Rank Tenders",
            "🔍 Search Tenders",
            "⚙️ Company Profile",
        ],
    )

    stats = services["vector_store"].get_stats()
    st.markdown("---")
    st.subheader("System Stats")
    col1, col2 = st.columns(2)
    with col1:
        unique_files = len({
            m.get("source_file", "")
            for m in services["vector_store"].chunk_metadata
        })
        st.metric("Files Indexed", unique_files)
    with col2:
        st.metric("Total Vectors", stats["total_vectors"])

    st.markdown("---")
    if st.button("🗑️ Reset Index", help="Clears FAISS index and all indexed data"):
        services["vector_store"].reset_index()
        st.session_state["processed_docs"] = []
        st.success("Index reset. Re-upload your documents.")
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: UPLOAD TENDERS
# ─────────────────────────────────────────────
if page == "📤 Upload Tenders":
    st.title("📤 Upload Tender Documents")

    extract_metadata = st.checkbox("Extract metadata using LLM", value=True)

    tab_folder, tab_upload = st.tabs(
        ["📁 From input_docs/ folder", "💻 Upload from your device"]
    )

    # ── Tab 1: input_docs folder ──────────────────────────────────────────
    with tab_folder:
        handler   = InputDocsHandler()
        documents = handler.get_available_documents()

        if not documents:
            st.info("No documents found in input_docs/ — add PDFs there and refresh.")
        else:
            display  = [f"{d['name']}  ({d['size_mb']} MB)" for d in documents]
            selected = st.multiselect("Select documents to ingest", display)

            if st.button("⚡ Process Selected", key="process_folder") and selected:
                progress = st.progress(0)
                status   = st.empty()

                for i, label in enumerate(selected):
                    doc     = documents[display.index(label)]
                    status.info(f"Processing {doc['name']} …")
                    summary = ingest_file(doc["full_path"], doc["name"], extract_metadata)

                    if summary is None:
                        status.warning(f"⏭ Already indexed: {doc['name']}")
                    else:
                        st.session_state["processed_docs"].append(summary)
                        status.success(
                            f"✅ {doc['name']} — "
                            f"{summary['chunks']} chunks, {summary['pages']} pages"
                        )
                    progress.progress((i + 1) / len(selected))

                st.success("Done! See results below.")

    # ── Tab 2: upload from device ─────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload one or more PDF files directly from your computer.")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="device_uploader",
        )

        if uploaded_files and st.button("⚡ Process Uploaded Files", key="process_upload"):
            progress = st.progress(0)
            status   = st.empty()

            for i, uploaded in enumerate(uploaded_files):
                save_path = Path(UPLOADS_DIR) / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                status.info(f"Processing {uploaded.name} …")
                summary = ingest_file(str(save_path), uploaded.name, extract_metadata)

                if summary is None:
                    status.warning(f"⏭ Already indexed: {uploaded.name}")
                else:
                    st.session_state["processed_docs"].append(summary)
                    status.success(
                        f"✅ {uploaded.name} — "
                        f"{summary['chunks']} chunks, {summary['pages']} pages"
                    )
                progress.progress((i + 1) / len(uploaded_files))

            st.success("All uploaded files processed!")

    # ── Session summary table ─────────────────────────────────────────────
    if st.session_state["processed_docs"]:
        st.markdown("---")
        st.subheader("Processed This Session")
        df = pd.DataFrame([
            {
                "File":         d["name"],
                "Pages":        d["pages"],
                "Chunks":       d["chunks"],
                "Tender ID":    d["metadata"].get("tender_id",       "—"),
                "Organisation": d["metadata"].get("organisation",    "—"),
                "Deadline":     d["metadata"].get("bid_deadline",    "—"),
                "Value":        d["metadata"].get("estimated_value", "—"),
            }
            for d in st.session_state["processed_docs"]
        ])
        st.dataframe(df, use_container_width=True)

        if st.button("Clear Session List"):
            st.session_state["processed_docs"] = []
            st.rerun()


# ─────────────────────────────────────────────
# PAGE: ANALYZE & SCORE
# ─────────────────────────────────────────────
if page == "📊 Analyze & Score":
    st.title("📊 Analyze & Score Tenders")

    profile = st.session_state["company_profile"]

    if not profile.get("skills") and not profile.get("experience_years"):
        st.warning(
            "⚠️ Your Company Profile is mostly empty. "
            "Scores will be inaccurate. "
            "Go to **⚙️ Company Profile** and fill it in first."
        )

    if not st.session_state["processed_docs"]:
        st.info("No processed tenders yet — go to **📤 Upload Tenders** first.")
    else:
        scorer = services["scorer"]

        for doc in st.session_state["processed_docs"]:
            meta = doc.get("metadata", {})

            tender_data = {
                "tender_id": meta.get("tender_id",     doc["name"]),
                "title":     meta.get("project_title", doc["name"]),
                "budget":    _parse_number(meta.get("estimated_value", "0")),
                "industry":  meta.get("department",    ""),
                "region":    meta.get("location",      ""),
            }
            tech_reqs = {
                "mandatory_skills": meta.get("mandatory_skills", []),
            }

            score     = scorer.score_tender(tender_data, profile, tech_reqs)
            rec_color = {"pursue": "🟢", "review": "🟡", "pass": "🔴"}.get(
                score["recommendation"], "⚪"
            )

            with st.expander(
                f"{rec_color} {doc['name']}  |  "
                f"Score: {score['overall_score']:.1f}  |  "
                f"{score['recommendation'].upper()}"
            ):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Technical Match", f"{score['technical_match_score']:.0f}%")
                col2.metric("Capability",       f"{score['capability_alignment_score']:.0f}%")
                col3.metric("Compliance",        f"{score['compliance_score']:.0f}%")
                col4.metric("Risk",              f"{score['risk_score']:.0f}%")

                st.markdown("**Score Breakdown**")
                for dim, info in score["breakdown"].items():
                    st.markdown(
                        f"- **{dim.replace('_', ' ').title()}**: {info['explanation']}"
                    )

                st.markdown("**Extracted Metadata**")
                st.json(doc.get("metadata", {}))


# ─────────────────────────────────────────────
# PAGE: RANK TENDERS
# ─────────────────────────────────────────────
if page == "🏆 Rank Tenders":
    st.title("🏆 Rank Tenders")

    profile = st.session_state["company_profile"]

    if not st.session_state["processed_docs"]:
        st.info("Upload tenders first.")
    else:
        scorer: TenderScorer       = services["scorer"]
        scored_list: list[dict]    = []

        for doc in st.session_state["processed_docs"]:
            meta = doc.get("metadata", {})
            tender_data = {
                "tender_id": meta.get("tender_id",     doc["name"]),
                "title":     meta.get("project_title", doc["name"]),
                "budget":    _parse_number(meta.get("estimated_value", "0")),
                "industry":  meta.get("department",    ""),
                "region":    meta.get("location",      ""),
            }
            score = scorer.score_tender(tender_data, profile, {})
            scored_list.append({**score, "file": doc["name"], "metadata": meta})

        # Sidebar filter controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("Rank Filters")
        min_score = st.sidebar.slider("Min overall score", 0, 100, 0)
        max_risk  = st.sidebar.slider("Max risk score",    0, 100, 100)

        filtered = TenderRanker.rank_tenders(
            scored_list, min_score=min_score, max_risk=max_risk
        )
        rec_icon = {"pursue": "🟢", "review": "🟡", "pass": "🔴"}

        for t in filtered:
            icon = rec_icon.get(t["recommendation"], "⚪")
            with st.expander(
                f"#{t['rank']}  {icon}  {t['file']}  — {t['overall_score']:.1f} pts"
            ):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Overall",    f"{t['overall_score']:.1f}")
                c2.metric("Tech Match", f"{t['technical_match_score']:.0f}%")
                c3.metric("Capability", f"{t['capability_alignment_score']:.0f}%")
                c4.metric("Compliance", f"{t['compliance_score']:.0f}%")
                c5.metric("Risk",       f"{t['risk_score']:.0f}%")
                st.caption(f"Recommendation: **{t['recommendation'].upper()}**")

        if filtered:
            df_rank = pd.DataFrame([{
                "Rank":       t["rank"],
                "File":       t["file"],
                "Score":      t["overall_score"],
                "Tech Match": t["technical_match_score"],
                "Risk":       t["risk_score"],
                "Verdict":    t["recommendation"],
            } for t in filtered])
            st.markdown("---")
            st.subheader("Summary Table")
            st.dataframe(df_rank, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: SEARCH
# ─────────────────────────────────────────────
if page == "🔍 Search Tenders":
    st.title("🔍 Search Tender Chunks")

    query = st.text_input(
        "Enter search query",
        placeholder="e.g. bid submission deadline EMD amount"
    )

    col_k, _ = st.columns([1, 3])
    with col_k:
        k = st.slider("Results to return", 1, 20, 5)

    if query:
        results = services["vector_store"].search(query, k=k)

        if not results:
            st.warning("No results found. Make sure documents are indexed.")
        else:
            for i, r in enumerate(results, start=1):
                meta      = r["metadata"]
                text      = meta.get("text", "")
                source    = meta.get("source_file", "unknown file")
                page_num  = meta.get("page", "?")
                section   = meta.get("section_type", "general")
                sim_score = r["similarity"]

                with st.expander(
                    f"Result {i}  |  {source}  |  "
                    f"page {page_num}  |  {section}  |  "
                    f"similarity {sim_score:.3f}"
                ):
                    st.write(text)
                    st.caption(
                        f"Chunk id: {meta.get('id', '?')}  |  "
                        f"tokens: {meta.get('token_count', '?')}"
                    )


# ─────────────────────────────────────────────
# PAGE: COMPANY PROFILE
# ─────────────────────────────────────────────
if page == "⚙️ Company Profile":
    st.title("⚙️ Company Profile")

    st.info(
        "This profile is used to **score and rank tenders**. "
        "The system compares your skills, certifications, and experience "
        "against each tender's requirements to calculate a match score. "
        "Fill this in accurately to get meaningful scores."
    )

    profile = st.session_state["company_profile"]

    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        profile["company_name"]      = st.text_input(
            "Company Name", profile.get("company_name", "")
        )
        profile["experience_years"]  = st.number_input(
            "Years of Experience",
            min_value=0,
            value=int(profile.get("experience_years", 0))
        )
    with col2:
        profile["team_size"]         = st.number_input(
            "Team Size",
            min_value=0,
            value=int(profile.get("team_size", 0))
        )
        profile["max_project_value"] = st.number_input(
            "Max Project Value (₹)",
            min_value=0,
            value=int(profile.get("max_project_value", 0)),
            step=100000,
            help="Largest contract your company has handled. Used for budget risk scoring."
        )

    st.subheader("Skills & Capabilities")
    st.caption("Enter comma-separated values for each field below.")

    col_a, col_b = st.columns(2)
    with col_a:
        skills_raw = st.text_area(
            "Technical Skills",
            value=", ".join(profile.get("skills", [])),
            height=100,
            placeholder="Python, Machine Learning, Civil Engineering, Supply Chain …",
            help="Matched against tender mandatory_skills → Technical Match score."
        )
        certifications_raw = st.text_area(
            "Certifications",
            value=", ".join(profile.get("certifications", [])),
            height=100,
            placeholder="ISO 9001, ISO 14001, ISO 45001 …",
            help="ISO certifications directly boost your Compliance score."
        )
    with col_b:
        industries_raw = st.text_area(
            "Industries Served",
            value=", ".join(profile.get("industries", [])),
            height=100,
            placeholder="Defence, Healthcare, Construction, IT …",
            help="Matching industry reduces Risk score."
        )
        regions_raw = st.text_area(
            "Served Regions / States",
            value=", ".join(profile.get("served_regions", [])),
            height=100,
            placeholder="Maharashtra, Delhi, Karnataka …",
            help="Matching region reduces Risk score."
        )
        industry_certs_raw = st.text_area(
            "Industry-Specific Certifications",
            value=", ".join(profile.get("industry_certifications", [])),
            height=80,
            placeholder="Power sector license, Construction permit …"
        )

    profile["skills"]                  = _split(skills_raw)
    profile["certifications"]          = _split(certifications_raw)
    profile["industries"]              = _split(industries_raw)
    profile["served_regions"]          = _split(regions_raw)
    profile["industry_certifications"] = _split(industry_certs_raw)

    st.markdown("---")
    col_save, col_preview = st.columns([1, 3])

    with col_save:
        if st.button("💾 Save Profile", type="primary"):
            st.session_state["company_profile"] = profile
            save_company_profile(profile)
            st.success("Profile saved!")

    with col_preview:
        with st.expander("Preview JSON"):
            st.json(profile)

    st.markdown("---")
    st.subheader("How your profile affects scores")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**🔧 Technical Match (35%)**")
        st.caption(
            "Overlap between your Skills and the tender's mandatory skills. "
            "Empty skills list → defaults to 50%."
        )
    with c2:
        st.markdown("**💪 Capability (25%)**")
        st.caption(
            "Years of Experience + Team Size + Certifications count. "
            "≥5 yrs + ≥20 team + ≥4 certs = full score."
        )
    with c3:
        st.markdown("**📋 Compliance (15%)**")
        st.caption(
            "ISO 9001, 14001, 45001 each add 25 pts. "
            "Industry certs add 25 pts when they match the tender sector."
        )
    with c4:
        st.markdown("**⚠️ Risk (25%)**")
        st.caption(
            "Risk REDUCES your score. Higher when tender budget exceeds "
            "your max project value or location/industry is unfamiliar."
        )
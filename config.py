"""
Configuration for Tender Intelligence Platform
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
INPUT_DOCS_DIR = BASE_DIR / "input_docs"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
INPUT_DOCS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# FIX 1: Corrected chunk sizes to match system spec (was 512/50)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Document Processing
OCR_ENABLED = True
EXTRACT_TABLES = True
MIN_TEXT_LENGTH = 10
MAX_FILE_SIZE_MB = 100

# Metadata Extraction Fields
METADATA_FIELDS = [
    "tender_id",
    "organisation",
    "department",
    "project_title",
    "project_description",
    "estimated_value",
    "emd_amount",
    "bid_deadline",
    "experience_required",
    "turnover_requirement",
    "location",
    "eligibility_criteria"
]

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# FIX 2: Removed typo fallback "gemini-1.5-flash-flash" — now reads cleanly from .env
# Your .env should have: LLM_MODEL=models/gemini-2.5-flash
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.5-flash")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Retrieval Configuration
TOP_K_RESULTS = 5
HYBRID_SEARCH = False       # BM25 + vector (future upgrade — keep False for now)
RERANK_RESULTS = False      # Reranking model (future upgrade — keep False for now)
MAX_CONTEXT_CHUNKS = 8

# CSV / JSON Paths
TENDERS_CSV = DATA_DIR / "tenders_metadata.csv"
COMPANY_PROFILE_JSON = DATA_DIR / "company_profile.json"
CHUNK_METADATA_CSV = DATA_DIR / "chunk_metadata.csv"

# Scoring Weights (must sum to 1.0)
SCORING_WEIGHTS = {
    "technical_match": 0.35,
    "risk_score": 0.25,
    "capability_alignment": 0.25,
    "compliance_score": 0.15
}

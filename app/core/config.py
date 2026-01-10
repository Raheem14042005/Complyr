from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import boto3
from fastapi.responses import JSONResponse


BASE_DIR = Path(__file__).resolve().parents[2]  # points to project root (where main.py is)

# ============================================================
# CONFIG / ENV
# ============================================================
load_dotenv()
R2_ENABLED = os.getenv("R2_ENABLED", "false").lower() in ("1", "true", "yes", "on")
R2_BUCKET = (os.getenv("R2_BUCKET") or "").strip()

R2_ENDPOINT = (os.getenv("R2_ENDPOINT") or "").strip()
R2_ACCESS_KEY_ID = (os.getenv("R2_ACCESS_KEY_ID") or "").strip()
R2_SECRET_ACCESS_KEY = (os.getenv("R2_SECRET_ACCESS_KEY") or "").strip()
R2_PREFIX = (os.getenv("R2_PREFIX") or "pdfs/").strip()
if R2_PREFIX and not R2_PREFIX.endswith("/"):
    R2_PREFIX += "/"

def r2_client():
    if not (R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        raise RuntimeError(
            "Missing R2 creds/env "
            "(R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)"
        )
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


# Optional: Vertex Embeddings (no extra deps)
try:
    from vertexai.language_models import TextEmbeddingModel  # type: ignore

    _VERTEX_EMBEDDINGS_AVAILABLE = True
except Exception:
    TextEmbeddingModel = None  # type: ignore
    _VERTEX_EMBEDDINGS_AVAILABLE = False

# Optional: Document AI ingest helper
try:
    from docai_ingest import docai_extract_pdf_to_text

    _DOCAI_HELPER_AVAILABLE = True
except Exception:
    docai_extract_pdf_to_text = None
    _DOCAI_HELPER_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))

CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", "40"))
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "32000"))

GCP_PROJECT_ID = (os.getenv("GCP_PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID") or "").strip()
GCP_LOCATION = (os.getenv("GCP_LOCATION") or os.getenv("VERTEX_LOCATION") or "europe-west4").strip()
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

MODEL_CHAT = (os.getenv("GEMINI_MODEL_CHAT", "gemini-2.0-flash-001") or "").strip()
MODEL_COMPLIANCE = (os.getenv("GEMINI_MODEL_COMPLIANCE", "gemini-2.0-flash-001") or "").strip()

# Web search (Serper)
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()  # https://serper.dev
WEB_ENABLED = (os.getenv("WEB_ENABLED", "true").lower() in ("1", "true", "yes", "on")) and bool(SERPER_API_KEY)

# Evidence defaults
DEFAULT_EVIDENCE_MODE = os.getenv("DEFAULT_EVIDENCE_MODE", "false").lower() in ("1", "true", "yes", "on")

# Retrieval sizing
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))
TOP_K_WEB = int(os.getenv("TOP_K_WEB", "5"))

# Vector embeddings (optional)
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "true").lower() in ("1", "true", "yes", "on")
EMBED_MODEL_NAME = (os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004") or "").strip()
EMBED_TOPK = int(os.getenv("EMBED_TOPK", "24"))  # candidate pool for rerank
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rerank (LLM reranker)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
RERANK_TOPK = int(os.getenv("RERANK_TOPK", str(TOP_K_CHUNKS)))

# Broad vs precise answering controls
BROAD_DOC_DIVERSITY_K = int(os.getenv("BROAD_DOC_DIVERSITY_K", "5"))
BROAD_TOPIC_HITS_K = int(os.getenv("BROAD_TOPIC_HITS_K", "3"))
BROAD_MAX_SUBQUERIES = int(os.getenv("BROAD_MAX_SUBQUERIES", "8"))

# Web allowlist (safe + higher trust)
WEB_ALLOWLIST_DEFAULT = [
    "irishstatutebook.ie",
    "gov.ie",
    "housing.gov.ie",
    "nsai.ie",
    "dublincity.ie",
    "dlrcoco.ie",
    "corkcity.ie",
    "kildarecoco.ie",
    "galwaycity.ie",
]
WEB_ALLOWLIST = [d.strip().lower() for d in (os.getenv("WEB_ALLOWLIST", "") or "").split(",") if d.strip()]
if not WEB_ALLOWLIST:
    WEB_ALLOWLIST = WEB_ALLOWLIST_DEFAULT

# Hard numeric verification
VERIFY_NUMERIC = os.getenv("VERIFY_NUMERIC", "true").lower() in ("1", "true", "yes", "on")

# Rules layer
RULES_FILE = Path(os.getenv("RULES_FILE", str(BASE_DIR / "rules.json")))
RULES_ENABLED = os.getenv("RULES_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Eval harness
EVAL_FILE = Path(os.getenv("EVAL_FILE", str(BASE_DIR / "eval_tests.json")))

# Timeouts
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "18"))
# ----------------------------
# Web fetch behaviour (ChatGPT-like)
# ----------------------------

MAX_WEB_BYTES = int(os.getenv("MAX_WEB_BYTES", str(2_500_000)))  # 2.5 MB safety cap
WEB_RETRIES = int(os.getenv("WEB_RETRIES", "2"))                 # retry flaky sites
WEB_RETRY_BACKOFF = float(os.getenv("WEB_RETRY_BACKOFF", "0.6")) # seconds

# Web cache
WEB_CACHE_TTL_SECONDS = int(os.getenv("WEB_CACHE_TTL_SECONDS", str(12 * 60 * 60)))  # 12h
# -----------------------------------------
# DIAGRAMS / IMAGES FROM PDFs (optional)
# -----------------------------------------
PDF_IMAGE_EXTRACT = os.getenv("PDF_IMAGE_EXTRACT", "false").lower() in ("1", "true", "yes", "on")
PDF_IMAGE_MAX_PAGES = int(os.getenv("PDF_IMAGE_MAX_PAGES", "12"))     # safety cap
PDF_IMAGE_MAX_IMAGES = int(os.getenv("PDF_IMAGE_MAX_IMAGES", "24"))   # safety cap
PDF_IMAGE_MIN_PIXELS = int(os.getenv("PDF_IMAGE_MIN_PIXELS", "120000"))  # ignore tiny icons


if os.getenv("RENDER"):
    DATA_DIR = Path("/tmp/raheemai")
else:
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR))).resolve()


PDF_DIR = DATA_DIR / "pdfs"
DOCAI_DIR = DATA_DIR / "parsed_docai"
CACHE_DIR = DATA_DIR / "cache"
WEB_CACHE_DIR = CACHE_DIR / "web"

for d in (PDF_DIR, DOCAI_DIR, CACHE_DIR, WEB_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
    
ADMIN_API_KEY = (os.getenv("ADMIN_API_KEY") or "").strip()
def require_admin_key(x_api_key: Optional[str]) -> Optional[JSONResponse]:
    if not ADMIN_API_KEY:
        return None  # dev mode
    if (x_api_key or "").strip() != ADMIN_API_KEY:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)
    return None


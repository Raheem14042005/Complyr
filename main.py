"""
Raheem AI — FastAPI backend
Goal: normal AI vibe + becomes a PDF/TGD expert when needed, WITHOUT scanning everything.

Key upgrades:
- Stronger retrieval (BM25-like scoring) so deep pages (e.g. p.350) get found.
- Iterative retrieval widening: starts small (cheap) and expands only if needed.
- Vision only on selected pages when tables/diagrams likely matter.
- Optional page_hint and pdf pinning.
"""

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from openai import OpenAI

import os
import re
import math
import base64
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter, OrderedDict

import fitz  # PyMuPDF


# ----------------------------
# Setup
# ----------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))
IMAGE_CACHE_MAX = int(os.getenv("IMAGE_CACHE_MAX", "64"))

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# Storage paths
# ----------------------------

def choose_pdf_dir() -> Path:
    env_dir = os.getenv("PDF_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    preferred = Path("/var/data/pdfs")
    fallback = BASE_DIR / "pdfs"

    if os.getenv("RENDER"):
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except Exception:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


PDF_DIR = choose_pdf_dir()


# ----------------------------
# Tokenization / stopwords
# ----------------------------

STOPWORDS = {
    "the","and","or","of","to","in","a","an","for","on","with","is","are","be","as","at","from","by",
    "that","this","it","your","you","we","they","their","there","what","which","when","where","how",
    "can","shall","should","must","may","not","than","then","into","onto","also","such"
}

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    # Keep hyphenated and numeric-ish tokens (e.g. "u-value", "m2", "25%")
    toks = re.findall(r"[a-z0-9][a-z0-9\-/\.%]*", t)
    toks = [x for x in toks if len(x) >= 2 and x not in STOPWORDS]
    return toks


# ----------------------------
# Indexes
# ----------------------------
# For each PDF:
# - page_text_lower: list[str]
# - page_tf: list[Counter]
# - df: Counter of term -> number of pages containing term
# - avgdl: average document length (tokens per page)
# - page_len: list[int] length per page
PDF_INDEX: Dict[str, Dict[str, Any]] = {}

def list_pdfs() -> List[str]:
    return [p.name for p in PDF_DIR.glob("*.pdf")]

def ensure_indexed(pdf_name: str) -> None:
    if pdf_name in PDF_INDEX:
        return
    p = PDF_DIR / pdf_name
    if p.exists():
        index_pdf(p)

def index_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    doc = fitz.open(pdf_path)
    try:
        page_text_lower: List[str] = []
        page_tf: List[Counter] = []
        df = Counter()
        page_len: List[int] = []

        for i in range(doc.page_count):
            page = doc.load_page(i)
            txt = clean_text(page.get_text("text") or "")
            low = txt.lower()
            page_text_lower.append(low)

            toks = tokenize(low)
            tf = Counter(toks)
            page_tf.append(tf)

            unique = set(tf.keys())
            df.update(unique)

            page_len.append(len(toks))

        avgdl = (sum(page_len) / len(page_len)) if page_len else 0.0

        PDF_INDEX[name] = {
            "page_text_lower": page_text_lower,
            "page_tf": page_tf,
            "df": df,
            "page_len": page_len,
            "avgdl": avgdl,
            "pages": doc.page_count,
        }
    finally:
        doc.close()

def index_all_pdfs() -> None:
    for p in PDF_DIR.glob("*.pdf"):
        try:
            index_pdf(p)
        except Exception:
            continue

index_all_pdfs()


# ----------------------------
# BM25-like scoring over pages
# ----------------------------

def bm25_page_score(
    tf: Counter,
    df: Counter,
    N: int,
    dl: int,
    avgdl: float,
    q_tokens: List[str],
    k1: float = 1.4,
    b: float = 0.75
) -> float:
    """
    Lightweight BM25-ish score for a single page.
    """
    if not q_tokens or N <= 0:
        return 0.0

    score = 0.0
    denom_norm = (1 - b) + b * (dl / (avgdl + 1e-9))

    for t in q_tokens:
        f = tf.get(t, 0)
        if f <= 0:
            continue

        n_t = df.get(t, 0)
        # IDF with smoothing
        idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))

        score += idf * (f * (k1 + 1)) / (f + k1 * denom_norm)

    return score

def phrase_bonus(page_text_lower: str, question: str) -> float:
    """
    Adds a small bonus for exact short phrase matches (2-4 grams).
    Helps pull the right page even if BM25 is close.
    """
    q_words = [w for w in re.findall(r"[a-z0-9]+", (question or "").lower()) if w not in STOPWORDS]
    bonus = 0.0
    for n in (2, 3, 4):
        for i in range(0, max(0, len(q_words) - n + 1)):
            phrase = " ".join(q_words[i:i+n])
            if len(phrase) < 9:
                continue
            if phrase in page_text_lower:
                bonus += 1.0 + 0.25 * n
    return bonus

def page_hint_bonus(page_index: int, page_hint: Optional[int]) -> float:
    """
    If the user hints a page (1-based), gently bias nearby pages.
    """
    if not page_hint or page_hint <= 0:
        return 0.0
    target = page_hint - 1
    dist = abs(page_index - target)
    if dist == 0:
        return 2.5
    if dist <= 2:
        return 1.5
    if dist <= 6:
        return 0.6
    return 0.0


# ----------------------------
# Retrieval: start small, widen if needed
# ----------------------------

def retrieve_top_pages_for_doc(
    pdf_name: str,
    question: str,
    top_k: int = 4,
    page_hint: Optional[int] = None
) -> List[Tuple[int, float]]:
    ensure_indexed(pdf_name)
    idx = PDF_INDEX.get(pdf_name)
    if not idx:
        return []

    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    N = idx["pages"]
    df = idx["df"]
    avgdl = idx["avgdl"]
    scores: List[Tuple[int, float]] = []

    for i in range(N):
        tf = idx["page_tf"][i]
        dl = idx["page_len"][i]
        base = bm25_page_score(tf, df, N, dl, avgdl, q_tokens)
        if base <= 0:
            continue
        base += phrase_bonus(idx["page_text_lower"][i], question)
        base += page_hint_bonus(i, page_hint)
        scores.append((i, float(base)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def retrieve_across_pdfs_iterative(
    question: str,
    pdf_pin: Optional[str] = None,
    page_hint: Optional[int] = None,
    max_docs: int = 3,
) -> List[Tuple[str, List[int], float]]:
    """
    Returns list of (pdf_name, page_indexes_0_based, score)

    Strategy:
    - If pdf_pin provided: only search that doc.
    - Else: score docs by best pages, take top docs.
    - Start with a small selection, widen only if weak.
    """

    available = list_pdfs()
    if not available:
        return []

    if pdf_pin:
        if pdf_pin not in available:
            return []
        top_pages = retrieve_top_pages_for_doc(pdf_pin, question, top_k=6, page_hint=page_hint)
        if not top_pages:
            return []
        best = top_pages[0][1]
        pages = [p for p, _ in top_pages[:3]]
        return [(pdf_pin, sorted(set(pages)), best)]

    # Score each doc by its best page score
    doc_best: List[Tuple[str, float]] = []
    per_doc_pages: Dict[str, List[Tuple[int, float]]] = {}

    for name in available:
        top_pages = retrieve_top_pages_for_doc(name, question, top_k=6, page_hint=page_hint)
        per_doc_pages[name] = top_pages
        best = top_pages[0][1] if top_pages else 0.0
        if best > 0:
            doc_best.append((name, best))

    doc_best.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, s in doc_best[:max_docs] if s > 0]
    if not top_docs:
        return []

    results: List[Tuple[str, List[int], float]] = []
    for doc_name in top_docs:
        pages_scored = per_doc_pages.get(doc_name, [])
        best = pages_scored[0][1] if pages_scored else 0.0
        pages = [p for p, _ in pages_scored[:3]]
        results.append((doc_name, sorted(set(pages)), best))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def expand_pages(pages: List[int], total_pages: int, window: int) -> List[int]:
    s = set()
    for p in pages:
        for n in range(-window, window + 1):
            idx = p + n
            if 0 <= idx < total_pages:
                s.add(idx)
    return sorted(s)

def iterative_select_pages(
    question: str,
    pdf_pin: Optional[str] = None,
    page_hint: Optional[int] = None,
    max_docs: int = 3,
    max_total_pages: int = 8,
) -> List[Tuple[str, List[int], float]]:
    """
    Start small (cheap), expand only if retrieval confidence seems low.
    """
    hits = retrieve_across_pdfs_iterative(question, pdf_pin=pdf_pin, page_hint=page_hint, max_docs=max_docs)
    if not hits:
        return []

    final: List[Tuple[str, List[int], float]] = []

    # Heuristic confidence: if top score is tiny, widen.
    top_score = hits[0][2]

    # Base window / pages per doc
    window = 0
    pages_per_doc = 3

    if top_score < 2.0:
        window = 1
        pages_per_doc = 4
    if top_score < 1.2:
        window = 2
        pages_per_doc = 6

    used_pages = 0
    for doc_name, base_pages, score in hits:
        ensure_indexed(doc_name)
        total = PDF_INDEX[doc_name]["pages"]

        # Take more pages if score is low, but cap total pages sent.
        pages = base_pages[:pages_per_doc]
        pages = expand_pages(pages, total, window=window)

        # Cap total pages across docs
        remaining = max_total_pages - used_pages
        if remaining <= 0:
            break

        pages = pages[:remaining]
        used_pages += len(pages)

        final.append((doc_name, pages, score))

    return final


# ----------------------------
# Vision helpers (tables/diagrams)
# ----------------------------

_IMAGE_CACHE: "OrderedDict[Tuple[str,int,int], str]" = OrderedDict()

def cache_get(k):
    if k in _IMAGE_CACHE:
        _IMAGE_CACHE.move_to_end(k)
        return _IMAGE_CACHE[k]
    return None

def cache_set(k, v):
    _IMAGE_CACHE[k] = v
    _IMAGE_CACHE.move_to_end(k)
    while len(_IMAGE_CACHE) > IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)

def pdf_page_to_data_url(pdf_path: Path, page_index: int, dpi: int = 120) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def pages_to_images(pdf_path: Path, pdf_name: str, pages: List[int], dpi: int = 120) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in pages:
        key = (pdf_name, p, dpi)
        data_url = cache_get(key)
        if data_url is None:
            data_url = pdf_page_to_data_url(pdf_path, p, dpi=dpi)
            cache_set(key, data_url)
        blocks.append({"type": "input_image", "image_url": data_url})
    return blocks

def needs_vision(question: str, excerpt: str) -> bool:
    q = (question or "").lower()
    if any(t in q for t in ["table","diagram","figure","fig.","chart","schedule","drawing","plan","elevation","section"]):
        return True
    # If extracted text looks like poor table extraction
    lines = (excerpt or "").splitlines()
    if not lines:
        return True
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    spaced = sum(1 for ln in lines if "    " in ln)
    if pipe_lines > 4 or spaced > 12:
        return True
    if len(excerpt) < 800:
        return True
    return False


# ----------------------------
# Router (normal AI vs doc-grounded)
# ----------------------------

def is_short_topic_prompt(q: str) -> bool:
    words = re.findall(r"[a-zA-Z0-9']+", (q or "").strip())
    return 1 <= len(words) <= 4 and "?" not in (q or "")

def looks_like_definition_question(q: str) -> bool:
    ql = (q or "").lower().strip()
    return any(ql.startswith(x) for x in ["what is ", "define ", "meaning of ", "what does ", "explain "])

def doc_intent_score(question: str) -> int:
    q = (question or "").lower()
    score = 0

    hard = [
        "tgd", "technical guidance", "building regulations", "part a", "part b", "part m", "part l",
        "deap", "seai", "irish building regs", "bcar",
        "according to", "in the document", "in the pdf", "pdf", "document",
        "what does it say", "where does it say", "cite", "citation", "page", "clause", "section", "appendix",
        "table", "diagram", "figure", "schedule"
    ]
    for t in hard:
        if t in q:
            score += 4

    soft = [
        "minimum","maximum","shall","must","required","requirement","compliance","comply","regulation",
        "guidance","standard","fire safety","accessibility","u-value","y-value","escape","travel distance"
    ]
    for t in soft:
        if t in q:
            score += 1

    return score

def evidence_exists_in_pdfs(question: str, pdf_pin: Optional[str] = None, page_hint: Optional[int] = None) -> bool:
    hits = retrieve_across_pdfs_iterative(question, pdf_pin=pdf_pin, page_hint=page_hint, max_docs=1)
    return bool(hits)

def should_use_docs(question: str, pdf_pin: Optional[str] = None, page_hint: Optional[int] = None) -> bool:
    score = doc_intent_score(question)
    if score >= 4:
        return True
    if looks_like_definition_question(question):
        # Only go doc-mode if we can find evidence quickly
        return evidence_exists_in_pdfs(question, pdf_pin=pdf_pin, page_hint=page_hint)
    return False


SYSTEM_RULES = """
You are Raheem AI — a calm, capable assistant.

Tone:
- Slightly positive and professionally humorous (subtle).
- If the user’s prompt is short/unclear, ask 1–2 clarifying questions before dumping a long answer.

Core behavior:
- Act like a normal AI by default: helpful, practical, natural.
- When the user asks what TGDs/Parts/PDFs say, or asks about compliance requirements, become document-grounded.

When SOURCES are provided:
- Treat SOURCES as authoritative.
- Cite pages like (DocName p.12).
- Do NOT invent clause numbers, numeric thresholds, or interpret a diagram you haven’t actually seen.
- If SOURCES don’t contain enough to answer, say so and suggest what to search next (and what terms/pages to look for).

Output style:
- Give the answer first in plain English.
- Then add a short “Where this comes from” section with citations if you used SOURCES.
""".strip()


# ----------------------------
# Source extraction / bundling
# ----------------------------

def extract_pages_text(pdf_path: Path, page_indexes: List[int], max_chars_per_page: int = 1800) -> str:
    doc = fitz.open(pdf_path)
    try:
        chunks = []
        for idx in page_indexes:
            page = doc.load_page(idx)
            txt = clean_text(page.get_text("text") or "")
            if len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + " …"
            chunks.append(f"[Page {idx+1}]\n{txt}")
        return "\n\n".join(chunks)
    finally:
        doc.close()

def build_sources_bundle(
    selected: List[Tuple[str, List[int], float]],
    max_chars_per_page: int = 1800
) -> Tuple[str, List[Tuple[str,int]]]:
    parts: List[str] = []
    cites: List[Tuple[str,int]] = []
    for doc_name, page_idxs, _score in selected:
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            continue
        excerpt = extract_pages_text(pdf_path, page_idxs, max_chars_per_page=max_chars_per_page)
        parts.append(f"SOURCE: {doc_name}\n{excerpt}")
        for pi in page_idxs:
            cites.append((doc_name, pi + 1))
    return "\n\n".join(parts).strip(), cites

def build_openai_blocks(
    question: str,
    sources_text: str,
    images: Optional[List[Dict[str, Any]]] = None,
    history_blob: Optional[str] = None
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [{"type": "input_text", "text": SYSTEM_RULES}]

    if history_blob:
        prompt = f"CHAT HISTORY (most recent):\n{history_blob}\n\nLATEST USER QUESTION:\n{question}\n"
    else:
        prompt = f"USER QUESTION:\n{question}\n"

    if sources_text:
        prompt += f"\nSOURCES:\n{sources_text}"
    else:
        prompt += "\n(No PDF sources provided.)"

    blocks.append({"type": "input_text", "text": prompt})

    if images:
        blocks.extend(images)

    return blocks


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Raheem AI API",
        "pdf_dir": str(PDF_DIR),
        "pdf_count": len(list_pdfs()),
        "indexed_pdfs": len(PDF_INDEX),
        "model": MODEL_NAME,
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/pdfs")
def pdfs():
    files = list_pdfs()
    return {"count": len(files), "pdfs": files}

def safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._\\- ]+", "", name).strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]

from fastapi import HTTPException
import shutil

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Robust uploader:
    - Streams to disk (does NOT load whole file into RAM)
    - Validates extension + size
    - Returns JSON errors instead of plain 500
    - Indexing failure won't delete the uploaded file (but will report it)
    """
    try:
        if not file or not file.filename:
            return {"ok": False, "error": "No file received (filename empty)."}

        if not file.filename.lower().endswith(".pdf"):
            return {"ok": False, "error": "Only PDF files allowed"}

        fname = safe_filename(file.filename)
        path = PDF_DIR / fname

        # Ensure directory exists and is writable
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        try:
            testfile = PDF_DIR / ".write_test"
            testfile.write_text("ok", encoding="utf-8")
            testfile.unlink(missing_ok=True)
        except Exception as e:
            return {
                "ok": False,
                "error": f"PDF_DIR is not writable: {PDF_DIR}",
                "detail": str(e),
            }

        # Stream-save to disk with a size cap
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024 if MAX_UPLOAD_MB > 0 else None
        written = 0
        with open(path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    out.close()
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return {"ok": False, "error": f"PDF too large. Max is {MAX_UPLOAD_MB}MB"}
                out.write(chunk)

        # Try indexing (but don't 500 if indexing fails)
        indexed_ok = True
        index_error = None
        try:
            index_pdf(path)
        except Exception as ie:
            indexed_ok = False
            index_error = str(ie)

        return {
            "ok": True,
            "pdf": fname,
            "bytes": written,
            "stored_in": str(PDF_DIR),
            "indexed": indexed_ok,
            "index_error": index_error,
        }

    except Exception as e:
        # Return a JSON error instead of plain 500
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc().splitlines()[-25:],
        }



# ----------------------------
# /ask (non-streaming)
# ----------------------------

@app.get("/ask")
def ask(
    q: str = Query(..., description="User question"),
    pdf: Optional[str] = Query(None, description="Pin a specific PDF name (optional)"),
    page_hint: Optional[int] = Query(None, description="1-based page number to bias retrieval near (optional)"),
    force_docs: bool = Query(False, description="Force using PDFs even if router says no"),
    max_docs: int = Query(3),
    max_total_pages: int = Query(8),
    dpi: int = Query(120),
    force_vision: bool = Query(False),
):
    try:
        # Short topic => ask 1 clarifying question unless user is clearly asking TGD rules
        if is_short_topic_prompt(q) and not force_docs and not should_use_docs(q, pdf_pin=pdf, page_hint=page_hint):
            blocks = build_openai_blocks(q, sources_text="", images=None)
            resp = client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": blocks}],
                max_output_tokens=220,
            )
            return {
                "ok": True,
                "answer": resp.output_text,
                "used_docs": False,
                "sources_used": [],
                "retrieved_docs": [],
                "vision_used": False,
                "model": MODEL_NAME,
            }

        use_docs = force_docs or should_use_docs(q, pdf_pin=pdf, page_hint=page_hint)

        selected: List[Tuple[str, List[int], float]] = []
        sources_text = ""
        cites: List[Tuple[str,int]] = []
        images = None

        if use_docs:
            selected = iterative_select_pages(
                q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages
            )
            sources_text, cites = build_sources_bundle(selected)

            # Decide if vision is needed, but ONLY for top doc + first couple pages
            if selected:
                top_doc, pages, _score = selected[0]
                pdf_path = PDF_DIR / top_doc
                excerpt = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                if force_vision or needs_vision(q, excerpt):
                    images = pages_to_images(pdf_path, top_doc, pages[:2], dpi=dpi)

        blocks = build_openai_blocks(q, sources_text, images)

        resp = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": blocks}],
            max_output_tokens=750 if use_docs else 500,
        )

        return {
            "ok": True,
            "answer": resp.output_text,
            "used_docs": bool(use_docs and sources_text),
            "sources_used": [{"doc": d, "page": p} for d, p in cites],
            "retrieved_docs": [s[0] for s in selected],
            "vision_used": bool(images),
            "model": MODEL_NAME,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc().splitlines()[-16:],
        }


# ----------------------------
# /ask_stream (SSE typing effect)
# ----------------------------

@app.get("/ask_stream")
def ask_stream(
    q: str = Query(...),
    pdf: Optional[str] = Query(None),
    page_hint: Optional[int] = Query(None),
    force_docs: bool = Query(False),
    max_docs: int = Query(3),
    max_total_pages: int = Query(8),
    dpi: int = Query(120),
    force_vision: bool = Query(False),
):
    def sse():
        try:
            # Short topic => clarify (unless clearly doc intent)
            if is_short_topic_prompt(q) and not force_docs and not should_use_docs(q, pdf_pin=pdf, page_hint=page_hint):
                yield f"event: meta\ndata: model={MODEL_NAME};used_docs=False;vision=False\n\n"
                blocks = build_openai_blocks(q, sources_text="", images=None)
                stream = client.responses.create(
                    model=MODEL_NAME,
                    input=[{"role": "user", "content": blocks}],
                    max_output_tokens=220,
                    stream=True,
                )
                for event in stream:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            safe = delta.replace("\r","").replace("\n","\\n")
                            yield f"data: {safe}\n\n"
                    if getattr(event, "type", None) == "response.completed":
                        break
                yield "event: done\ndata: ok\n\n"
                return

            use_docs = force_docs or should_use_docs(q, pdf_pin=pdf, page_hint=page_hint)

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str,int]] = []
            images = None

            if use_docs:
                selected = iterative_select_pages(
                    q, pdf_pin=pdf, page_hint=page_hint, max_docs=max_docs, max_total_pages=max_total_pages
                )
                sources_text, cites = build_sources_bundle(selected)

                if selected:
                    top_doc, pages, _score = selected[0]
                    pdf_path = PDF_DIR / top_doc
                    excerpt = extract_pages_text(pdf_path, pages[:1], max_chars_per_page=900)
                    if force_vision or needs_vision(q, excerpt):
                        images = pages_to_images(pdf_path, top_doc, pages[:2], dpi=dpi)

            yield f"event: meta\ndata: model={MODEL_NAME};used_docs={bool(use_docs and sources_text)};vision={bool(images)}\n\n"
            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d,p in cites]) + "\n\n"

            blocks = build_openai_blocks(q, sources_text, images)

            stream = client.responses.create(
                model=MODEL_NAME,
                input=[{"role": "user", "content": blocks}],
                max_output_tokens=750 if use_docs else 500,
                stream=True,
            )

            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        safe = delta.replace("\r","").replace("\n","\\n")
                        yield f"data: {safe}\n\n"
                if getattr(event, "type", None) == "response.completed":
                    break

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r","").replace("\n"," ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"},
    )


# ----------------------------
# /chat_stream (history-aware SSE)
# ----------------------------

@app.post("/chat_stream")
def chat_stream(payload: Dict[str, Any] = Body(...)):
    """
    POST:
    {
      "messages": [{"role":"user","content":"..."}, ...],
      "pdf": "SomeDoc.pdf",
      "page_hint": 350,
      "force_docs": false
    }
    """
    messages = payload.get("messages", [])
    pdf = payload.get("pdf")
    page_hint = payload.get("page_hint")
    force_docs = bool(payload.get("force_docs", False))

    last_user = ""
    if isinstance(messages, list):
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "") or ""
                break

    def sse():
        try:
            trimmed = messages[-10:] if isinstance(messages, list) else []
            history_lines = []
            for m in trimmed:
                r = m.get("role")
                c = (m.get("content") or "").strip()
                if r in ("user", "assistant") and c:
                    history_lines.append(f"{r.upper()}: {c}")
            history_blob = "\n".join(history_lines).strip()

            use_docs = force_docs or should_use_docs(last_user, pdf_pin=pdf, page_hint=page_hint)

            selected: List[Tuple[str, List[int], float]] = []
            sources_text = ""
            cites: List[Tuple[str,int]] = []

            if use_docs:
                selected = iterative_select_pages(
                    last_user, pdf_pin=pdf, page_hint=page_hint, max_docs=3, max_total_pages=8
                )
                sources_text, cites = build_sources_bundle(selected)

            yield f"event: meta\ndata: model={MODEL_NAME};used_docs={bool(use_docs and sources_text)};vision=False\n\n"
            if cites:
                yield "event: meta\ndata: sources=" + ",".join([f"{d}:{p}" for d,p in cites]) + "\n\n"

            blocks = build_openai_blocks(last_user, sources_text, images=None, history_blob=history_blob)

            stream = client.responses.create(
                model=MODEL_NAME,
                input=[{"role":"user","content": blocks}],
                max_output_tokens=750 if use_docs else 500,
                stream=True,
            )

            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        safe = delta.replace("\r","").replace("\n","\\n")
                        yield f"data: {safe}\n\n"
                if getattr(event, "type", None) == "response.completed":
                    break

            yield "event: done\ndata: ok\n\n"

        except Exception as e:
            msg = str(e).replace("\r","").replace("\n"," ")
            yield f"event: error\ndata: {msg}\n\n"
            yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"},
    )


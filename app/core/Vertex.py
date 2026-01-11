from __future__ import annotations

import os, json, tempfile
from typing import Any, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

from app.core.config import (
    GOOGLE_CREDENTIALS_JSON, GCP_PROJECT_ID, GCP_LOCATION,
    EMBED_ENABLED, EMBED_MODEL_NAME,
    _VERTEX_EMBEDDINGS_AVAILABLE, TextEmbeddingModel,
    MODEL_COMPLIANCE,
)

_VERTEX_READY = False
_VERTEX_ERR: Optional[str] = None
_EMBED_MODEL: Any = None


# ============================================================
# VERTEX INIT
# ============================================================

_VERTEX_READY = False
_VERTEX_ERR: Optional[str] = None
_EMBED_MODEL: Any = None


def ensure_vertex_ready() -> None:
    global _VERTEX_READY, _VERTEX_ERR, _EMBED_MODEL
    if _VERTEX_READY:
        return
    try:
        if GOOGLE_CREDENTIALS_JSON and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            creds = json.loads(GOOGLE_CREDENTIALS_JSON)
            fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(creds, f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

        if not GCP_PROJECT_ID:
            raise RuntimeError("Missing VERTEX_PROJECT_ID (or GCP_PROJECT_ID)")
        if not GCP_LOCATION:
            raise RuntimeError("Missing VERTEX_LOCATION (or GCP_LOCATION)")

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

        # Lazy init embeddings model (optional)
        _EMBED_MODEL = None
        if EMBED_ENABLED and _VERTEX_EMBEDDINGS_AVAILABLE and TextEmbeddingModel is not None:
            try:
                _EMBED_MODEL = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)
            except Exception:
                _EMBED_MODEL = None

        _VERTEX_READY = True
        _VERTEX_ERR = None
    except Exception as e:
        _VERTEX_READY = False
        _VERTEX_ERR = str(e)


def get_model(model_name: str, system_prompt: str) -> GenerativeModel:
    ensure_vertex_ready()
    return GenerativeModel(model_name, system_instruction=[Part.from_text(system_prompt)])


def get_generation_config(is_evidence: bool) -> GenerationConfig:
    # More ChatGPT-like: stable, paragraphy
    if is_evidence:
        return GenerationConfig(temperature=0.2, top_p=0.8, max_output_tokens=3500)
    return GenerationConfig(temperature=0.65, top_p=0.9, max_output_tokens=3500)




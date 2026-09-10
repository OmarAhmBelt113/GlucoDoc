import os
from pathlib import Path

import torch
from dotenv import load_dotenv


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Environment
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Project Paths
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PDF_DIRECTORY = RAW_DATA_DIR / "pdf"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTORDB_DIR = DATA_DIR / "vectordb"
CHROMA_DIRECTORY = VECTORDB_DIR / "chroma"


# ============================================================
# API Keys
# ============================================================

LLAMA_CLOUD_API_KEY = os.getenv(
    "LLAMA_CLOUD_API_KEY"
)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)


# ============================================================
# Embedding Model
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# ============================================================
# ChromaDB
# ============================================================

CHROMA_COLLECTION_NAME = "glucodoc"


# ============================================================
# Retrieval
# ============================================================

RETRIEVAL_K = 5


# ============================================================
# LLM
# ============================================================

LLM_MODEL_NAME = (
    "gemini-3.1-flash-lite-preview"
)

LLM_TEMPERATURE = 0.0
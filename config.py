"""
Central config — loads .env and exposes all shared clients/constants.
Import from here instead of initializing clients in every file.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ── API keys ──────────────────────────────────────────────────
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX_NAME", "financial-rag")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")

# ── Tesseract / Poppler (Windows optional) ────────────────────
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "")
POPPLER_PATH   = os.environ.get("POPPLER_PATH", "")

# ── Shared clients (single instances) ─────────────────────────
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index  = pinecone_client.Index(PINECONE_INDEX)

groq_client = Groq(api_key=GROQ_API_KEY)

# Single shared embedding model — imported by retrieve.py and pinecone_store.py
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ── Model names ───────────────────────────────────────────────
TEXT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Retrieval ─────────────────────────────────────────────────
SCORE_THRESHOLD = 0.45   # min similarity score — below this is irrelevant
TOP_K           = 8

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE      = 500
OVERLAP_PERCENT = 0.10

# ── Folders ───────────────────────────────────────────────────
REPORTS_FOLDER = "reports"
OUTPUT_FOLDER  = "output"
CHUNKS_FOLDER  = "chunks"
IMAGES_FOLDER  = "images"

# Financial RAG Chat

A RAG-based financial report Q&A system built with Chainlit, Pinecone, and Groq.

## Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env` and fill in your keys:
```
GROQ_API_KEY=gsk_...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=financial-rag
GEMINI_API_KEY=AIza...          # optional

# Windows only (leave blank on Linux/Mac):
# TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
# POPPLER_PATH=C:\poppler-26.02.0\Library\bin
```

### 3. Run
```
chainlit run app.py
```

## File Structure
| File | Purpose |
|------|---------|
| `config.py` | Central config — all API clients, constants, shared model |
| `app.py` | Chainlit UI + upload handlers |
| `pdf_extractor.py` | Extract text + tables from PDFs |
| `chunker.py` | Split text into overlapping chunks with cross-page windowing |
| `pinecone_store.py` | Batch encode + upload chunks to Pinecone |
| `retrieve.py` | Hybrid dense + BM25 retrieval with year/company filtering |
| `rag_chain.py` | Combines retrieval + LLM generation |
| `llm_gemini.py` | Groq LLM calls (text + vision) |
| `llm.py` | Prompt templates and context builder |
| `run_chat.py` | Terminal chat interface (no UI) |
| `base.py` | Abstract retriever base class |

## Naming PDFs
Include the year in the filename for best results:
- `Mahindra_2024.pdf` → year metadata = 2024
- `TataMotors_FY2023.pdf` → year metadata = 2023

## Features
- PDF upload → text + table extraction → Pinecone indexing
- Image upload → vision model extraction → Pinecone indexing
- Hybrid search: dense vectors + BM25 keyword fallback
- Year and company metadata filtering
- Clarification flow for ambiguous questions
- Conversation history with follow-up rewriting

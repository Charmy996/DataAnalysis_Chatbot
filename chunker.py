"""
Chunker — splits extracted text into overlapping chunks.
- year read from JSON (set by pdf_extractor) not filename
- cross-page windowing
- globally unique chunk IDs
"""
import json
import os
import hashlib
from config import CHUNK_SIZE, OVERLAP_PERCENT, CHUNKS_FOLDER, OUTPUT_FOLDER

OVERLAP = int(CHUNK_SIZE * OVERLAP_PERCENT)
STEP    = CHUNK_SIZE - OVERLAP

os.makedirs(CHUNKS_FOLDER, exist_ok=True)


def make_chunk_id(company, page, idx):
    raw = f"{company}_{page}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16] + f"_{company[:8]}"


def chunk_company(company, input_folder=OUTPUT_FOLDER, chunk_folder=CHUNKS_FOLDER):
    input_path = os.path.join(input_folder, f"{company}_text.json")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return 0

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Year is stored in JSON by pdf_extractor — no filename guessing needed
    year = pages[0].get("year") if pages else None

    all_words = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        pg = page.get("page", 0)
        for word in text.split():
            all_words.append((word, pg))

    chunks = []
    idx = 0
    for start in range(0, len(all_words), STEP):
        word_page_pairs = all_words[start:start + CHUNK_SIZE]
        if not word_page_pairs:
            continue
        words     = [wp[0] for wp in word_page_pairs]
        pages_hit = [wp[1] for wp in word_page_pairs]
        page_num  = max(set(pages_hit), key=pages_hit.count)
        chunks.append({
            "chunk_id":   make_chunk_id(company, page_num, idx),
            "company":    company,
            "year":       year,
            "page":       page_num,
            "word_count": len(words),
            "chunk":      " ".join(words),
            "source":     "text"
        })
        idx += 1

    output_path = os.path.join(chunk_folder, f"{company}_text_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"{company} -> {len(chunks)} text chunks (year={year})")
    return len(chunks)


def load_chunks_for_company(company, chunk_folder=CHUNKS_FOLDER):
    path = os.path.join(chunk_folder, f"{company}_text_chunks.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    for file in os.listdir(OUTPUT_FOLDER):
        if not file.lower().endswith("_text.json"):
            continue
        company = file.replace("_text.json", "")
        chunk_company(company)
    print("\nChunking Complete!")

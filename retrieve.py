"""
Retrieval — hybrid dense + BM25 keyword search with year/company filtering.
Fixes:
- Hybrid BM25 + dense search
- Year + company metadata filters
- Score threshold to filter irrelevant chunks
- Single shared embed_model from config
- Entity/label search for subsidiary names
- Chunk cache
"""
import re
import json
import os
from rank_bm25 import BM25Okapi
from config import pinecone_index, embed_model, CHUNKS_FOLDER, SCORE_THRESHOLD, TOP_K

# ── Cache ─────────────────────────────────────────────────────
_cached_chunks = None
_bm25_index    = None
_bm25_chunks   = None   # parallel list to _bm25_index corpus


def load_all_chunks():
    global _cached_chunks, _bm25_index, _bm25_chunks
    if _cached_chunks is not None:
        return _cached_chunks
    _cached_chunks = []
    if not os.path.exists(CHUNKS_FOLDER):
        return _cached_chunks
    for f in os.listdir(CHUNKS_FOLDER):
        if f.endswith(".json"):
            with open(os.path.join(CHUNKS_FOLDER, f), encoding="utf-8") as fp:
                _cached_chunks.extend(json.load(fp))
    # Build BM25 index over all cached chunks
    corpus = [c.get("chunk", "").lower().split() for c in _cached_chunks]
    if corpus:
        _bm25_index  = BM25Okapi(corpus)
        _bm25_chunks = _cached_chunks
    return _cached_chunks


def invalidate_cache():
    global _cached_chunks, _bm25_index, _bm25_chunks
    _cached_chunks = _bm25_index = _bm25_chunks = None


def _bm25_search(question, top_n=5):
    """Keyword BM25 search over local chunks."""
    if _bm25_index is None:
        return []
    tokens = question.lower().split()
    scores = _bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = []
    for i in top_idx:
        if scores[i] > 0:
            c = _bm25_chunks[i]
            results.append({
                "score":   min(scores[i] / 10, 1.0),  # normalise loosely
                "company": c.get("company", ""),
                "year":    c.get("year"),
                "page":    c.get("page", 0),
                "chunk":   c.get("chunk", ""),
                "source":  c.get("source", "text")
            })
    return results


def _local_number_search(numbers, all_chunks, top_n=3):
    matches = []
    for chunk in all_chunks:
        text = chunk.get("chunk", "").replace("(", "").replace(")", "").replace(",", "")
        if any(n.replace(",", "") in text for n in numbers):
            matches.append({
                "score":   0.95,
                "company": chunk.get("company", ""),
                "year":    chunk.get("year"),
                "page":    chunk.get("page", 0),
                "chunk":   chunk.get("chunk", ""),
                "source":  chunk.get("source", "text")
            })
        if len(matches) >= top_n:
            break
    return matches


def _entity_search(question, all_chunks, top_n=5):
    """Match subsidiary/entity names in table chunk labels."""
    stopwords = {"what","is","the","for","of","in","a","an","tell","me","value",
                 "give","find","show","how","much","were","was","are","its","and",
                 "or","to","at","with","limited","ltd"}
    words = [w for w in re.findall(r'[a-zA-Z]+', question.lower())
             if w not in stopwords and len(w) > 2]
    if not words:
        return []
    matches = []
    for chunk in all_chunks:
        if chunk.get("source") != "table":
            continue
        label = chunk.get("label", "").lower()
        score = sum(1 for w in words if w in label)
        if score >= 2:
            matches.append({
                "score":   min(0.85 + score * 0.02, 0.99),
                "company": chunk.get("company", ""),
                "year":    chunk.get("year"),
                "page":    chunk.get("page", 0),
                "chunk":   chunk.get("chunk", ""),
                "source":  "table"
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_n]


def retrieve(question, company=None, year=None, top_k=TOP_K):
    """
    Hybrid retrieval: dense Pinecone + BM25 keyword + number + entity search.
    Applies score threshold and optional company/year filters.
    """
    numbers = re.findall(r'\d[\d,\.]+', question)

    # ── Dense search ──────────────────────────────────────────
    query_vec = embed_model.encode(question, normalize_embeddings=True)
    filter_dict = {}
    if company:
        filter_dict["company"] = company
    if year:
        filter_dict["year"] = year

    pinecone_results = pinecone_index.query(
        vector=query_vec.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    output = []
    seen_ids = set()

    for match in pinecone_results["matches"]:
        if match["score"] < SCORE_THRESHOLD:
            continue
        chunk_text = match["metadata"]["text"]
        if chunk_text in seen_ids:
            continue
        seen_ids.add(chunk_text)
        output.append({
            "score":   match["score"],
            "company": match["metadata"]["company"],
            "year":    match["metadata"].get("year"),
            "page":    match["metadata"]["page"],
            "chunk":   chunk_text,
            "source":  match["metadata"].get("source", "text")
        })

    # ── Local searches (BM25 + number + entity) ───────────────
    all_chunks = load_all_chunks()

    def add_unique(matches):
        for m in matches:
            if m["chunk"] not in seen_ids:
                seen_ids.add(m["chunk"])
                output.insert(0, m)

    # BM25 keyword search
    add_unique(_bm25_search(question, top_n=5))

    # Exact number search
    if numbers:
        add_unique(_local_number_search(numbers, all_chunks))

    # Entity/label search for subsidiaries
    add_unique(_entity_search(question, all_chunks))

    return output


if __name__ == "__main__":
    q = input("Question: ")
    results = retrieve(q)
    for r in results:
        print(f"\n--- {r['company']} | year={r['year']} | page={r['page']} | score={r['score']:.2f} [{r.get('source','text')}]")
        print(r["chunk"][:300])

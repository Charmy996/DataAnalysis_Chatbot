"""
Pinecone upload — batch encoding, shared model, year metadata.
"""
import json
import os
from config import pinecone_index, embed_model, CHUNKS_FOLDER


def load_chunks(chunks_dir=CHUNKS_FOLDER):
    all_chunks = []
    if not os.path.exists(chunks_dir):
        return all_chunks
    for filename in os.listdir(chunks_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            all_chunks.extend(data)
            print(f"  Loaded {len(data)} chunks from {filename}")
    return all_chunks


def upsert_chunks(chunks=None, chunks_dir=CHUNKS_FOLDER):
    """Batch encode and upload chunks. Includes year in Pinecone metadata."""
    if chunks is None:
        chunks = load_chunks(chunks_dir)
    if not chunks:
        print("No chunks to upload.")
        return 0

    print(f"Encoding {len(chunks)} chunks...")
    texts      = [c["chunk"] for c in chunks]
    embeddings = embed_model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )

    vectors = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "company": chunk["company"],
            "page":    chunk["page"],
            "text":    chunk["chunk"],
            "source":  chunk.get("source", "text")
        }
        if chunk.get("year"):
            metadata["year"] = chunk["year"]
        if chunk.get("label"):
            metadata["label"] = chunk["label"]

        vectors.append({
            "id":     str(chunk["chunk_id"]),
            "values": embeddings[i].tolist(),
            "metadata": metadata
        })

    for i in range(0, len(vectors), 100):
        batch = vectors[i:i + 100]
        pinecone_index.upsert(vectors=batch)
        print(f"  Batch {i // 100 + 1} uploaded ({len(batch)} vectors)")

    print(f"Total uploaded: {len(vectors)}")
    return len(vectors)


if __name__ == "__main__":
    upsert_chunks()

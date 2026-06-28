from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="pcsk_cT42z_NYvXHVbKRhdxk3PSNxasHDTqdQLJ3HYZj1APKUmLk3BgQaoMkbNZzSA1CZwoT6v")

existing = pc.list_indexes().names()

if "financial-rag" not in existing:

    pc.create_index(
        name="financial-rag",
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created")
else:
    print("Index already exists")
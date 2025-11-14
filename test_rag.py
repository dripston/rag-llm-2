import os
import json
from pinecone import Pinecone
import requests


# ----------------------------
# CONFIG
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
BASE_URL = "https://api.sambanova.ai/v1"


# ----------------------------
# 1. TEST SAMBANOVA EMBEDDINGS
# ----------------------------
def get_sambanova_embedding(text):
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBED_MODEL,
        "input": [text]
    }

    print("\n🔵 Requesting embedding from SambaNova…")

    res = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload
    )

    if res.status_code != 200:
        print("❌ SambaNova embedding error:")
        print(res.text)
        return None

    data = res.json()
    emb = data["data"][0]["embedding"]

    print(f"✅ Received embedding with dimension: {len(emb)}")

    return emb


# ----------------------------
# 2. TEST PINECONE INDEX
# ----------------------------
def test_pinecone_query(embed):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("\n🔵 Fetching Pinecone index details…")

    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    print("\n📊 Index Stats:")
    print(json.dumps(stats, indent=4))

    index_dim = stats["dimension"]
    print(f"\n📌 Index dimension = {index_dim}")

    if index_dim != len(embed):
        print("\n❌ DIMENSION MISMATCH!")
        print(f"Embedding = {len(embed)}, Index = {index_dim}")
        print("Fix: Recreate Pinecone index with dimension=4096")
        return

    print("\n🔵 Querying Pinecone…")

    res = index.query(
        vector=embed,
        top_k=3,
        include_metadata=True
    )

    print("\n📥 Pinecone Query Response:")
    print(json.dumps(res, indent=4))

    matches = res.get("matches", [])

    if not matches:
        print("\n⚠️ No matches found!")
        print("Possible reasons:")
        print("1. Index is empty")
        print("2. Wrong dimension")
        print("3. You didn't re-ingest data")
    else:
        print("\n✅ Matches found!")
        for m in matches:
            print(f"\nID: {m['id']}")
            print(f"Score: {m['score']}")
            print("Content:", m['metadata'].get("content", "")[:200], "...")


# ----------------------------
# MAIN TEST RUNNER
# ----------------------------
if __name__ == "__main__":
    print("\n========== RAG SYSTEM TEST ==========")

    test_text = "What was the diagnosis of the patient?"

    # Step 1: Get embedding
    emb = get_sambanova_embedding(test_text)

    if emb:
        # Step 2: Test Pinecone query
        test_pinecone_query(emb)

    print("\n========== TEST COMPLETE ==========\n")

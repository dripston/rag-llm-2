import os
import json
from pinecone import Pinecone
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-index")  # Fixed to match .env
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "E5-Mistral-7B-Instruct")  # Added default
BASE_URL = "https://api.sambanova.ai/v1"


# ----------------------------
# 1. TEST SAMBANOVA EMBEDDINGS
# ----------------------------
def get_sambanova_embedding(text):
    # Check for required environment variables
    if not SAMBANOVA_API_KEY:
        print("❌ Error: SAMBANOVA_API_KEY not set in environment variables")
        return None
        
    if not EMBED_MODEL:
        print("❌ Error: EMBEDDING_MODEL not set in environment variables")
        return None
    
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
    # Check for required environment variables
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not set in environment variables")
        return
        
    if not INDEX_NAME:
        print("❌ Error: PINECONE_INDEX_NAME not set in environment variables")
        return
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("\n🔵 Fetching Pinecone index details…")

    try:
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"❌ Error connecting to Pinecone index: {e}")
        return

    try:
        stats = index.describe_index_stats()
        print("\n📊 Index Stats:")
        # Safely convert to string representation
        print(str(stats))
    except Exception as e:
        print(f"❌ Error getting index stats: {e}")
        return

    # Get dimension from the stats - try multiple approaches
    index_dim = 0
    try:
        # Try direct attribute access
        index_dim = getattr(stats, 'dimension', 0)
    except:
        print("⚠️ Could not determine index dimension")
        index_dim = 0
        
    print(f"\n📌 Index dimension = {index_dim}")

    if index_dim != len(embed):
        print("\n❌ DIMENSION MISMATCH!")
        print(f"Embedding = {len(embed)}, Index = {index_dim}")
        print("Fix: Recreate Pinecone index with dimension=4096")
        return

    print("\n🔵 Querying Pinecone…")

    try:
        res = index.query(
            vector=embed,
            top_k=3,
            include_metadata=True
        )
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return

    print("\n📥 Pinecone Query Response:")
    # Safely convert response to string
    print(str(res))

    # Access matches - try multiple approaches
    matches = []
    try:
        # Try direct attribute access
        matches = getattr(res, 'matches', [])
    except:
        print("⚠️ Could not access matches")
        matches = []

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
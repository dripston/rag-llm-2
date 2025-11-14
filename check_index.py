from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

print("\n========= CHECKING INDEX =========")

if not index_name:
    print("❌ PINECONE_INDEX_NAME not set in environment variables")
    exit(1)

if not os.getenv("PINECONE_API_KEY"):
    print("❌ PINECONE_API_KEY not set in environment variables")
    exit(1)

try:
    # Fetch index description
    desc = pc.describe_index(index_name)
    
    print(f"Index Name: {index_name}")
    print(f"Dimension: {desc.dimension}")
    print(f"Metric: {desc.metric}")
    print(f"Host: {desc.host}")
    
    # Check index stats
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Total Vector Count: {stats.total_vector_count}")
    print(f"Namespaces: {stats.namespaces}")
    
except Exception as e:
    print(f"❌ Error checking index: {e}")

print("=================================\n")
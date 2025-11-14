#!/usr/bin/env python3
"""
Test script to retrieve and display all data from Pinecone index.
This script loads environment variables and fetches all vectors from the Pinecone index.
"""

import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def get_all_pinecone_data():
    """Retrieve all data from Pinecone index"""
    
    # Get Pinecone configuration from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-index")
    
    if not api_key:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        return
    
    print(f"🔧 Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)
    
    print(f"🔍 Checking for index: {index_name}")
    existing_indexes = pc.list_indexes().names()
    print(f"📚 Existing indexes: {existing_indexes}")
    
    if index_name not in existing_indexes:
        print(f"❌ Index '{index_name}' not found")
        return
    
    # Connect to index
    print(f"🔗 Connecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Get index stats
    print("📊 Getting index stats...")
    stats = {}
    try:
        stats = index.describe_index_stats()
        print(f"📈 Index stats: {json.dumps(stats, indent=2, default=str)}")
    except Exception as e:
        print(f"⚠️ Could not fetch index stats: {e}")
    
    # Try to fetch all vectors (note: this might not work with all Pinecone configurations)
    print("🔍 Attempting to fetch all vectors...")
    try:
        # For serverless indexes, we can't directly list all vectors
        # Instead, we'll do a sample query to get some data
        print("📝 Performing sample query to retrieve vectors...")
        
        # Create a simple query vector (all zeros)
        dimension = stats.get('dimension', 4096) if stats else 4096
        query_vector = [0.0] * dimension
        
        # Query for all vectors (high top_k)
        response = index.query(
            vector=query_vector,
            top_k=100,  # Adjust this number based on how many vectors you expect
            include_metadata=True,
            include_values=False  # Set to True if you want to see the actual vector values
        )
        
        # Access matches correctly
        matches = []
        if isinstance(response, dict):
            matches = response.get("matches", [])
        else:
            matches = getattr(response, "matches", [])
        
        print(f"✅ Query successful. Found {len(matches)} matches")
        
        if matches:
            print("\n📄 Retrieved vectors:")
            for i, match in enumerate(matches):
                print(f"\n--- Vector {i+1} ---")
                match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "N/A")
                match_score = match.get("score") if isinstance(match, dict) else getattr(match, "score", "N/A")
                match_metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
                
                print(f"ID: {match_id}")
                print(f"Score: {match_score}")
                print(f"Metadata: {json.dumps(match_metadata, indent=2, default=str)}")
        else:
            print("📭 No vectors found in the index")
            
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        import traceback
        traceback.print_exc()

def test_pinecone_connection():
    """Test basic Pinecone connection and configuration"""
    print("=== Pinecone Connection Test ===")
    
    # Required environment variables
    required_vars = ["PINECONE_API_KEY"]
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ Missing environment variable: {var}")
            return False
        # Show variable is set (but don't print the actual API key)
        print(f"✅ {var}: {'*' * len(value) if var == 'PINECONE_API_KEY' else value}")
    
    # Optional variables
    optional_vars = ["PINECONE_INDEX_NAME", "EMBEDDING_DIMENSION"]
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
    
    return True

if __name__ == "__main__":
    print("🚀 Pinecone Data Retrieval Test Script")
    print("=" * 50)
    
    # Test connection first
    if not test_pinecone_connection():
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Get all data
    get_all_pinecone_data()
    
    print("\n🏁 Test completed")
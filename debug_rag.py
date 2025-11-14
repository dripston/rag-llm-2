import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag_service import RAGService

def debug_rag():
    print("========== DEBUGGING RAG PIPELINE ==========")
    
    # Initialize RAG service
    try:
        rag_service = RAGService()
        print("✅ RAG Service initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG service: {e}")
        return
    
    # Test simple query first
    print("\n🔵 Testing simple query...")
    try:
        query = "What is pneumonia?"
        response = rag_service.query(query, top_k=1)
        print(f"✅ Simple query processed successfully")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in simple query: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n========== DEBUG COMPLETE ==========")

if __name__ == "__main__":
    debug_rag()
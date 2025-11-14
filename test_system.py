import os
import sys
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    """Test Pinecone connection and index creation"""
    try:
        from pinecone_service import PineconeService
        pinecone_service = PineconeService()
        print("✓ Pinecone service initialized successfully")
        
        # Test upsert
        test_vectors = [{
            "id": "test-vector-1",
            "values": [0.1] * 4096,  # 4096-dimensional vector
            "metadata": {"content": "This is a test document"}
        }]
        
        success = pinecone_service.upsert_vectors(test_vectors)
        if success:
            print("✓ Vector upsert successful")
        else:
            print("✗ Vector upsert failed")
            
        # Test query
        query_vector = [0.1] * 4096
        results = pinecone_service.query_similar(query_vector, top_k=1)
        print(f"✓ Vector query returned {len(results)} results")
        
        # Clean up
        pinecone_service.delete_vectors(["test-vector-1"])
        print("✓ Test vector cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Pinecone test failed: {e}")
        return False

def test_embedding_service():
    """Test SambaNova embedding service"""
    try:
        from embedding_service import SambaNovaEmbedding
        embedding_service = SambaNovaEmbedding()
        print("✓ SambaNova embedding service initialized successfully")
        
        # Test single embedding
        test_text = "This is a test sentence for medical diagnosis."
        embedding = embedding_service.get_single_embedding(test_text)
        
        if len(embedding) == 4096:
            print("✓ Single embedding generation successful")
        else:
            print(f"✗ Embedding dimension mismatch: expected 4096, got {len(embedding)}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Embedding service test failed: {e}")
        return False

def test_rag_service():
    """Test RAG service"""
    try:
        from rag_service import RAGService
        rag_service = RAGService()
        print("✓ RAG service initialized successfully")
        
        # Test adding a document
        test_content = "Patient has fever and cough. Likely diagnosis: common cold."
        success = rag_service.add_document(test_content, {"source": "test"})
        
        if success:
            print("✓ Document added to RAG successfully")
        else:
            print("✗ Failed to add document to RAG")
            return False
            
        # Test querying
        response = rag_service.query("What is the diagnosis for fever and cough?")
        if response:
            print("✓ RAG query successful")
        else:
            print("✗ RAG query returned empty response")
            return False
            
        return True
    except Exception as e:
        print(f"✗ RAG service test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running Medical RAG Chatbot Tests...\n")
    
    tests = [
        ("Pinecone Connection", test_pinecone_connection),
        ("Embedding Service", test_embedding_service),
        ("RAG Service", test_rag_service)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag_service import RAGService

def test_full_rag_pipeline():
    print("========== TESTING FULL RAG PIPELINE ==========")
    
    # Initialize RAG service
    try:
        rag_service = RAGService()
        print("✅ RAG Service initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG service: {e}")
        return
    
    # Test document ingestion
    print("\n🔵 Testing document ingestion...")
    try:
        test_document = "The patient is a 65-year-old male with a history of diabetes and hypertension. He presented with high fever, productive cough with yellow sputum, and chest pain. Physical examination revealed decreased breath sounds in the right lower lung field. Laboratory tests showed elevated white blood cell count. Chest X-ray confirmed consolidation in the right lower lobe. The diagnosis was bacterial pneumonia. Treatment included intravenous antibiotics, fluids, and oxygen therapy."
        metadata = {
            "patient_id": "P98765",
            "visit_date": "2023-11-15",
            "doctor": "Dr. Smith"
        }
        
        success = rag_service.add_document(test_document, metadata)
        if success:
            print("✅ Document ingested successfully")
        else:
            print("❌ Document ingestion failed")
            return
    except Exception as e:
        print(f"❌ Error in document ingestion: {e}")
        return
    
    # Wait a moment for indexing
    import time
    time.sleep(2)
    
    # Test querying
    print("\n🔵 Testing RAG query...")
    try:
        query = "What was the patient's diagnosis and treatment?"
        response = rag_service.query(query, top_k=3)
        print(f"✅ Query processed successfully")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in RAG query: {e}")
        return
    
    # Test another query
    print("\n🔵 Testing specific query...")
    try:
        query = "What medications were prescribed to the patient?"
        response = rag_service.query(query, top_k=3)
        print(f"✅ Specific query processed successfully")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in specific query: {e}")
        return
    
    print("\n========== FULL RAG PIPELINE TEST COMPLETE ==========")

if __name__ == "__main__":
    test_full_rag_pipeline()
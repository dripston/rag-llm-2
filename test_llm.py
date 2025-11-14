import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from llm_service import SambaNovaLLM

def test_llm_service():
    print("========== TESTING LLM SERVICE ==========")
    
    # Initialize LLM service
    try:
        llm = SambaNovaLLM()
        print("✅ LLM Service initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM service: {e}")
        return
    
    # Test basic response generation
    print("\n🔵 Testing basic response generation...")
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant."
            },
            {
                "role": "user",
                "content": "What is the common treatment for pneumonia?"
            }
        ]
        
        response = llm.generate_response(messages, temperature=0.7, max_tokens=200)
        print(f"✅ Basic response generated successfully")
        print(f"Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Error in basic response generation: {e}")
        return
    
    # Test response with context
    print("\n🔵 Testing response with context...")
    try:
        context = "The patient presented with fever, cough, and shortness of breath. Physical examination revealed crackles in the lower lung fields. Chest X-ray showed bilateral infiltrates. The diagnosis was community-acquired pneumonia."
        query = "What was the patient's diagnosis?"
        
        response = llm.generate_response_with_context(context, query, temperature=0.7, max_tokens=200)
        print(f"✅ Response with context generated successfully")
        print(f"Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Error in response with context: {e}")
        return
    
    print("\n========== LLM SERVICE TEST COMPLETE ==========")

if __name__ == "__main__":
    test_llm_service()
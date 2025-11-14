import os
from pinecone import Pinecone

def initialize_pinecone():
    """
    Initialize Pinecone index for the RAG system
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=int(os.getenv("EMBEDDING_DIMENSION", 4096)),
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
        print(f"Index {index_name} created successfully")
    else:
        print(f"Index {index_name} already exists")
    
    return pc.Index(index_name)

if __name__ == "__main__":
    try:
        initialize_pinecone()
        print("Pinecone initialization completed")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
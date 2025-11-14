import pinecone
from pinecone import Pinecone
import os
from typing import List, Dict, Any
import numpy as np

class PineconeService:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-index")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=int(os.getenv("EMBEDDING_DIMENSION", 4096)),
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        
        # Connect to the index
        self.index = self.pc.Index(name=self.index_name)
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors into the Pinecone index
        
        Args:
            vectors: List of dictionaries with 'id', 'values', and 'metadata'
        """
        try:
            # Convert to the format expected by Pinecone
            formatted_vectors = []
            for vector in vectors:
                formatted_vectors.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": vector["metadata"]
                })
            
            self.index.upsert(vectors=formatted_vectors)
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
    
    def query_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query similar vectors from the Pinecone index
        
        Args:
            query_vector: The query vector
            top_k: Number of similar vectors to return
            
        Returns:
            List of similar vectors with metadata
        """
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            # Convert to dictionary using vars() and access matches
            response_dict = vars(response)
            return response_dict.get('matches', [])
        except Exception as e:
            print(f"Error querying vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs from the Pinecone index
        """
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
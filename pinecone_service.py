import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from typing import List, Dict, Any
import numpy as np

class PineconeService:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=int(os.getenv("EMBEDDING_DIMENSION", 4096)),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors into the Pinecone index
        
        Args:
            vectors: List of dictionaries with 'id', 'values', and 'metadata'
        """
        try:
            self.index.upsert(vectors=vectors)
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
            return response['matches']
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
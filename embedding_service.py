import os
import numpy as np
import requests
from typing import List
import json

class SambaNovaEmbedding:
    def __init__(self):
        self.api_key = os.getenv("SAMBANOVA_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", 4096))
        self.base_url = "https://api.sambanova.ai/v1"
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using SambaNova API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting embeddings: {response.text}")
            
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        """
        return self.get_embeddings([text])[0]
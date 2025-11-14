import logging
import os
import numpy as np
import requests
from typing import List
import json

# Configure logging
logger = logging.getLogger(__name__)

class SambaNovaEmbedding:
    def __init__(self):
        self.api_key = os.getenv("SAMBANOVA_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", 4096))
        self.base_url = "https://api.sambanova.ai/v1"
        
        logger.info("Initializing SambaNova Embedding service...")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Expected dimension: {self.dimension}")
        logger.info(f"Base URL: {self.base_url}")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using SambaNova API
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        logger.info(f"Embedding model used for INGESTION: {self.model}")  # Added required logging
        logger.debug(f"Text lengths: {[len(text) for text in texts]}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        logger.debug("Calling SambaNova embeddings API...")
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error getting embeddings: {response.text}")
            logger.error(f"Status code: {response.status_code}")
            raise Exception(f"Error getting embeddings: {response.text}")
            
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        logger.info(f"Embeddings generated successfully. Dimension: {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        """
        logger.debug(f"Generating embedding for single text with length: {len(text)}")
        logger.info(f"Embedding model used for QUERY: {self.model}")  # Added required logging
        result = self.get_embeddings([text])[0]
        logger.debug(f"Single embedding generated with dimension: {len(result)}")
        return result
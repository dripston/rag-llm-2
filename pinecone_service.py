import logging
import pinecone
from pinecone import Pinecone
import os
from typing import List, Dict, Any
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-index")
        
        logger.info("Initializing Pinecone service...")
        if self.api_key:
            logger.info(f"Using API key: {self.api_key[:8]}...")  # Log only first 8 characters for security
        else:
            logger.warning("PINECONE_API_KEY not found in environment variables")
        logger.info(f"Using index name: {self.index_name}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        logger.info("Pinecone client initialized")
        
        # Create index if it doesn't exist
        logger.info("Checking if index exists...")
        index_names = self.pc.list_indexes().names()
        logger.info(f"Existing indexes: {index_names}")
        
        if self.index_name not in index_names:
            logger.info(f"Creating new index: {self.index_name}")
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
            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Index {self.index_name} already exists")
        
        # Connect to the index
        logger.info(f"Connecting to index: {self.index_name}")
        self.index = self.pc.Index(name=self.index_name)
        logger.info("Connected to Pinecone index successfully")
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        """
        Upsert vectors into the Pinecone index
        
        Args:
            vectors: List of dictionaries with 'id', 'values', and 'metadata'
        """
        try:
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone index")
            if vectors:
                logger.debug(f"First vector sample - ID: {vectors[0]['id']}, Values length: {len(vectors[0]['values'])}, Metadata keys: {list(vectors[0]['metadata'].keys())}")
            
            # Convert to the format expected by Pinecone
            formatted_vectors = []
            for vector in vectors:
                formatted_vectors.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": vector["metadata"]
                })
            
            logger.debug("Calling Pinecone upsert...")
            response = self.index.upsert(vectors=formatted_vectors)
            logger.debug(f"Pinecone upsert response: {response}")
            logger.info("Vectors upserted successfully")
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            logger.exception(e)
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
            logger.info(f"Querying Pinecone for top {top_k} similar vectors")
            logger.debug(f"Query vector dimension: {len(query_vector)}")
            
            logger.debug("Calling Pinecone query...")
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            logger.debug("Pinecone query completed")
            logger.debug(f"Raw Pinecone response: {response}")
            
            # Convert response to dictionary using vars() and access matches
            response_dict = vars(response)
            matches = response_dict.get('matches', [])
            logger.info(f"Query returned {len(matches)} matches")
            
            # Log details about matches for debugging
            for i, match in enumerate(matches):
                logger.debug(f"Match {i+1}: id={match.get('id')}, score={match.get('score')}, metadata_keys={list(match.get('metadata', {}).keys())}")
            
            return matches
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            logger.exception(e)
            return []
    
    def list_vectors(self) -> List[str]:
        """
        List all vector IDs in the index (for debugging purposes)
        """
        try:
            logger.info("Listing vectors in index")
            # Note: Pinecone doesn't have a direct list all IDs method in the new API
            # We'll return an empty list for now
            logger.info("Vector listing not implemented in new Pinecone API")
            return []
        except Exception as e:
            logger.error(f"Error listing vectors: {e}")
            logger.exception(e)
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs from the Pinecone index
        """
        try:
            logger.info(f"Deleting {len(ids)} vectors from Pinecone index")
            response = self.index.delete(ids=ids)
            logger.debug(f"Pinecone delete response: {response}")
            logger.info("Vectors deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            logger.exception(e)
            return False
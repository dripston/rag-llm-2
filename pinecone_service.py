import logging
import os
import json
import ast
from typing import List, Dict, Any
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medical-assistant-index")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", 4096))

        logger.info("Initializing Pinecone service…")

        if not self.api_key:
            raise ValueError("❌ PINECONE_API_KEY is missing in environment variables")

        self.pc = Pinecone(api_key=self.api_key)
        logger.info("Pinecone client initialized")

        # List indexes
        existing = self.pc.list_indexes().names()
        logger.info(f"Existing indexes: {existing}")

        # Create if missing
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )

        # Connect to index
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to index: {self.index_name}")

        # Optional: check stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not fetch index stats: {e}")

    # -------------------------------------------------------------------------
    # UPSERT VECTORS
    # -------------------------------------------------------------------------
    def upsert_vectors(self, vectors) -> bool:
        try:
            if not vectors:
                logger.warning("No vectors to upsert")
                return False

            for v in vectors:
                if len(v["values"]) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {len(v['values'])}"
                    )

            logger.info(f"Upserting {len(vectors)} vectors…")
            
            # Add logging for the first vector to debug
            if vectors:
                logger.info(f"First vector ID: {vectors[0].get('id', 'N/A')}")
                logger.info(f"First vector dimension: {len(vectors[0].get('values', []))}")
                logger.info(f"First vector metadata keys: {list(vectors[0].get('metadata', {}).keys())}")

            self.index.upsert(vectors=vectors)
            logger.info("✔ Upsert successful")

            return True

        except Exception as e:
            logger.error(f"❌ Error upserting vectors: {e}")
            logger.exception(e)
            return False

    # -------------------------------------------------------------------------
    # QUERY SIMILAR VECTORS
    # -------------------------------------------------------------------------
    def query_similar(self, query_vector, top_k: int = 5):
        try:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
                )

            logger.info(f"Querying Pinecone (top_k={top_k})")

            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            # NEW serverless format = dict
            if isinstance(response, dict):
                matches = response.get("matches", [])
            else:
                matches = getattr(response, "matches", [])

            logger.info(f"✔ Pinecone returned {len(matches)} matches")

            cleaned = []
            for m in matches:
                meta = m.get("metadata", {})

                # Metadata cleanup: convert string → dict if needed
                if isinstance(meta, str):
                    try:
                        meta = ast.literal_eval(meta)
                    except Exception:
                        try:
                            meta = json.loads(meta.replace("'", '"'))
                        except Exception:
                            logger.warning(f"Could not parse metadata for vector {m.get('id')}")
                            meta = {}

                cleaned.append({
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "metadata": meta
                })

            return cleaned

        except Exception as e:
            logger.error(f"❌ Error querying Pinecone: {e}")
            logger.exception(e)
            return []

    # -------------------------------------------------------------------------
    # DELETE VECTORS
    # -------------------------------------------------------------------------
    def delete_vectors(self, ids: List[str]) -> bool:
        try:
            logger.info(f"Deleting {len(ids)} vectors…")
            self.index.delete(ids=ids)
            logger.info("✔ Delete successful")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting vectors: {e}")
            logger.exception(e)
            return False

    # -------------------------------------------------------------------------
    # LISTING (NOT SUPPORTED IN SERVERLESS)
    # -------------------------------------------------------------------------
    def list_vectors(self):
        logger.info("Listing vectors not supported in serverless Pinecone")
        return []

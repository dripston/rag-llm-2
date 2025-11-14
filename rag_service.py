import logging
import uuid
from typing import List, Dict, Any, Optional
from embedding_service import SambaNovaEmbedding
from pinecone_service import PineconeService
from llm_service import SambaNovaLLM

# Configure logging
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("Initializing RAG service components...")
        self.embedding_service = SambaNovaEmbedding()
        logger.info("Embedding service initialized")
        self.vector_store = PineconeService()
        logger.info("Vector store initialized")
        self.llm_service = SambaNovaLLM()
        logger.info("LLM service initialized")
        logger.info("RAG service components initialized successfully")

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into chunks with overlap.

        Fixed edge-cases:
        - Prevents negative start index when overlap > chunk length.
        - Breaks if start doesn't advance to avoid infinite loops.
        - Adds a defensive max-iterations guard.
        """
        if not text:
            return []

        chunks: List[str] = []
        text_length = len(text)
        start = 0

        # Defensive guard to avoid accidental infinite loops
        max_iterations = max(10, (text_length // max(1, chunk_size - overlap)) + 10)
        iterations = 0

        while start < text_length and iterations < max_iterations:
            end = min(start + chunk_size, text_length)
            # append the chunk slice
            chunks.append(text[start:end])

            # compute next start taking care not to go negative
            next_start = max(0, end - overlap)

            # If next_start doesn't move forward, break to prevent infinite loop
            if next_start <= start:
                # try to move to end to finish
                if end >= text_length:
                    break
                # otherwise move to end (no overlap)
                start = end
            else:
                start = next_start

            iterations += 1

        if iterations >= max_iterations:
            logger.warning(
                "_chunk_text reached max_iterations; text_length=%d, chunk_size=%d, overlap=%d",
                text_length, chunk_size, overlap
            )

        return chunks

    def _combine_text_for_embedding(self, chunk: str, metadata: Dict[str, Any]) -> str:
        """
        Combine metadata + chunk content for embedding.
        """
        patient_name = metadata.get("patient_name", "")
        patient_id = metadata.get("patient_id", "")
        patient_age = metadata.get("age", "")
        patient_gender = metadata.get("gender", "")
        date_time = metadata.get("date_time", "")

        combined_text = (
            f"Patient Name: {patient_name}\n"
            f"Patient ID: {patient_id}\n"
            f"Age: {patient_age}\n"
            f"Gender: {patient_gender}\n"
            f"Date/Time: {date_time}\n\n"
            f"Medical Notes: {chunk}"
        )

        return combined_text

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            logger.info(f"Starting document addition. Content length: {len(content) if content else 0}")
            logger.info(f"Metadata: {metadata}")
            chunks = self._chunk_text(content)
            logger.info(f"Document chunked into {len(chunks)} chunks")
            vector_data = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                combined_text = self._combine_text_for_embedding(chunk, metadata or {})
                logger.info(f"Combined text length: {len(combined_text)}")

                embedding = self.embedding_service.get_single_embedding(combined_text)
                logger.info(f"Embedding generated. Dimension: {len(embedding) if embedding else 0}")

                doc_id = str(uuid.uuid4())
                logger.info(f"Generated document ID: {doc_id}")

                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["content"] = chunk
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                logger.info(f"Chunk metadata: {chunk_metadata}")

                vector_data.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
                logger.info(f"Chunk {i+1} processed successfully")

            if not vector_data:
                logger.info("No vectors to upsert (empty document after chunking). Skipping upsert.")
                return True

            logger.info(f"Upserting {len(vector_data)} vectors to Pinecone")
            result = self.vector_store.upsert_vectors(vector_data)
            logger.info(f"Vector upsert result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            logger.exception(e)
            return False

    def query(self, query_text: str, top_k: int = 3) -> str:
        try:
            query_embedding = self.embedding_service.get_single_embedding(query_text)

            similar_docs = self.vector_store.query_similar(query_embedding, top_k)

            context_parts = []
            for doc in similar_docs:
                meta = doc.get("metadata", {})
                if isinstance(meta, dict) and "content" in meta:
                    context_parts.append(meta["content"])

            context = "\n\n".join(context_parts)

            response = self.llm_service.generate_response_with_context(context, query_text)

            return response

        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return "Sorry, an error occurred."

    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        self.vector_store.delete_vectors([doc_id])
        return self.add_document(content, metadata)

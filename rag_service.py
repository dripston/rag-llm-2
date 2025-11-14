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
        Split text into chunks of specified size with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        logger.debug(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)} with length {len(chunk)}")
            
            # Move start position
            start = end - overlap
            if start >= text_length:
                break
                
            # If we're at the end and have a small chunk, break
            if end == text_length:
                break
                
        logger.debug(f"Text chunking completed. Total chunks: {len(chunks)}")
        return chunks
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a document to the RAG system with chunking
        
        Args:
            content: Document content
            metadata: Additional metadata for the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Adding document with content length: {len(content)}")
            # Chunk the content
            chunks = self._chunk_text(content)
            logger.info(f"Document chunked into {len(chunks)} chunks")
            
            # Process each chunk
            vector_data = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                # Generate embedding
                logger.debug("Generating embedding for chunk...")
                embedding = self.embedding_service.get_single_embedding(chunk)
                logger.debug(f"Embedding generated with dimension: {len(embedding)}")
                
                # Create document ID
                doc_id = str(uuid.uuid4())
                logger.debug(f"Generated document ID: {doc_id}")
                
                # Prepare metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["content"] = chunk
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                # Store in vector database
                vector_entry = {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                }
                vector_data.append(vector_entry)
                logger.debug(f"Chunk {i+1} processed successfully. Vector ID: {doc_id}")
            
            logger.info(f"Upserting {len(vector_data)} vectors to Pinecone...")
            logger.debug(f"Vector data sample: {vector_data[0] if vector_data else 'No vectors'}")
            result = self.vector_store.upsert_vectors(vector_data)
            logger.info(f"Document addition result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            logger.exception(e)
            return False
    
    def query(self, query_text: str, top_k: int = 3) -> str:
        """
        Query the RAG system and generate a response
        
        Args:
            query_text: User query
            top_k: Number of similar documents to retrieve
            
        Returns:
            Generated response
        """
        try:
            logger.info(f"Processing query: {query_text}")
            # Generate query embedding
            logger.debug("Generating embedding for query...")
            query_embedding = self.embedding_service.get_single_embedding(query_text)
            logger.debug(f"Query embedding generated with dimension: {len(query_embedding)}")
            
            # Retrieve similar documents
            logger.debug(f"Querying vector store for top {top_k} similar documents...")
            similar_docs = self.vector_store.query_similar(query_embedding, top_k)
            logger.info(f"Retrieved {len(similar_docs)} similar documents")
            
            # Log details about retrieved documents
            for i, doc in enumerate(similar_docs):
                logger.debug(f"Retrieved doc {i+1}: id={doc.get('id')}, score={doc.get('score')}")
            
            # Extract context from similar documents
            context = "\n\n".join([doc["metadata"]["content"] for doc in similar_docs])
            logger.debug(f"Context extracted with length: {len(context)}")
            
            # Generate response using LLM
            logger.debug("Generating response with LLM...")
            response = self.llm_service.generate_response_with_context(context, query_text)
            logger.info("Response generated successfully")
            
            return response
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            logger.exception(e)
            return "Sorry, I encountered an error while processing your query."
    
    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document in the RAG system
        
        Args:
            doc_id: Document ID
            content: Updated content
            metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating document with ID: {doc_id}")
        # Delete the old document
        self.vector_store.delete_vectors([doc_id])
        
        # Add the updated document
        result = self.add_document(content, metadata)
        logger.info(f"Document update result: {result}")
        return result
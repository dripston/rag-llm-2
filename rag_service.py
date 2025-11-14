import uuid
from typing import List, Dict, Any, Optional
from embedding_service import SambaNovaEmbedding
from pinecone_service import PineconeService
from llm_service import SambaNovaLLM

class RAGService:
    def __init__(self):
        self.embedding_service = SambaNovaEmbedding()
        self.vector_store = PineconeService()
        self.llm_service = SambaNovaLLM()
        
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
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position
            start = end - overlap
            if start >= text_length:
                break
                
            # If we're at the end and have a small chunk, break
            if end == text_length:
                break
                
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
            # Chunk the content
            chunks = self._chunk_text(content)
            
            # Process each chunk
            vector_data = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_service.get_single_embedding(chunk)
                
                # Create document ID
                doc_id = str(uuid.uuid4())
                
                # Prepare metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["content"] = chunk
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                # Store in vector database
                vector_data.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
            
            return self.vector_store.upsert_vectors(vector_data)
        except Exception as e:
            print(f"Error adding document: {e}")
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
            # Generate query embedding
            query_embedding = self.embedding_service.get_single_embedding(query_text)
            
            # Retrieve similar documents
            similar_docs = self.vector_store.query_similar(query_embedding, top_k)
            
            # Extract context from similar documents
            context = "\n\n".join([doc["metadata"]["content"] for doc in similar_docs])
            
            # Generate response using LLM
            response = self.llm_service.generate_response_with_context(context, query_text)
            
            return response
        except Exception as e:
            print(f"Error querying RAG system: {e}")
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
        # Delete the old document
        self.vector_store.delete_vectors([doc_id])
        
        # Add the updated document
        return self.add_document(content, metadata)
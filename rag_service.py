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
        
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a document to the RAG system
        
        Args:
            content: Document content
            metadata: Additional metadata for the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_service.get_single_embedding(content)
            
            # Create document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata["content"] = content
            
            # Store in vector database
            vector_data = [{
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }]
            
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
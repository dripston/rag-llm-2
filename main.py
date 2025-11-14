import logging
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from rag_service import RAGService
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical RAG Chatbot", description="A RAG chatbot for medical assistance")

# Initialize services
rag_service = None

@app.on_event("startup")
async def startup_event():
    global rag_service
    try:
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        logger.exception(e)

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    response: str

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

# Health check endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Medical RAG Chatbot is running"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Query endpoint accessed with query: {request.query}")
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    # Use default value if top_k is None
    top_k = request.top_k if request.top_k is not None else 3
    logger.info(f"Querying RAG service with top_k: {top_k}")
    response = rag_service.query(request.query, top_k)
    logger.info("Query processed successfully")
    return QueryResponse(response=response)

# Add document endpoint - users send content via POST request
@app.post("/documents")
async def add_document(request: DocumentRequest):
    logger.info("Document submission endpoint accessed")
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    logger.info(f"Adding document with content length: {len(request.content)}")
    success = rag_service.add_document(request.content, request.metadata)
    if success:
        logger.info("Document added successfully")
        return {"message": "Document chunked, embedded, and stored successfully"}
    else:
        logger.error("Failed to add document")
        raise HTTPException(status_code=500, detail="Failed to process document")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application server...")
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
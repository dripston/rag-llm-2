import logging
import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details
    logger.info(f"Incoming {request.method} request to {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Log request body for POST requests
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            try:
                # Try UTF-8 decoding first
                body_str = body.decode("utf-8")
                logger.info(f"Request body: {body_str}")
            except UnicodeDecodeError:
                # If UTF-8 fails, try with error handling
                try:
                    body_str = body.decode("utf-8", errors="replace")
                    logger.warning(f"Request body (with replaced errors): {body_str}")
                except Exception as e:
                    logger.error(f"Could not decode request body: {e}")
                    # Log raw bytes length for debugging
                    logger.info(f"Request body bytes length: {len(body)}")
            except Exception as e:
                logger.error(f"Could not decode request body: {e}")
                # Log raw bytes length for debugging
                logger.info(f"Request body bytes length: {len(body)}")
    
    response = await call_next(request)
    return response

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

# SOAP Notes format model
class SOAPNotesRequest(BaseModel):
    patient_id: str
    soap_notes: str
    date_time: str
    patient_name: str

# Health check endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Medical RAG Chatbot is running"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

# Debug endpoint to test Pinecone connection
@app.get("/debug/pinecone")
async def debug_pinecone():
    logger.info("Pinecone debug endpoint accessed")
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Test Pinecone connection by listing indexes
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes().names()
        logger.info(f"Pinecone indexes: {indexes}")
        
        # Test querying the index
        test_vector = [0.1] * 4096
        matches = rag_service.vector_store.query_similar(test_vector, top_k=1)
        
        logger.debug(f"Test query vector dimension: {len(test_vector)}")
        logger.debug(f"Matches: {matches}")
        logger.info(f"Test query returned {len(matches)} matches")
        
        return {
            "indexes": list(indexes),
            "test_query_matches": len(matches)
        }
    except Exception as e:
        logger.error(f"Error in Pinecone debug: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error in Pinecone debug: {str(e)}")

# Comprehensive debug endpoint
@app.post("/debug/test-document")
async def debug_test_document():
    logger.info("Debug test document endpoint accessed")
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Create a test document
        test_content = "This is a test document for debugging the RAG system. Patient has fever and cough."
        test_metadata = {"source": "debug_test", "timestamp": "2023-11-15"}
        
        logger.info("Adding test document...")
        success = rag_service.add_document(test_content, test_metadata)
        
        if not success:
            logger.error("Failed to add test document")
            raise HTTPException(status_code=500, detail="Failed to add test document")
        
        logger.info("Test document added successfully")
        
        # Wait a moment for indexing
        import time
        time.sleep(1)
        
        # Query for the test document
        logger.info("Querying for test document...")
        query_text = "What are the patient's symptoms?"
        response = rag_service.query(query_text, top_k=3)
        
        return {
            "document_added": True,
            "query_response": response
        }
    except Exception as e:
        logger.error(f"Error in debug test: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error in debug test: {str(e)}")

# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Query endpoint accessed with query: {request.query}")
    logger.info(f"Query parameters: top_k={request.top_k}")
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    # Use default value if top_k is None
    top_k = request.top_k if request.top_k is not None else 3
    logger.info(f"Querying RAG service with top_k: {top_k}")
    response = rag_service.query(request.query, top_k)
    logger.info("Query processed successfully")
    logger.info(f"Response length: {len(response) if response else 0}")
    return QueryResponse(response=response)

# Add SOAP notes endpoint - accepts specific SOAP notes format
@app.post("/soap-notes")
async def add_soap_notes(request: SOAPNotesRequest):
    logger.info("SOAP notes submission endpoint accessed")
    logger.info(f"Patient ID: {request.patient_id}")
    logger.info(f"Patient Name: {request.patient_name}")
    logger.info(f"Date/Time: {request.date_time}")
    logger.info(f"SOAP Notes length: {len(request.soap_notes)}")
    
    if rag_service is None:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    # Prepare metadata
    metadata = {
        "patient_id": request.patient_id,
        "patient_name": request.patient_name,
        "date_time": request.date_time,
        "source": "soap_notes"
    }
    
    logger.info(f"Adding SOAP notes with metadata: {metadata}")
    success = rag_service.add_document(request.soap_notes, metadata)
    if success:
        logger.info("SOAP notes added successfully")
        return {"message": "SOAP notes chunked, embedded, and stored successfully"}
    else:
        logger.error("Failed to add SOAP notes")
        raise HTTPException(status_code=500, detail="Failed to process SOAP notes")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application server...")
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
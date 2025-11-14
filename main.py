import logging
import os
import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    try:
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        logger.exception(e)
    yield
    logger.info("Application shutdown initiated. Waiting for background tasks to complete...")

    import asyncio
    try:
        await asyncio.wait_for(background_task_completion(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Background tasks did not complete within 10 seconds")
    logger.info("Application shutdown complete.")

async def background_task_completion():
    import asyncio
    await asyncio.sleep(5)

app = FastAPI(
    title="Medical RAG Chatbot",
    description="A RAG chatbot for medical assistance",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")

    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            try:
                body_str = body.decode("utf-8")
                logger.info(f"Request body: {body_str}")
            except Exception:
                logger.warning("Could not decode request body")

    response = await call_next(request)
    return response

rag_service = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    response: str

class SOAPNotesRequest(BaseModel):
    patient_id: str
    soap_notes: str
    date_time: str
    patient_name: str

@app.get("/")
async def root():
    return {"message": "Medical RAG Chatbot is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug/pinecone")
async def debug_pinecone():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes().names()

        test_vector = [0.1] * 4096
        matches = rag_service.vector_store.query_similar(test_vector, top_k=1)

        return {
            "indexes": list(indexes),
            "test_query_matches": len(matches)
        }

    except Exception as e:
        logger.error(f"Error in Pinecone debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/test-document")
async def debug_test_document():
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        test_content = "This is a test document for debugging the RAG system. Patient has fever and cough."
        test_metadata = {"source": "debug_test", "timestamp": "2023-11-15"}

        success = rag_service.add_document(test_content, test_metadata)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add test document")

        import time
        time.sleep(1)

        query_text = "What are the patient's symptoms?"
        response = rag_service.query(query_text, top_k=3)

        return {
            "document_added": True,
            "query_response": response
        }
    except Exception as e:
        logger.error(f"Error in debug test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    try:
        response = rag_service.query(request.query, top_k)
        logger.info("Query processed successfully")
        logger.info(f"Response length: {len(response) if response else 0}")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# ✅ FIXED BACKGROUND TASK — uses existing global rag_service
def process_soap_notes_in_background(soap_text: str, metadata: dict):
    global rag_service
    try:
        logger.info("Background: Processing SOAP notes...")

        if rag_service is None:
            logger.error("Background: RAG service not initialized")
            return

        success = rag_service.add_document(soap_text, metadata)

        if success:
            logger.info("Background: SOAP notes processed successfully")
        else:
            logger.error("Background: Failed to process SOAP notes")

    except Exception as e:
        logger.error(f"Background Error: {e}")
        logger.exception(e)

@app.post("/soap-notes")
async def add_soap_notes(request: SOAPNotesRequest, background_tasks: BackgroundTasks):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    metadata = {
        "patient_id": request.patient_id,
        "patient_name": request.patient_name,
        "date_time": request.date_time,
        "source": "soap_notes"
    }

    background_tasks.add_task(process_soap_notes_in_background, request.soap_notes, metadata)

    return {
        "message": "SOAP notes received and queued for processing",
        "status": "processing"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

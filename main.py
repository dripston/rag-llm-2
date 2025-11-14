from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from rag_service import RAGService
from webhook_processor import WebhookProcessor
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical RAG Chatbot", description="A RAG chatbot for medical assistance")

# Initialize services
rag_service = None
webhook_processor = None

@app.on_event("startup")
async def startup_event():
    global rag_service, webhook_processor
    try:
        rag_service = RAGService()
        webhook_processor = WebhookProcessor(rag_service)
        print("Services initialized successfully")
    except Exception as e:
        print(f"Error initializing services: {e}")

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    response: str

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

class WebhookData(BaseModel):
    event: str
    table: str
    data: dict

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Medical RAG Chatbot is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    response = rag_service.query(request.query, request.top_k)
    return QueryResponse(response=response)

# Add document endpoint
@app.post("/documents")
async def add_document(request: DocumentRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    success = rag_service.add_document(request.content, request.metadata)
    if success:
        return {"message": "Document added successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to add document")

# Webhook endpoint for PostgreSQL
@app.post("/webhook")
async def handle_webhook(request: Request):
    if webhook_processor is None:
        raise HTTPException(status_code=500, detail="Webhook processor not initialized")
        
    try:
        body = await request.json()
        # Process the webhook data and update RAG accordingly
        success = webhook_processor.process_webhook(body)
        
        if success:
            return {"message": "Webhook processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process webhook")
    except Exception as e:
        print(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to process webhook")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
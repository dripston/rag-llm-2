from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from rag_service import RAGService
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical RAG Chatbot", description="A RAG chatbot for medical assistance")

# Initialize services
rag_service = None

@app.on_event("startup")
async def startup_event():
    global rag_service
    try:
        rag_service = RAGService()
        print("RAG service initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG service: {e}")

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
    return {"message": "Medical RAG Chatbot is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# RAG Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    # Use default value if top_k is None
    top_k = request.top_k if request.top_k is not None else 3
    response = rag_service.query(request.query, top_k)
    return QueryResponse(response=response)

# Add document endpoint - users send content via POST request
@app.post("/documents")
async def add_document(request: DocumentRequest):
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    success = rag_service.add_document(request.content, request.metadata)
    if success:
        return {"message": "Document chunked, embedded, and stored successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to process document")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
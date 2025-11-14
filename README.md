# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical assistance, deployed on Render with Pinecone vector database integration.

## Features

- Uses SambaNova's E5-Mistral-7B-Instruct for embeddings (4096 dimensions)
- Uses Meta Llama 3.1 8B Instruct as the language model
- Pinecone vector database for document storage and retrieval
- Document chunking for better handling of large medical texts
- FastAPI backend for RESTful API endpoints

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Run the application: `python main.py`

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /query` - Query the RAG system
- `POST /documents` - Add medical documents to the RAG system

## Usage

### Adding Medical Documents
Send a POST request to `/documents` with the document content:
```json
{
  "content": "Medical document content here...",
  "metadata": {
    "source": "medical_record",
    "patient_id": "P12345"
  }
}
```

### Querying the System
Send a POST request to `/query` with your medical question:
```json
{
  "query": "What are the symptoms of pneumonia?",
  "top_k": 3
}
```

## Deployment

This application is configured for deployment on Render using the provided `render.yaml` and `Dockerfile`.

## Example Usage

See `example_usage.sh` (Linux/Mac) or `example_usage.bat` (Windows) for example curl commands.
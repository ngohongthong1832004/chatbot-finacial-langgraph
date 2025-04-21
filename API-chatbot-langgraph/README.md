# LangGraph RAG API

This is a FastAPI-based API for a RAG (Retrieval Augmented Generation) system that combines vector search, SQL database queries, and web search to answer questions using LangGraph.

## Features

- Vector-based document retrieval
- SQL database integration
- Web search capability
- Intelligent answer generation
- Flexible workflow using LangGraph

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create .env file:
   ```bash
   cp .env.example .env
   ```
   Then fill in your API keys and database credentials.

4. Run the application:
   ```bash
   uvicorn src.main:app --reload
   ```

## API Usage

Send POST requests to `/api/ask` with JSON body:
```json
{
    "text": "your question here"
}
```

Example with curl:
```bash
curl -X POST "http://localhost:8000/api/ask" -H "Content-Type: application/json" -d '{"text":"What is the capital of France?"}'
```

## API Documentation

After starting the server, access the Swagger UI documentation at:
```
http://localhost:8000/docs
```

## Project Structure

```
src/
├── api/            # API routes
├── config/         # Configuration
├── database/       # Database connection
├── models/         # Data models
├── services/       # Business logic
└── utils/          # Utilities
```

## Workflow

1. User submits a question
2. System tries to retrieve relevant documents from vector database
3. System grades the relevance of retrieved documents
4. Based on the grading:
   - If question is about databases → Execute SQL query
   - If documents are not relevant → Perform web search
   - If documents are relevant → Generate answer directly
5. Return final answer to user 
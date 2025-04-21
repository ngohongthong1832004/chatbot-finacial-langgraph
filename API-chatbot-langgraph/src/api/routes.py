from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.services.rag_service import process_query

router = APIRouter()

class Question(BaseModel):
    text: str = Field(..., description="The question to ask")

class Answer(BaseModel):
    answer: str = Field(..., description="The answer to the question")

@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Process a question through the RAG pipeline and return an answer
    """
    try:
        result = process_query(question.text)
        return {"answer": result["generation"]}
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
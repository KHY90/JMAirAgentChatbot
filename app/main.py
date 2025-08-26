from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.rag_chain import get_rag_chain

app = FastAPI()

# For production, you would want to re-enable CORS and API key verification
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import Depends, HTTPException, Header
# import os
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("API_KEY")
# ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# if not API_KEY:
#     raise ValueError("API_KEY must be set in the .env file.")

# async def verify_api_key(x_api_key: str = Header(..., description="API Key")):
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=403, detail="Invalid API Key")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(',')],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class Question(BaseModel):
    q: str = Field(..., min_length=1, max_length=200, description="User question")

rag_chain = get_rag_chain()

@app.post("/ask") #, dependencies=[Depends(verify_api_key)])
def ask_question(question: Question):
    """Returns an answer to the question"""
    try:
        answer = rag_chain.invoke(question.q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")

    return {
        "answer": answer,
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "good!"}
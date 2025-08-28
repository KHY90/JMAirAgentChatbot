from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from app.rag_chain import get_rag_chain
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not API_KEY:
    raise ValueError(".env 파일에 API_KEY를 설정해야 합니다.")

async def verify_api_key(x_api_key: str = Header(..., description="API 키")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="유효하지 않은 API 키")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    q: str = Field(..., min_length=1, max_length=200, description="사용자 질문")


rag_chain = get_rag_chain()


@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(question: Question):
    """질문에 대한 답변을 반환합니다"""
    try:
        answer = rag_chain.invoke(question.q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류 발생: {e}")

    return {
        "answer": answer,
    }


@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "정상!"}
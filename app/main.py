import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.utils import load_md_file
from app.embedder import embed_chunks, model as embedder_model
from app.retriever import retrieve_relevant_chunk
from app.llm_infer import generate_answer

load_dotenv()

# 보안 관련 환경 변수 로드
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not API_KEY:
    raise ValueError("API_KEY가 .env 파일에 설정되지 않았습니다.")

async def verify_api_key(x_api_key: str = Header(..., description="API Key")):
    """API Key 검증"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="유효하지 않은 API Key 입니다")


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터 로딩, 청크
chunks = load_md_file("documents/infomation.md", chunk_size=100, overlap=20)
chunk_embeddings = embed_chunks(chunks)

class Question(BaseModel):
    q: str = Field(..., min_length=1, max_length=200, description="사용자 질문")

@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(question: Question):
    """질문에 대한 답변을 반환"""

    try:
        relevant_chunk = retrieve_relevant_chunk(
            question.q, chunks, chunk_embeddings, embedder_model
        )
        answer = generate_answer(relevant_chunk, question.q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류가 발생했습니다: {e}")

    return {
        "answer": answer,
    }


@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "good!"}

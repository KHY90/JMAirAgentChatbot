import os
from fastapi import FastAPI, Query, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.utils import load_md_file
from app.embedder import embed_chunks, model as embedder_model
from app.retriever import retrieve_relevant_chunk
from app.llm_infer import generate_answer

load_dotenv()

# Security API Key 설정
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY가 .env 파일에 설정되지 않았습니다.")

async def verify_api_key(x_api_key: str = Header(..., description="API Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터 로딩, 청크
chunks = load_md_file("documents/infomation.md", chunk_size=100, overlap=20)
chunk_embeddings = embed_chunks(chunks)

@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(q: str = Query(..., description="사용자 질문")):
    relevant_chunk = retrieve_relevant_chunk(q, chunks, chunk_embeddings, embedder_model)
    answer = generate_answer(relevant_chunk, q)
    return {
        "question": q,
        "context": relevant_chunk,
        "answer": answer
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "good!"}  
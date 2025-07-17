from fastapi import FastAPI, Query
from app.utils import load_md_file
from app.embedder import embed_chunks, model as embedder_model
from app.retriever import retrieve_relevant_chunk
from app.llm_infer import generate_answer

app = FastAPI()

# 초기 로딩
chunks = load_md_file("documents/infomation.md")
chunk_embeddings = embed_chunks(chunks)

@app.get("/ask")
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
import os
import json
import pickle
from typing import List, Tuple
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema import Document
from app.supabase_client import supabase
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import torch

# ============================================
# 1. 한국어 토크나이저
# ============================================
kiwi = Kiwi()

def tokenize_korean(text: str) -> List[str]:
    """한국어 형태소 분석 기반 토큰화"""
    tokens = kiwi.tokenize(text)
    meaningful_pos = {'NNG', 'NNP', 'VV', 'VA', 'MAG', 'SL', 'SN'}
    return [token.form for token in tokens if token.tag in meaningful_pos]


# ============================================
# 2. BM25 검색기 클래스
# ============================================
class BM25Retriever:
    def __init__(self, index_path: str = "data/bm25_index.pkl",
                 corpus_path: str = "data/bm25_corpus.json"):
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.corpus_texts = data["corpus_texts"]
            self.tokenized_corpus = data["tokenized_corpus"]

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """BM25 검색 수행"""
        tokenized_query = tokenize_korean(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 결과
        top_indices = scores.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.corpus_texts[idx], float(scores[idx])))
        return results


# ============================================
# 3. 하이브리드 검색 (RRF 결합)
# ============================================
class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever: BM25Retriever,
                 k: int = 5, rrf_k: int = 60):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.k = k
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(self,
                               vector_results: List[Document],
                               bm25_results: List[Tuple[str, float]]) -> List[str]:
        """RRF로 두 검색 결과 결합"""
        doc_scores = {}

        # Vector 검색 결과에 RRF 점수 부여
        for rank, doc in enumerate(vector_results):
            content = doc.page_content
            rrf_score = 1 / (self.rrf_k + rank + 1)
            doc_scores[content] = doc_scores.get(content, 0) + rrf_score

        # BM25 검색 결과에 RRF 점수 부여
        for rank, (content, _) in enumerate(bm25_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            doc_scores[content] = doc_scores.get(content, 0) + rrf_score

        # 점수순 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:self.k]]

    def search(self, query: str) -> List[str]:
        """하이브리드 검색 수행"""
        # Vector 검색
        vector_docs = self.vector_retriever.invoke(query)

        # BM25 검색
        bm25_results = self.bm25_retriever.search(query, k=self.k)

        # RRF 결합
        combined = self.reciprocal_rank_fusion(vector_docs, bm25_results)
        return combined


# ============================================
# 4. 한국어 프롬프트 템플릿
# ============================================
KOREAN_RAG_PROMPT = """당신은 에어컨 설치 전문 상담 챗봇입니다.
아래 제공된 참고 정보를 바탕으로 고객의 질문에 정확하고 친절하게 답변해주세요.

### 참고 정보:
{context}

### 고객 질문:
{question}

### 답변 지침:
1. 참고 정보에 있는 내용만 사용하여 답변하세요.
2. 가격 정보는 정확하게 전달하세요.
3. 참고 정보에 없는 내용은 "해당 정보는 확인이 어렵습니다"라고 답변하세요.
4. 존댓말을 사용하고 친절하게 응답하세요.

### 답변:
"""


# ============================================
# 5. RAG 체인 생성
# ============================================
def get_rag_chain():
    # ----- 임베딩 모델 -----
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ----- Vector Store -----
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # ----- BM25 Retriever -----
    bm25_retriever = BM25Retriever()

    # ----- 하이브리드 검색기 -----
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        k=3,       # 최종 반환 문서 수
        rrf_k=60   # RRF 파라미터
    )

    # ----- Gemma-2-2b-it 모델 로드 -----
    model_id = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Transformers 파이프라인
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # LangChain 래퍼
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ----- 프롬프트 -----
    prompt = PromptTemplate(
        template=KOREAN_RAG_PROMPT,
        input_variables=["context", "question"]
    )

    # ----- 헬퍼 함수 -----
    def retrieve_and_format(question: str) -> dict:
        """검색 및 컨텍스트 포맷팅"""
        docs = hybrid_retriever.search(question)
        context = "\n\n".join(docs)
        return {"context": context, "question": question}

    def extract_answer(llm_output: str) -> str:
        """LLM 출력에서 답변 추출"""
        if "### 답변:" in llm_output:
            answer = llm_output.split("### 답변:")[-1].strip()
            return answer
        return llm_output.strip()

    # ----- RAG 체인 구성 -----
    rag_chain = (
        RunnableLambda(retrieve_and_format)
        | prompt
        | llm
        | RunnableLambda(extract_answer)
    )

    return rag_chain


if __name__ == "__main__":
    # 테스트
    rag_chain = get_rag_chain()

    test_questions = [
        "벽걸이 에어컨 설치 비용을 알려줘",
        "설치 기사 등록은 어떻게 해?",
        "추가 비용이 발생하는 경우는?",
    ]

    for q in test_questions:
        print(f"\n질문: {q}")
        answer = rag_chain.invoke(q)
        print(f"답변: {answer}")

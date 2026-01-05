import json
import pickle
from typing import List, Tuple
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
        print(f"[BM25] 쿼리 토큰: {tokenized_query}")
        print(f"[BM25] corpus 크기: {len(self.corpus_texts)}")

        scores = self.bm25.get_scores(tokenized_query)
        print(f"[BM25] 점수: {scores}")

        # 상위 k개 결과 (점수 관계없이 반환)
        top_indices = scores.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            results.append((self.corpus_texts[idx], float(scores[idx])))
        return results


# ============================================
# 3. 한국어 프롬프트 템플릿 (간소화)
# ============================================
KOREAN_RAG_PROMPT = """에어컨 설치 상담 챗봇입니다. 참고 정보를 바탕으로 답변하세요.

참고 정보:
{context}

질문: {question}

답변 규칙:
- 질문한 에어컨 타입의 정보만 답변
- 가격은 "약 ~원"으로 안내하고 "정확한 금액은 현장 견적 후 확정됩니다" 추가
- 존댓말 사용

답변:"""


# ============================================
# 5. RAG 체인 생성
# ============================================
def get_rag_chain():
    # ----- BM25 Retriever (Supabase 없이 동작) -----
    bm25_retriever = BM25Retriever()

    # ----- EXAONE-3.5-2.4B-Instruct 모델 로드 (한국어 특화) -----
    model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Transformers 파이프라인
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
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
        """검색 및 컨텍스트 포맷팅 (BM25 사용)"""
        results = bm25_retriever.search(question, k=3)
        docs = [content for content, score in results]
        context = "\n\n".join(docs) if docs else "관련 정보를 찾을 수 없습니다."
        # 디버깅 로그
        print(f"[검색] 질문: {question}")
        print(f"[검색] 결과 수: {len(results)}")
        for i, (content, score) in enumerate(results):
            print(f"[검색] {i+1}. 점수={score:.2f}, 내용={content[:100]}...")
        return {"context": context, "question": question}

    def extract_answer(llm_output: str) -> str:
        """LLM 출력에서 답변만 추출 (프롬프트 반복 제거)"""
        answer = llm_output

        # "답변:" 이후 내용만 추출
        if "답변:" in answer:
            answer = answer.split("답변:")[-1]

        # 불필요한 후속 내용 제거 (프롬프트 반복 방지)
        stop_markers = ["질문:", "###", "참고 정보:", "답변 규칙:"]
        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker)[0]

        return answer.strip()

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

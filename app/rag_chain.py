from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import SupabaseVectorStore
from app.supabase_client import supabase
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_rag_chain():
    # 허깅페이스 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 수퍼베이스 백터스토어 초기화
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )

    # 검색기 생성
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})

    # 허깅페이스 시퀀스-투-시퀀스 모델 로드
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype="auto")

    # Transformers 파이프라인 생성
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512
    )

    # LangChain 래퍼 생성
    llm = HuggingFacePipeline(
        pipeline=hf_pipeline,
        pipeline_kwargs={"return_full_text": False}
    )

    def format_docs(docs):
        """Document 객체들을 텍스트로 변환"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_korean_response(inputs):
        """한국어 응답을 직접 생성"""
        context = inputs["context"]
        question = inputs["question"]
        
        # print("---\n--- 디버그 정보 ---")
        # print(f"[검색된 컨텍스트]: {context}")
        # print(f"[사용자 질문]: {question}")
        # print("---\n--- 디버그 종료 ---")
        
        lines = context.split('\n')
        
        # 1. 회원 탈퇴 관련 질문
        if any(keyword in question for keyword in ["탈퇴", "회원탈퇴", "계정삭제", "회원 탈퇴"]):
            if "회원 탈퇴" in context:
                response = "회원 탈퇴 방법을 안내해드립니다:\n\n"
                for line in lines:
                    if line.strip() and ("마이페이지" in line or "나의 정보" in line or "회원탈퇴 버튼" in line):
                        response += f"- {line.strip()}\n"
                return response if len(response.split('\n')) > 2 else "회원 탈퇴는 마이페이지 > 나의 정보 > 회원탈퇴 버튼을 클릭하시면 됩니다."
        
        # 2. 설치 기사 등록 관련 질문
        if any(keyword in question for keyword in ["기사", "등록", "신청", "설치 기사", "기사 등록"]):
            if "설치 기사" in context:
                response = "설치 기사 등록 방법을 안내해드립니다:\n\n"
                for line in lines:
                    if line.strip() and ("회원 가입" in line or "마이페이지" in line or "설치 기사 신청" in line or "관리자" in line or "등록 진행" in line):
                        response += f"- {line.strip()}\n"
                return response if len(response.split('\n')) > 2 else "설치 기사 등록은 회원가입 후 마이페이지에서 신청 가능합니다."
        
        # 3. 추가 비용 관련 질문
        if any(keyword in question for keyword in ["추가", "추가비용", "별도", "용접", "타공", "사다리", "펌프"]):
            if "추가 비용" in context:
                response = "추가 비용이 발생하는 경우를 안내해드립니다:\n\n"
                for line in lines:
                    if line.strip() and ("앵글 재설치" in line or "배수 펌프" in line or "용접비" in line or "타공" in line or "사다리차" in line):
                        response += f"- {line.strip()}\n"
                return response if len(response.split('\n')) > 2 else "추가 비용은 현장 상황에 따라 발생할 수 있습니다."
        
        # 4. 특정 에어컨 타입별 설치 비용 질문
        if "벽걸이" in question and ("설치" in question or "비용" in question or "가격" in question):
            if "벽걸이형" in context:
                response = "벽걸이형 에어컨 설치 비용은 다음과 같습니다:\n\n"
                for line in lines:
                    if line.strip() and ("기본 설치비용" in line or "철거" in line or "배관" in line or "가스보충비" in line or "앵글설치" in line):
                        response += f"- {line.strip()}\n"
                return response
        
        elif "스탠드" in question and ("설치" in question or "비용" in question or "가격" in question):
            if "스탠드형" in context or "스탠드 중대형용" in context:
                response = "스탠드형 에어컨 설치 비용은 다음과 같습니다:\n\n"
                for line in lines:
                    if line.strip() and ("기본 설치비용" in line or "철거" in line or "배관" in line or "가스보충비" in line or "앵글설치" in line):
                        response += f"- {line.strip()}\n"
                return response
        
        elif "천장" in question and ("설치" in question or "비용" in question or "가격" in question):
            if "천장형" in context:
                response = "천장형 에어컨 설치 비용은 다음과 같습니다:\n\n"
                for line in lines:
                    if line.strip() and ("기본 설치비용" in line or "철거" in line or "배관" in line or "가스보충비" in line or "앵글설치" in line):
                        response += f"- {line.strip()}\n"
                return response
        
        elif "투인원" in question and ("설치" in question or "비용" in question or "가격" in question):
            if "투인원" in context:
                response = "투인원 에어컨 설치 비용은 다음과 같습니다:\n\n"
                for line in lines:
                    if line.strip() and ("기본 설치비용" in line or "철거" in line or "배관" in line or "가스보충비" in line or "앵글설치" in line):
                        response += f"- {line.strip()}\n"
                return response
        
        # 5. 일반적인 비용/가격 질문 (구체적인 타입이 명시되지 않은 경우)
        elif any(keyword in question for keyword in ["비용", "가격", "설치"]):
            relevant_lines = []
            for line in lines:
                if line.strip() and any(keyword in line for keyword in ["기본 설치비용", "철거", "배관", "가스보충비", "앵글설치"]):
                    relevant_lines.append(f"- {line.strip()}")
            
            if relevant_lines:
                return "에어컨 설치 비용 정보를 안내해드립니다:\n\n" + "\n".join(relevant_lines)
        
        # 6. 컨텍스트에 관련 정보가 있는 경우 일반적인 응답
        if context.strip():
            # 컨텍스트의 핵심 정보를 간단히 정리해서 제공
            key_info = []
            for line in lines:
                if line.strip() and len(line.strip()) > 5:  # 너무 짧은 라인 제외
                    key_info.append(f"- {line.strip()}")
            
            if key_info:
                return "관련 정보를 안내해드립니다:\n\n" + "\n".join(key_info[:5])  # 최대 5개 항목만
        
        return "죄송합니다, 해당 정보는 제가 가지고 있지 않습니다."

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(create_korean_response)
    )

    return rag_chain


if __name__ == "__main__":
    # For testing purposes
    rag_chain = get_rag_chain()
    question = "벽걸이 에어컨 설치 비용을 알려줘"
    answer = rag_chain.invoke(question)
    # print(f"\n--- 최종 답변 ---")
    # print(f"질문: {question}")
    # print(f"답변: {answer}")


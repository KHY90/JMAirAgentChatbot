def generate_answer(context: str, query: str) -> str:
    """
    주어진 컨텍스트와 질문을 사용하여 LLM으로부터 답변을 생성합니다.
    (실제 LLM 호출 대신, 형식을 맞춘 프롬프트를 반환합니다)
    """
    prompt = f"""
    당신은 주어진 컨텍스트를 바탕으로 사용자의 질문에 답변하는 AI 어시스턴트입니다.
    컨텍스트 정보만을 사용하여, 만약 정보가 부족하다면 "정보가 부족하여 답변할 수 없습니다."라고 답변하세요.

    컨텍스트:
    {context}

    질문:
    {query}

    답변:
    """
    # 실제 LLM 호출이 여기서 이루어집니다.
    # 예: response = llm.generate(prompt)
    # 지금은 시뮬레이션을 위해 완성된 프롬프트를 반환합니다.
    # 실제 구현에서는 아래 줄을 LLM 호출 코드로 대체해야 합니다.
    return f"'{query}'에 대한 답변입니다: {context}"

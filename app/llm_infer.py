def generate_answer(context: str, query: str) -> str:
    """
    주어진 컨텍스트와 질문을 사용하여 LLM으로부터 답변을 생성합니다.
    """
    """
    간단한 예제이므로 복잡한 LLM 추론 대신 검색된 컨텍스트를 그대로
    반환한다. 만약 컨텍스트가 없으면 정보 부족 메시지를 전달한다.
    """

    if not context:
        return "정보가 부족하여 답변할 수 없습니다."

    return context

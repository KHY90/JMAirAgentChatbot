from sentence_transformers import SentenceTransformer


# 임베딩 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# 청크 목록을 받아 임베딩 배열을 반환
def embed_chunks(chunks: list[str]):
    if not chunks:
        raise ValueError("임베딩할 청크가 없습니다.")

    try:
        return model.encode(chunks)
    except Exception as e:
        raise RuntimeError(f"임베딩 생성 중 오류가 발생했습니다: {e}")

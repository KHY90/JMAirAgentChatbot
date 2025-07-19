from sklearn.metrics.pairwise import cosine_similarity

# 주어진 질문과 문서 청크들 중 가장 연관성이 높은 청크를 반환합니다.
def retrieve_relevant_chunk(query, chunks, chunk_embeddings, embedder):
    if not chunks:
        raise ValueError("검색할 문서가 없습니다.")

    try:
        query_embedding = embedder.encode([query])
        scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        best_idx = scores.argmax()
    except Exception as e:
        raise RuntimeError(f"검색 과정에서 오류가 발생했습니다: {e}")

    return chunks[best_idx]

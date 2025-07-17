from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunk(query, chunks, chunk_embeddings, embedder):
    query_embedding = embedder.encode([query])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    best_idx = scores.argmax()
    return chunks[best_idx]

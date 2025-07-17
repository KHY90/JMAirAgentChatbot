from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_chunks(chunks: list[str]):
    """Return embeddings for each chunk as NumPy arrays."""
    return model.encode(chunks)

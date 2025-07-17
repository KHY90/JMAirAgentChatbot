from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_chunks(chunks: list[str]):
    return model.encode(chunks, convert_to_tensor=True)

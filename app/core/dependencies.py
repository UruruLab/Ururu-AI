from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("jhgan/ko-sbert-multitask")
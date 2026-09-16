import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str):
        self._modelName= model_name
        self._embedder= SentenceTransformer(model_name)

    def encode(self, text: str | list[str]) -> np.ndarray:
        embeddings= self._embedder.encode(text, convert_to_numpy=True)
        return embeddings
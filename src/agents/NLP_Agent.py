'''
Steps to Agent

1. Recieve question and Visual breakdown (either class or separated by features of a plant)

2. preprocess features of plant and get returned 
'''

"""Are we able to leverage the local host model"""
"""Add in crowd sourced data to the model"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from src.util.logger import Logger
from src.data_preprocessing.TextEmbedder import TextEmbedder, EmbeddingRecord

@dataclass
class TextAnswer:
    answer: str
    confidence: float
    top_k: List[Tuple[str, float]]
    def alternatives(self, threshold: float = 0.0) -> List[str]:
        return [text for text, score in self.top_k[1:] if score >= threshold]

class AnswerIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = None
        self.records = []
        self._columns = {}

    def __len__(self):
        return len(self.records)

    def add_records(self, records):
        vector = np.stack([record.embedding for record in records]).astype(np.float32)
        vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        self.vectors = vector if self.vectors is None else np.vstack([self.vectors, vector])
        self.records.extend(records)
        self._columns = {}

    def column(self, key):
        if key not in self._columns:
            self._columns[key] = np.array([r.metadata.get(key) for r in self.records], dtype=object)
        return self._columns[key]

    def add_text(self, texts, metadata):
        texts = list(texts)
        embeddings = self.embedder.encode(texts)
        self.add_records([EmbeddingRecord(text, embedding, dict(meta)) for text, embedding, meta in zip(texts, embeddings, metadata)])

    def search(self, query, k=3):
        query_vec = self.embedder.encode(query, normalize = True)
        scores = self.vectors @ query_vec

        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return []

        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.records[i], float(scores[i])) for i in top]


class AnswerRanker:
    def __init__(self, index):
        self.index = index

    @staticmethod
    def build_query(question, visual_prediction):
        h = visual_prediction.classification_heads
        return (f"{question} Crop: {h['crop'][0][0]}. Disease: {h['disease'][0][0]}. "
            + f"Severity: {h['severity'][0][0]}.")


    def predict(self, question, visual_prediction, k=3):
        query = self.build_query(question, visual_prediction)
        ranked = self.index.search(query, k=k)

        if not ranked:
            Logger.warning("[predict] no candidates matched")
            return None
        top_k = [(record.text, score) for record, score in ranked]
        return TextAnswer(top_k[0][0], top_k[0][1], top_k)

def build_answer_index(embedder, data):
    rows = data[["answer", "crop", "disease"]].drop_duplicates()
    metadata = [{"source": "qa", "crop": crop, "disease": disease}
                for crop, disease in zip(rows["crop"], rows["disease"])]

    index = AnswerIndex(embedder)
    index.add_text(rows["answer"].tolist(), metadata)
    return index


def add_usda_records(index, records):
    tagged = [EmbeddingRecord(r.text, r.embedding, dict(r.metadata, source="usda")) for r in records]
    index.add_records(tagged)
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

    def mask(self, **filters):
        keep = np.ones(len(self), dtype=bool)
        for key, value in filters.items():
            keep &= self.column(key) == value
        return keep
    def search(self, query, k=3, mask=None):
        query_vec = self.embedder.encode(query, normalize = True)
        scores = self.vectors @ query_vec

        if mask is not None:
            scores = np.where(mask, scores, -np.inf) #if mask is not true set to negative infinity

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
        h = visual_prediction.heads
        return (f"{question} Crop: {h['crop'][0][0]}. Disease: {h['disease'][0][0]}. "
            + f"Severity: {h['severity'][0][0]}.")


    def predict(self, question, visual_prediction, k=3):
        query = self.build_query(question, visual_prediction)
        ranked = self.index.search(query, k=k, mask=candidate_mask(self.index, visual_prediction))

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

def candidate_mask(index, visual_prediction, minimum=5):
    crop = visual_prediction.heads["crop"][0][0]
    disease = visual_prediction.heads["disease"][0][0]

    for filters in ({"crop": crop, "disease": disease}, {"crop": crop}, {}):
        mask = index.mask(**filters)
        if mask.sum() >= minimum:
            return mask
    return np.ones(len(index), dtype=bool)

if __name__ == "__main__":
    import pandas as pd
    from src.data_preprocessing.TextEmbedder import TextEmbedder
    from src.agents.Visual_Agent import VisualPrediction

    #Some random training utterances
    frame = pd.DataFrame({
        "question_text": ["what disease shown here", "what crop", "how severe infection",
                          "what symptoms present leaf", "leaf healthy diseased",
                          "what conditions led infection", "how differentiate mosaic virus",
                          "what happen leaf without treatment"] * 8,
        "answer": ["gourd anthracnose severe extensive damage visible leaf",
                   "bottle gourd lagenaria siceraria cucurbitaceae family",
                   "severe infection tissue damage active 1 2 weeks",
                   "circular dark brown lesions water soaked spots",
                   "leaf diseased showing anthracnose lesions",
                   "prolonged leaf wetness high humidity warm temperatures",
                   "downy mildew lesions angular bounded veins anthracnose circular",
                   "lesions coalesce vine dieback fruit rot within 2 3 weeks"] * 8,
        "crop": ["bottle gourd"] * 64,
        "disease": ["gourd anthracnose"] * 64,
        "severity": ["SEVERE"] * 64,
        "question_category": ["Specific Disease Identification"] * 64,
    })
    frame["answer_raw"] = [
        "This is gourd anthracnose, displaying severe symptoms including extensive damage.",
        "This is a bottle gourd (Lagenaria siceraria), in the Cucurbitaceae family.",
        "This is a severe infection. The tissue damage shows it has been active 1-2 weeks.",
        "There are circular, dark brown lesions and small, water-soaked spots.",
        "This leaf is diseased, showing anthracnose lesions.",
        "It is favored by prolonged leaf wetness, high humidity, and warm temperatures.",
        "Downy mildew lesions are angular and bounded by veins; anthracnose lesions are circular.",
        "Lesions will coalesce, causing vine dieback and fruit rot within 2-3 weeks.",
    ] * 8

    embedder = TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    index = build_answer_index(embedder, frame)
    ranker = AnswerRanker(index)

    heads = {
        "crop":     [("bottle gourd", 0.91)],
        "disease":  [("gourd anthracnose", 0.84)],
        "category": [("disease", 0.97)],
        "severity": [("SEVERE", 0.72)],
    }
    visual = VisualPrediction("bottle gourd", 0.91, heads["crop"], heads)

    print("index size: ", len(index))

    result = ranker.predict("what disease shown here", visual)

    print("\nanswer: ", result.answer)
    print("confidence: ", round(result.confidence, 3))
    print("top_k: ", [(t[:40], round(s, 3)) for t, s in result.top_k])
    print("alternatives: ", [a[:40] for a in result.alternatives()])

#top_k:  [('This is gourd anthracnose, displaying se', 0.706), ('This leaf is diseased, showing anthracno', 0.625), ('This is a bottle gourd (Lagenaria sicera', 0.568)]
#alternatives:  ['This leaf is diseased, showing anthracno', 'This is a bottle gourd (Lagenaria sicera']
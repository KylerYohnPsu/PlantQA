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


@dataclass
class TextAnswer:
    answer: str
    confidence: float
    top_k: List[Tuple[str, float]]

    def alternatives(self, threshold: float = 0.0) -> List[str]:
        return [text for text, score in self.top_k[1:] if score >= threshold]


class AnswerRanker:
    def __init__(self, embedder):
        self.embedder = embedder

    @staticmethod
    def build_query(question, visual_prediction):
        h = visual_prediction.heads
        return (f"{question} Crop: {h['crop'][0][0]}. Disease: {h['disease'][0][0]}. "
            + f"Severity: {h['severity'][0][0]}.")

    @staticmethod
    def candidate_pools(data, max_per_pool=64):
        display = "answer_raw" if "answer_raw" in data.columns else "answer"
        pools = {}
        for key, group in data.groupby(["question_category", "crop", "disease", "severity"]):
            pairs = dict(zip(group["answer"], group[display]))
            pools[key] = list(pairs.items())[:max_per_pool]
        return pools


    def predict(self, question, visual_prediction, candidates, k=3):
        if not candidates:
            Logger.error("[predict] No candidate answers to rank")
            return None

        shown = [c[1] for c in candidates]

        query_vec = self.embedder.encode(self.build_query(question, visual_prediction))
        answer_vecs = self.embedder.encode(shown)

        # normalize so the dot product is cosine similarity
        query_vec = query_vec / np.linalg.norm(query_vec)
        answer_vecs = answer_vecs / np.linalg.norm(answer_vecs, axis=1, keepdims=True)

        scores = answer_vecs @ query_vec
        top = np.argsort(scores)[::-1][:k]
        ranked = [(shown[i], float(scores[i])) for i in top]

        return TextAnswer(ranked[0][0], ranked[0][1], ranked)

    def save(self, path):
        self.model.save(path)



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

    ranker = AnswerRanker(TextEmbedder("sentence-transformers/all-MiniLM-L6-v2"))

    heads = {
        "crop":     [("bottle gourd", 0.91)],
        "disease":  [("gourd anthracnose", 0.84)],
        "category": [("disease", 0.97)],
        "severity": [("SEVERE", 0.72)],
    }
    visual = VisualPrediction("bottle gourd", 0.91, heads["crop"], heads)

    pools = AnswerRanker.candidate_pools(frame)
    candidates = pools[("Specific Disease Identification", "bottle gourd",
                        "gourd anthracnose", "SEVERE")]

    result = ranker.predict("what disease shown here", visual, candidates)

    print("\nanswer: ", result.answer)
    print("confidence: ", round(result.confidence, 3))
    print("top_k: ", [(t[:40], round(s, 3)) for t, s in result.top_k])
    print("alternatives: ", [a[:40] for a in result.alternatives()])

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
import tensorflow as tf
import keras
from keras import layers
from src.util.logger import Logger


@dataclass
class TextAnswer:
    answer: str
    confidence: float
    top_k: List[Tuple[str, float]]

    def alternatives(self, threshold: float = 0.0) -> List[str]:
        return [text for text, score in self.top_k[1:] if score >= threshold]


class AnswerRanker:
    def __init__(self, vocab_size=30522, dim=256, temperature=0.05):
        self.vocab_size = vocab_size
        self.dim = dim
        self.temperature = temperature
        self.encoder = None
        self.model = None

    @staticmethod
    def build_query(question, visual_prediction):
        facts = " ".join(preds[0][0] for preds in visual_prediction.heads.values())
        return f"{question} [SEP] {facts}"

    @staticmethod
    def candidate_pools(data, max_per_pool=64):
        display = "answer_raw" if "answer_raw" in data.columns else "answer"
        pools = {}
        for key, group in data.groupby(["question_category", "crop", "disease", "severity"]):
            pairs = dict(zip(group["answer"], group[display]))
            pools[key] = list(pairs.items())[:max_per_pool]
        return pools

    @staticmethod
    def make_dataset(data, tokenizer, batch_size=256, training=False):
        queries = [f"{q} [SEP] {c} {d} {s}" for q, c, d, s in
                   zip(data["question_text"], data["crop"], data["disease"], data["severity"])]

        question_ids, _ = tokenizer.encode_text(queries)
        answer_ids, _ = tokenizer.encode_text(data["answer"])

        inputs = {"question_ids": question_ids, "answer_ids": answer_ids}
        labels = np.zeros(len(data), dtype="int32")

        data_set = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if training:
            data_set = data_set.shuffle(8192)
        return data_set.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def build(self):
        ids = keras.Input(shape=(None,), dtype="int32")
        x = layers.Embedding(self.vocab_size, self.dim, mask_zero=True)(ids)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.UnitNormalization()(x)
        self.encoder = keras.Model(ids, x, name="encoder")

        question_ids = keras.Input(shape=(None,), dtype="int32", name="question_ids")
        answer_ids = keras.Input(shape=(None,), dtype="int32", name="answer_ids")

        scores = layers.Lambda(
            lambda t: tf.matmul(t[0], t[1], transpose_b=True) / self.temperature, # Cosine similarity scorer in logits
            output_shape=lambda s: (s[0][0], s[0][0]),
            name="scores",
        )([self.encoder(question_ids), self.encoder(answer_ids)])

        self.model = keras.Model([question_ids, answer_ids], scores, name="plantqa_ranker")
        return self.model

    def compile(self, lr=1e-3):
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=in_batch_loss,
            metrics=[recall_at_1],
        )

    def fit(self, train_ds, val_ds, epochs=20):
        calls = [
            keras.callbacks.EarlyStopping("val_recall_at_1", patience=4, mode="max",
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint("best_text.keras", monitor="val_recall_at_1",
                                            mode="max", save_best_only=True),
        ]
        return self.model.fit(train_ds, validation_data=val_ds,
                              epochs=epochs, callbacks=calls)

    def predict(self, question, visual_prediction, candidates, tokenizer, k=3):
        if not candidates:
            Logger.error("[predict] No candidate answers to rank")
            return None

        scored = [c[0] for c in candidates]
        shown = [c[1] for c in candidates]

        query_ids, _ = tokenizer.encode_text(self.build_query(question, visual_prediction))
        answer_ids, _ = tokenizer.encode_text(scored)

        query_vec = self.encoder.predict(query_ids, verbose=0)
        answer_vecs = self.encoder.predict(answer_ids, verbose=0)

        scores = (answer_vecs @ query_vec.T).ravel()
        top = np.argsort(scores)[::-1][:k]
        ranked = [(shown[i], float(scores[i])) for i in top]

        return TextAnswer(ranked[0][0], ranked[0][1], ranked)

    def save(self, path):
        self.model.save(path)


def in_batch_loss(y_true, y_pred):
    targets = tf.range(tf.shape(y_pred)[0])
    return keras.losses.sparse_categorical_crossentropy(targets, y_pred, from_logits=True)


def recall_at_1(y_true, y_pred):
    targets = tf.range(tf.shape(y_pred)[0])
    predicted = tf.cast(tf.argmax(y_pred, axis=-1), targets.dtype)
    return tf.cast(tf.equal(predicted, targets), "float32")


if __name__ == "__main__":
    import pandas as pd
    from src.data_preprocessing.TextTokenizer import TextTokenizer
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

    #using distilbert as an example
    tokenizer = TextTokenizer("distilbert-base-uncased", 64)

    ranker = AnswerRanker()
    ranker.build()
    ranker.compile()
    ranker.model.summary()

    train_ds = AnswerRanker.make_dataset(frame, tokenizer, batch_size=16, training=True)
    ranker.model.fit(train_ds, epochs=3, verbose=1)

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

    result = ranker.predict("what disease shown here", visual, candidates, tokenizer)

    print("\nanswer: ", result.answer)
    print("confidence: ", round(result.confidence, 3))
    print("top_k: ", [(t[:40], round(s, 3)) for t, s in result.top_k])
    print("alternatives: ", [a[:40] for a in result.alternatives()])

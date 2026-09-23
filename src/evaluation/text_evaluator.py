import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.data_preprocessing.image_preprocessing as img_pre
from src.agents.Visual_Agent import VisualPrediction
from src.util.logger import Logger


def prediction_from_row(row, heads=("crop", "disease", "severity")):
    predictions = {}
    for head in heads:
        predictions[head] = [(getattr(row, head), 1.0)]

    return VisualPrediction(row.crop, 1.0, predictions["crop"], predictions)


def score_queries(index, queries):
    vectors = index.embedder.encode(queries, normalize=True)
    return vectors @ index.vectors.T


def top_k_texts(index, scores, k):
    top = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    texts = [record.text for record in index.records]
    return np.array(texts)[top]


def recall_at_k(ranked_texts, answers, k):
    matches = ranked_texts[:, :k] == np.array(answers)[:, None]
    return float(matches.any(axis=1).mean())


def gold_rank(index, scores, answers):
    positions = {r.text: i for i, r in enumerate(index.records)}
    ranks = []

    for row, answer in enumerate(answers):
        position = positions.get(answer)
        if position is not None:
            ranks.append(int((scores[row] > scores[row, position]).sum()) + 1)

    return np.array(ranks)


def top1_similarity(index, scores, answers):
    best = scores.argmax(axis=1)
    gold = index.embedder.encode(answers, normalize=True)
    return (index.vectors[best] * gold).sum(axis=1)


def label_match(index, scores, rows):
    best = scores.argmax(axis=1)
    metadata = [index.records[i].metadata for i in best]

    crop = [m.get("crop") == r.crop for m, r in zip(metadata, rows.itertuples())]
    disease = [m.get("disease") == r.disease for m, r in zip(metadata, rows.itertuples())]

    return np.array(crop), np.array(disease)


def evaluate_ranker(ranker, data, sample=200, ks=(1, 3, 5), seed=0, name="Text"):
    index = ranker.index
    rows = data.sample(min(sample, len(data)), random_state=seed)

    queries = [
        ranker.build_query(row.question_text, prediction_from_row(row))
        for row in rows.itertuples()
    ]

    scores = score_queries(index, queries)
    ranked = top_k_texts(index, scores, max(ks))
    answers = rows["answer"].to_numpy()

    indexed = {record.text for record in index.records}
    ranks = gold_rank(index, scores, answers)
    similarity = top1_similarity(index, scores, answers)
    crop, disease = label_match(index, scores, rows)

    results = {f"recall@{k}": recall_at_k(ranked, answers, k) for k in ks}
    results["ceiling"] = np.mean([a in indexed for a in answers])
    results[f"baseline@{max(ks)}"] = max(ks) / len(index)
    results["median_gold_rank"] = np.median(ranks) if len(ranks) else np.nan
    results["mean_top1_similarity"] = similarity.mean()
    results["top1_similarity>0.7"] = (similarity > 0.7).mean()
    results["crop_match"] = crop.mean()
    results["disease_match"] = disease.mean()
    results["index_size"] = len(index)
    results["n"] = len(rows)

    table = pd.DataFrame({name: results}).round(5)
    Logger.info("\n" + table.to_string())
    return table


def inspect(ranker, data, count=3, seed=0, k=5, show_image=True):
    index = ranker.index
    indexed = {record.text for record in index.records}

    for row in data.sample(count, random_state=seed).itertuples():
        visual = prediction_from_row(row)
        query = ranker.build_query(row.question_text, visual)
        ranked = index.search(query, k=k)

        if show_image:
            image = img_pre.load_image(row.image_path)
            if image is not None:
                plt.figure(figsize=(3, 3))
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"{row.crop} | {row.disease} | {row.severity}", fontsize=9)
                plt.show()

        print("image:", row.image_path)
        print("question:", row.question_text)
        print("query:", query)
        print("expected:", row.answer)
        print("in index:", row.answer in indexed)
        print("returned:")

        for record, score in ranked:
            print(f"{score:.3f} [{record.metadata.get('source')}] {record.text[:110]}")

        print("-" * 110)
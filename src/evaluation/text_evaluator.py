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


def show_truth(data, count=3, seed=0, k=5):

    for row in data.sample(count, random_state=seed).itertuples():
        visual = prediction_from_row(row)

        image = img_pre.load_image(row.image_path)
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{row.crop} - {row.disease} - {row.severity}", fontsize=12)
        plt.show()


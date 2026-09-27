import matplotlib.pyplot as plt
from pathlib import Path
import base64
import anthropic

import src.data_preprocessing.image_preprocessing as img_pre
from src.agents.Visual_Agent import VisualPrediction
from src.util.logger import Logger
from dotenv import load_dotenv

load_dotenv()

EVALUATION_MODEL = "claude-haiku-4-5-20251001"
def prediction_from_row(row, heads=("crop", "disease", "severity")):
    predictions = {}
    for head in heads:
        predictions[head] = [(getattr(row, head), 1.0)]

    return VisualPrediction(row.crop, 1.0, predictions["crop"], predictions)

def _image_base64(image_path):
    path = Path(image_path)
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": media, "data": data}}
def evaluate_response(model_response, image, question, chunks, claude_client = None):
    client =claude_client or anthropic.Anthropic()
    prompt = (
        f"Question asked about the plant pictured in the image:\n{question}\n\n"
        f"Training Model's answer:\n{model_response}\n\n"
        f"Chunks the model was given:\n{chunks}\n\n"
        "First judge the model against the image and the accuracy of the response"
        "Then judge the model against the context it was given and if this swayed the accuracy."
        "CRUCIAL - List any claim by the model not supported by evidence as a hallucination"
        "List any inaccuracy found by the model and why"
    )
    response = client.messages.parse(
        model=EVALUATION_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": [_image_base64(image), {"type": "text", "text": prompt}]}],
    )
    return response.parsed_output

def show_truth(data, count=3, seed=0, k=5):

    for row in data.sample(count, random_state=seed).itertuples():
        visual = prediction_from_row(row)

        image = img_pre.load_image(row.image_path)
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{row.crop} - {row.disease} - {row.severity}", fontsize=12)
        plt.show()


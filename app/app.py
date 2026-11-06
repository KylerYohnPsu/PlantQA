import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["KERAS_BACKEND"] = "torch"
from src.main import PlantQA

def main():
    plantqa= PlantQA(
    ip_address="127.0.0.1",
    port=8000,
    match_threshold=0.5,
    unet_weights="models/unet_models/UNet_Best.keras",
    visual_model_weights="models/visual_models/10_11.keras",
    visual_model_json="models/visual_models/10_11_classes.json",
    response_model_adapter="models/response_models/10_9",
)
    plantqa.run()
if __name__ == '__main__':
    main()
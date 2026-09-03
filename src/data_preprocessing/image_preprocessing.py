from src.util.logger import Logger
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image and return it
    PARAM:
        image_path: str | The path to the image
    RETURN:
        np.array: The image as a np array
    """
    # load the image using pillow and convert it into a numpy array
    with Image.open(image_path) as img:
        Logger.debug(f"Loaded image: {image_path}")
        np_img= img.convert("RGB")
        return np_img
    
def show_image(image: np.ndarray):
    """
    Display a np array image using matplotlib
    PARAM:
        image: np.ndarray | The image you want to show
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def preprocess_image(image_path: str, processed_size: tuple=(500,500)):
    """
    Preprocess a single image
    Just a placeholder for now, feel free to edit/remove
    """
    return


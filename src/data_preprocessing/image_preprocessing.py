from src.util.logger import Logger
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image as a RGB np array.
    PARAM:
        image_path: str | The path to the image
    RETURN:
        np.ndarray: The image as a np RGB array.
    """
    # Load and convert while the file is open so the returned array is fully
    with Image.open(image_path) as img:
        loaded_image = np.asarray(img.convert("RGB"))

    Logger.debug(f"Loaded image: {image_path}")

    return loaded_image
    
def show_image(image: np.ndarray | Image.Image):
    """
    Display a np array image using matplotlib
    PARAM:
        image: np.ndarray, Image.Image | The image you want to show
    """
    if isinstance(image, np.ndarray) and image.ndim == 2:
        # this is a greyscale image, tell plt to handle it specially
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    normalize a given np image
    PARAM:
        image: np.ndarray | The image to normalize
    RETURN:
        np.ndarray: The normalized image
    """
    # convert the array to floats so each pixel RGB value can fit between 0 and 1
    float_image= np.asarray(image, dtype=np.float32)
    normalized_image= float_image / 255.0
    return normalized_image

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Make an image greyscale
    PARAM:
        image: np.ndarray | The image to grey-ify
    RETURN:
        np.ndarray: The greyscale image
    """
    img= Image.fromarray(image)
    grey_image= img.convert("L")
    grey_array= np.array(grey_image)
    return grey_array

# I (chris) spell it grEy (not grAy), so this is a special wrapper just for me
convert_to_greyscale= convert_to_grayscale

def resize_image(image: np.ndarray, new_size: tuple[int,int]= (500,500)) -> np.ndarray:
    """
    resize the given image to the given size
    PARAM:
        image: np.ndarray | The image to resize
        new_size: (int, int) | The (width, height) you want to resize to
    RETURN:
        np.ndarray: The resized image
    """
    if new_size[0] <= 0 or new_size[1] <= 0:
        return None # invalid size
    
    pil_image= Image.fromarray(image)
    resized_image= pil_image.resize(new_size, Image.BILINEAR)
    resized_np_image= np.array(resized_image)
    return resized_np_image

def preprocess_image(image_path: str, processed_size: tuple=(500,500)):
    """
    Preprocess a single image
    Just a placeholder for now, feel free to edit/remove
    """
    return

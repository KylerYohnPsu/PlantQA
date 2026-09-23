import cv2
import numpy as np

import src.data_preprocessing.image_preprocessing as img_pre

MASK_CHANNELS = 2


def to_rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def clahe(image, clip_limit=2.0, tile=(8, 8)):
    gray_image = cv2.cvtColor(to_rgb_uint8(image), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(gray_image)


def foreground(image):
    hsv = cv2.cvtColor(to_rgb_uint8(image), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def make_mask(image, size=(224, 224)):
    image = to_rgb_uint8(image)
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    return np.stack([clahe(image), foreground(image)], axis=-1)


def mask_from_path(image_path, size=(224, 224), image=None):
    if image is None:
        image = img_pre.load_image(str(image_path))
    if image is None:
        return np.zeros((size[1], size[0], MASK_CHANNELS), np.float32)
    return make_mask(image, size).astype(np.float32) / 255.0


def k_means(image, k: int):

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = img_rgb.reshape((-1,3))

    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    compactness, labels, centers = cv2.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image


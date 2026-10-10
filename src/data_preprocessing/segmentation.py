import cv2
import numpy as np
import src.config as config
import src.data_preprocessing.image_preprocessing as img_pre
from pathlib import Path

MASK_CHANNELS = 2
MASK_ROOT = config.Data.SEGMENTED_IMAGES

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

def unet_mask(image_path, size):
    p = Path(image_path)
    mask = cv2.imread(str(MASK_ROOT / p.parent.name / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return cv2.resize(mask, tuple(size), interpolation=cv2.INTER_AREA)


def mask_from_path(image_path, size=(224, 224), image=None):
    if image is None:
        image = img_pre.load_image(str(image_path))
    if image is None:
        return np.zeros((size[1], size[0], MASK_CHANNELS), np.float32)
    return mask_channels(image, unet_mask(image_path, size), size)

def mask_channels(rgb_image, mask_image, size):
    rgb_image = to_rgb_uint8(rgb_image)
    if (rgb_image.shape[1], rgb_image.shape[0]) != tuple(size):
        rgb_image = cv2.resize(rgb_image, tuple(size), interpolation=cv2.INTER_AREA)
    if mask_image is None:
        mask_image = clahe(rgb_image)
    return np.stack([foreground(rgb_image), mask_image], axis=-1).astype(np.float32) / 255.0


def unet_mask_from_image(unet, image, size):
    resized_image = img_pre.resize_image(to_rgb_uint8(image), (384, 384))
    x = np.asarray(resized_image, np.float32)[None] / 255.0
    if unet.input_shape[1] == 3:
        pred = unet.predict(np.transpose(x, (0, 3, 1, 2)), verbose=0)[0, 0]
    else:
        pred = unet.predict(x, verbose=0)[0, ..., 0]
    return cv2.resize((pred * 255).astype(np.uint8), tuple(size), interpolation=cv2.INTER_AREA)

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


def get_percent_of_Disease(img_path, mask_path):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    img = img_pre.crop_border(img)
    img = cv2.resize(img, (384, 384))
    mask = (mask > 0).astype(np.uint8) * 255
    lesion_pixels = int(np.count_nonzero(mask))
    gray_Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray_Image, (5,5),0)
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #find contours

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_mask = np.zeros_like(gray_Image)

    cv2.drawContours(image=filled_mask, contours=contours, contourIdx=-1, color=(255,255,255), thickness=cv2.FILLED)

    if contours:
        area = 0
        for cont in contours:
            area += cv2.contourArea(cont)
    else:
        return ('Unknown', 'UNKNOWN')
    pct_diseased = (lesion_pixels / area) * 100

    if pct_diseased < 15:
        return ('healthy', 'HEALTHY')
    elif  pct_diseased < 40:
        return('disease', 'MILD')
    elif  pct_diseased < 75:
        return('disease', 'MODERATE')
    elif  pct_diseased < 100:
        return('disease', 'SEVERE')
    else:
        return('disease', 'UNKNOWN')


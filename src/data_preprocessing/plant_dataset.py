from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import tensorflow as tf

def get_img_pairs(img_dir, mask_dir):
    img_p, mask_p = [], []
    for img in sorted(img_dir.iterdir()):
        mask = mask_dir / str(img.stem + '.png')
        if mask.exists():
            img_p.append(str(img))
            mask_p.append(str(mask))
        else:
            print(f"No Mask for {img.name}")
    return img_p, mask_p

def load_and_preprocess_image(img_path, mask_path, img_size=(384,384)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32)
    mask = tf.cast(mask > 0, tf.float32)
    return img, mask

def generate_dataset(img_dir, mask_dir, img_size=(384,384), batch_size=32):
    img_paths, mask_paths = get_img_pairs(img_dir, mask_dir)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    dataset = dataset.map(
        lambda img, mask: load_and_preprocess_image(img, mask, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

"""
Put code for exploring image data here
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.data_preprocessing.image_preprocessing as img_pre
import src.data_preprocessing.segmentation as seg


def full_path(image_path: str, images_root: str) -> str:
    image_path = str(image_path)
    if image_path.startswith("/"):
        return image_path
    return f"{str(images_root).rstrip('/')}/{image_path}"


def load_sample(data, images_root, count: int, seed: int = 0) -> list:
    paths = data["image_path"].drop_duplicates().sample(count, random_state=seed)

    images = []
    for path in paths:
        image = img_pre.load_image(full_path(path, images_root))
        if image is not None:
            images.append(image)

    return images


def show_label_conflicts(data, heads):
    rows = len(data)
    images = data["image_path"].nunique()
    print(f"question rows: {rows}   unique images: {images}   rows per image: {rows / images:.1f}")

    for head in heads:
        per_image = data.groupby("image_path")[head].nunique()
        print(f"{head}: {(per_image > 1).sum()} images carry more than one label")


def show_class_support(images, heads, plot_head: str = "disease"):
    support = pd.DataFrame({
        "classes": {h: images[h].nunique() for h in heads},
        "median_per_class": {h: int(images[h].value_counts().median()) for h in heads},
        "classes_under_100": {h: int((images[h].value_counts() < 100).sum()) for h in heads},
    })
    print(support)

    images[plot_head].value_counts().plot(logy=True, figsize=(10, 4),
                                          title=f"{plot_head} class support")
    plt.ylabel("images")
    plt.xticks([])
    plt.show()

    return support


def show_padding(data, images_root, count: int = 8, columns: int = 4, seed: int = 0):
    images = load_sample(data, images_root, count, seed)
    rows = int(np.ceil(len(images) / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.4 * rows),
                                constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, image in zip(axes, images):
        content = img_pre.crop_border(image).size / image.size
        ax.imshow(image)
        ax.set_title(f"{image.shape[1]}x{image.shape[0]} | {content:.0%} content", fontsize=9)

    for ax in axes:
        ax.axis("off")

    figure.suptitle("Raw images before cropping")
    plt.show()


def show_cropped_and_mask(data, images_root, count: int = 3, size: tuple = (224, 224), seed: int = 0):
    images = load_sample(data, images_root, count, seed)
    titles = ["cropped", "clahe", "foreground"]

    figure, axes = plt.subplots(len(images), 3, figsize=(9, 3.4 * len(images)),
                                constrained_layout=True)

    for row, image in zip(np.atleast_2d(axes), images):
        cropped = img_pre.resize_image(img_pre.crop_border(image), size)
        mask = seg.make_mask(cropped, size)

        for ax, panel, title in zip(row, [cropped, mask[:, :, 0], mask[:, :, 1]], titles):
            ax.imshow(panel, cmap=None if title == "cropped" else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    figure.suptitle("Cropped image and its mask channels")
    plt.show()

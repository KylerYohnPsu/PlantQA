import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score

from src.agents.Visual_Agent import VisualModel
from src.util.logger import Logger


def get_predictions(vm: VisualModel, df: pd.DataFrame, images_root, batch_size: int = 32):
    ds = VisualModel.make_dataset(df, vm.classes, images_root, size=vm.img_size,
                                  batch_size=batch_size, training=False, use_mask=vm.use_mask)
    heads = list(vm.classes)
    y_true = {h: [] for h in heads}
    probs = {h: [] for h in heads}

    for images, labels in ds:
        outputs = vm.model.predict_on_batch(images)
        for head, out in zip(heads, outputs):
            y_true[head].append(labels[head].numpy())
            probs[head].append(np.asarray(out))

    return {h: (np.concatenate(y_true[h]), np.concatenate(probs[h])) for h in heads}


def metrics_table(predictions: dict, k: int = 3) -> pd.DataFrame:
    rows = []
    for head, (y_true, probs) in predictions.items():
        y_pred = probs.argmax(axis=1)
        top_k = np.argsort(probs, axis=1)[:, -k:]
        rows.append({
            "head": head,
            "n": len(y_true),
            "accuracy": (y_pred == y_true).mean(),
            f"top{k}_accuracy": (top_k == y_true[:, None]).any(axis=1).mean(),
            "macro_f1": f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0),
            "majority_baseline": np.bincount(y_true).max() / len(y_true),
        })
    return pd.DataFrame(rows).set_index("head").round(3)


def log_classification_report(predictions: dict, classes: dict, head: str):
    y_true, probs = predictions[head]
    y_pred = probs.argmax(axis=1)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=[classes[head][i] for i in labels], zero_division=0)
    Logger.info(f"\n{head} classification report:\n{report}")


def plot_confusion_matrix(predictions: dict, classes: dict, head: str):
    y_true, probs = predictions[head]
    y_pred = probs.argmax(axis=1)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=[classes[head][i] for i in labels],
        normalize="true", values_format=".2f", cmap="Blues", colorbar=False, xticks_rotation=45,
    )
    plt.title(f"{head}: confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.show()


def history_frame(*histories) -> pd.DataFrame:
    frames = [pd.DataFrame(h.history if hasattr(h, "history") else h) for h in histories]
    frame = pd.concat(frames, ignore_index=True)
    frame.index = frame.index + 1
    return frame


def plot_loss(*histories, heads=None):
    frame = history_frame(*histories)
    names = [f"{h}_loss" for h in heads] if heads else ["loss"]

    plt.figure(figsize=(8, 5))
    for name in names:
        line, = plt.plot(frame.index, frame[name], label=f"train {name}")
        if f"val_{name}" in frame:
            plt.plot(frame.index, frame[f"val_{name}"], "--", color=line.get_color(), label=f"val {name}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy(*histories, heads=None):
    frame = history_frame(*histories)
    heads = heads or [c[:-len("_accuracy")] for c in frame if c.endswith("_accuracy") and not c.startswith("val_")]

    plt.figure(figsize=(8, 5))
    for head in heads:
        line, = plt.plot(frame.index, frame[f"{head}_accuracy"], label=f"train {head}")
        if f"val_{head}_accuracy" in frame:
            plt.plot(frame.index, frame[f"val_{head}_accuracy"], "--", color=line.get_color(), label=f"val {head}")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training vs validation accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate(vm: VisualModel, df: pd.DataFrame, images_root, name: str = "Test",
             k: int = 3, max_matrix_classes: int = 15):
    Logger.info(f"\n\n-------- {name}: evaluating {len(df)} images --------\n\n")
    predictions = get_predictions(vm, df, images_root)

    table = metrics_table(predictions, k)
    Logger.info(f"\n{table}")

    for head in predictions:
        log_classification_report(predictions, vm.classes, head)
        if len(vm.classes[head]) <= max_matrix_classes:
            plot_confusion_matrix(predictions, vm.classes, head)

    return table, predictions
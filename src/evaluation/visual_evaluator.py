import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score

from src.agents.Visual_Agent import VisualModel
from src.util.logger import Logger


def predict(visual_model: VisualModel, df: pd.DataFrame, images_root, batch_size: int = 32):
    ds = VisualModel.make_dataset(df, visual_model.classes, images_root, size=visual_model.img_size,
                                  batch_size=batch_size, training=False, use_mask=visual_model.use_mask)
    c_heads = list(visual_model.classes)
    y_true = {h: [] for h in c_heads}
    probs = {h: [] for h in c_heads}

    for images, labels in ds:
        outputs = visual_model.model.predict_on_batch(images)
        for head, out in zip(c_heads, outputs):
            y_true[head].append(labels[head].numpy())
            probs[head].append(np.asarray(out))

    return {head: (np.concatenate(y_true[head]), np.concatenate(probs[head])) for head in c_heads}


def metrics_table(predictions, num_result = 3):
    rows = []
    for head, (y_true, probs) in predictions.items():
        y_pred = probs.argmax(axis=1)
        top_k = np.argsort(probs, axis=1)[:, -num_result:]
        rows.append({
            "head": head,
            "num results": len(y_true),
            "accuracy": (y_pred == y_true).mean(),
            f"top{num_result}_accuracy": (top_k == y_true[:, None]).any(axis=1).mean(),
            "macro average f1": f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
        })
    return pd.DataFrame(rows).set_index("head").round(3)


def print_report(predictions, classes, pred):
    y_true, probs = predictions[pred]
    y_pred = probs.argmax(axis=1)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=[classes[pred][i] for i in labels], zero_division=0)
    Logger.info(f"\n{pred} classification report:\n{report}")


def plot_confusion_matrix(predictions, classes, c_head):
    y_true, probs = predictions[c_head]
    y_pred = probs.argmax(axis=1)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=[classes[c_head][i] for i in labels],
        normalize="true", values_format=".2f", cmap="Blues", colorbar=False, xticks_rotation=45,
    )
    plt.title(f"{c_head}: confusion matrix")
    plt.tight_layout()
    plt.show()



def frame_history(*histories):
    frames = []
    for history in histories:
        if hasattr(history, "history"):
            frames.append(pd.DataFrame(history.history))
        else:
            frames.append(pd.DataFrame(history))

    frame = pd.concat(frames, ignore_index=True)
    frame.index += 1

    return frame

def plot_loss(*histories, heads=None):
    frame = frame_history(*histories)
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
    frame = frame_history(*histories)
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


def evaluate(visual_model: VisualModel, df: pd.DataFrame, images_root, name = "Visual Model Test",
             num_results = 3, max_matrix_classes = 15):
    Logger.info(f"\n\n {name}: evaluating {len(df)} images \n\n")
    predictions = predict(visual_model, df, images_root)

    table = metrics_table(predictions, num_results)
    Logger.info(f"\n{table}")

    for pred in predictions:
        print_report(predictions, visual_model.classes, pred)
        if len(visual_model.classes[pred]) <= max_matrix_classes:
            plot_confusion_matrix(predictions, visual_model.classes, pred)

    return table, predictions
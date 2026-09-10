from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import src.data_preprocessing.image_preprocessing as img_pre


@dataclass
class VisualPrediction:
    plant_species: str
    conf_interval: float
    top_k: List[Tuple[str, float]]
    heads: Dict[str, List[Tuple[str, float]]]


    def species_filter(self, min_confidence: float = 0.60):
        if self.top_k[0][0] >= min_confidence:
            return [(self.plant_species, self.conf_interval)]
        else:
            return self.top_k
    def keywords(self, threshold: float = 0.50) -> List[str]:
        return {
            head: [name for name, conf in preds if conf >= threshold]
                 for head, preds in self.heads.items()
                 }


class VisualModel:
    def __init__(self, classes: Dict[str, List[str]],
                 img_size=(224, 224), weights="imagenet"):
        self.classes = classes
        self.img_size = img_size
        self.weights = weights  # "imagenet" = transfer and None = from custom
        self.model = None

    @staticmethod
    def make_dataset(df, classes, raw_root, size=(224, 224),
                     batch_size=32, training=False):
        labels = {
            head: df[head].astype(str)
            .map(lambda v, c= names : c.index(v) if v in c else len(c) - 1)
            .to_numpy("int32")
            for head, names in classes.items()
        }

        root = str(raw_root).rstrip("/")
        paths = (root + "/" + df["image_path"].astype(str)).to_numpy()

        def load_image(path, label):
            def preprocess(p):
                array = img_pre.preprocess_image(p.decode("utf-8"), size)
                if array is None:
                    array = np.zeros((size, 3), np.float32)
                return array

            img = tf.numpy_function(preprocess, [path], tf.float32)
            img.set_shape((*size, 3))
            return img, label

        data_set = tf.data.Dataset.from_tensor_slices((paths, labels))
        data_set = data_set.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            data_set = data_set.shuffle(1024)
        return data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def build(self):
        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights=self.weights,
            input_shape=(*self.img_size, 3),
            pooling="avg",
        )
        base.trainable = self.weights is None  # freezing the pretrained weights if any

        inputs = keras.Input(shape=(None, None, 3), name="image")
        x = layers.Resizing(*self.img_size)(inputs)
        x = base(x)
        x = layers.Dropout(0.3)(x)

        outputs = [layers.Dense(len(names), activation = "softmax", name = head)(x) for head, names in self.classes.items()]

        self.model = keras.Model(inputs, outputs, name="plantqa_visual")
        return self.model

    def compile(self, lr=1e-3, head_weight = None):
        weights = head_weight or {"crop": 1.0, "disease": 1.0, "category": .3, "severity": .3}
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss={h: "sparse_categorical_crossentropy" for h in self.classes},
            loss_weights={h: weights.get(h, 1.0) for h in self.classes},
            metrics={h: ["accuracy"] for h in self.classes},
        )

    def fit(self, train_ds, val_ds, epochs=20):
        calls = [
            keras.callbacks.EarlyStopping("val_crop_accuracy", patience=4, mode="max",
                                          restore_best_weights=True),
            keras.callbacks.ModelCheckpoint("best.keras", monitor="val_crop_accuracy",
                                            mode="max", save_best_only=True),
        ]
        return self.model.fit(train_ds, validation_data=val_ds,
                              epochs=epochs, callbacks=calls)

    def predict(self, image: np.ndarray, k: int = 3) -> VisualPrediction:
        preds = self.model.predict(np.expand_dims(image, 0), verbose=0)
        out = {}
        for (head, names), pred_out in zip(self.classes.items(), preds):
            p = pred_out[0]
            top = np.argsort(p)[::-1][:k]
            out[head] = [(names[i], float(p[i])) for i in top]
        return VisualPrediction(
            plant_species = out["crop"][0][0],
            conf_interval = out["crop"][0][1],
            top_k = out["crop"],
            heads = out
        )

    def save(self, path):
        self.model.save(path)


if __name__ == "__main__":
    # Fake data, but has the same shape as the real one from train_df
    classes = {
        "crop":     ["banana", "coconut", "tomato"],
        "disease":  ["healthy", "blight", "anthracnose", "unknown"],
        "category": ["disease", "pest"],
        "severity": ["MILD", "SEVERE", "UNKNOWN"],
    }

    m = VisualModel(classes=classes)
    m.build()
    m.compile()
    m.model.summary()

    # 8 fake images at 256x256
    x = np.random.randint(0, 256, (8, 256, 256, 3)).astype("float32")
    y = {head: np.random.randint(0, len(names), 8)
         for head, names in classes.items()}

    m.model.fit(x, y, epochs=1, batch_size=4, verbose=1)

    # Inference on a single image
    pred = m.predict(np.random.randint(0, 256, (300, 400, 3)).astype("float32"))
    print("\nspecies: ", pred.plant_species, round(pred.conf_interval, 3))
    print("top_k: ", pred.top_k)
    print("heads: ", list(pred.heads.keys()))
    print("keywords: ", pred.keywords(threshold=0.0))   # 0.0mit
    print("filter: ", pred.species_filter(min_confidence=0.6))
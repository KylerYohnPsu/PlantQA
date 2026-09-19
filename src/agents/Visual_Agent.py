from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import keras
from keras import layers
import src.data_preprocessing.image_preprocessing as img_pre
import src.data_preprocessing.segmentation as seg


@dataclass
class VisualPrediction:
    plant_species: str
    conf_interval: float
    top_k: List[Tuple[str, float]]
    heads: Dict[str, List[Tuple[str, float]]]



class VisualModel:
    def __init__(self, classes: Dict[str, List[str]],
                 img_size=(224, 224), weights="imagenet", use_mask=False):
        self.classes = classes
        self.img_size = img_size
        self.weights = weights  # "imagenet" = transfer and None = from custom
        self.use_mask = use_mask
        self.model = None

    @staticmethod
    def unknown_label(values):
        return next((v for v in values if str(v).lower() == "unknown"), "unknown")

    @staticmethod
    def one_row_per_image(df, heads):
        images = pd.DataFrame({"image_path": df["image_path"].unique()})
        for head in heads:
            values = df[head].astype(str)
            known = df[values.str.lower() != "unknown"]
            top = (known.groupby(["image_path", head]).size()
                   .sort_values(ascending=False).reset_index()
                   .drop_duplicates("image_path").set_index("image_path")[head])
            images[head] = images["image_path"].map(top).fillna(VisualModel.unknown_label(values.unique()))

        if "disease" in images:
            healthy = images["disease"] == "healthy"
            if "category" in images:
                images.loc[healthy, "category"] = "healthy"
            if "severity" in images:
                images.loc[healthy, "severity"] = "HEALTHY"
        return images

    @staticmethod
    def build_classes(df, heads):
        classes = {}
        for head in heads:
            names = sorted(df[head].astype(str).unique())
            unknown = VisualModel.unknown_label(names)
            classes[head] = [n for n in names if n != unknown] + [unknown]
        return classes

    @staticmethod
    def make_dataset(df, classes, raw_root, size=(224, 224),
                     batch_size=32, training=False, use_mask=False):
        labels = {}
        for head, names in classes.items():
            index = {name: i for i, name in enumerate(names)}
            unknown = index.get(VisualModel.unknown_label(names), len(names) - 1)
            labels[head] = df[head].astype(str).map(lambda v: index.get(v, unknown)).to_numpy("int32")

        root = str(raw_root).rstrip("/")
        paths = df["image_path"].astype(str).map(
            lambda p: p if p.startswith("/") else f"{root}/{p}"
        ).to_numpy()

        def load(path, label):
            def preprocess(p):
                p = p.decode("utf-8")
                image = img_pre.load_image(p)
                if image is None:
                    image = np.zeros((*size, 3), np.uint8)
                image = img_pre.crop_border(image)
                array = np.asarray(img_pre.resize_image(image, size), dtype=np.float32)
                if use_mask:
                    return array, seg.mask_from_path(p, size, image)
                return array

            if not use_mask:
                img = tf.numpy_function(preprocess, [path], tf.float32)
                img.set_shape((*size, 3))
                return img, label

            img, mask = tf.numpy_function(preprocess, [path], (tf.float32, tf.float32))
            img.set_shape((*size, 3))
            mask.set_shape((*size, seg.MASK_CHANNELS))
            return {"image": img, "mask": mask}, label

        data_set = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            data_set = data_set.shuffle(len(paths))
        data_set = data_set.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def build(self):
        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights=self.weights,
            input_shape=(*self.img_size, 3),
            pooling="avg",
        )
        base.trainable = self.weights is None  # freezing the pretrained weights if any

        image_input = keras.Input(shape=(None, None, 3), name="image")
        x = layers.Resizing(*self.img_size)(image_input)
        x = base(x)
        x = layers.Dropout(0.3)(x)

        inputs = image_input
        if self.use_mask:
            mask_input = keras.Input(shape=(*self.img_size, seg.MASK_CHANNELS), name="mask")
            m = layers.Conv2D(32, 3, activation="relu", padding="same")(mask_input)
            m = layers.BatchNormalization()(m)
            m = layers.MaxPooling2D()(m)
            m = layers.Conv2D(64, 3, activation="relu", padding="same")(m)
            m = layers.BatchNormalization()(m)
            m = layers.GlobalAveragePooling2D()(m)
            m = layers.Dense(128, activation="relu")(m)

            inputs = [image_input, mask_input]
            x = layers.Concatenate()([x, m])

        outputs = [layers.Dense(len(names), activation = "softmax", name = head)(x) for head, names in self.classes.items()]

        self.model = keras.Model(inputs, outputs, name="plantqa_visual")
        return self.model

    def compile(self, lr=1e-3):
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss={h: "sparse_categorical_crossentropy" for h in self.classes},
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
        if self.use_mask:
            resized = img_pre.resize_image(seg.to_rgb_uint8(image), self.img_size)
            mask = seg.make_mask(resized, self.img_size).astype(np.float32) / 255.0
            model_input = {"image": np.expand_dims(resized.astype(np.float32), 0),
                           "mask": np.expand_dims(mask, 0)}
        else:
            model_input = np.expand_dims(image, 0)

        preds = self.model.predict(model_input, verbose=0)
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

    def unfreeze(self, n_layers=30, lr=1e-5):
        base = next(layer for layer in self.model.layers if isinstance(layer, keras.Model))
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        self.compile(lr=lr)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, model_path, classes_path=None, img_size=(224, 224)) -> "VisualModel":
        classes_path = Path(classes_path) if classes_path else cls.classes_path_for(model_path)
        classes = json.loads(Path(classes_path).read_text())

        vm = cls(classes=classes, img_size=img_size)
        vm.model = keras.models.load_model(model_path)
        vm.use_mask = len(vm.model.inputs) > 1

        for head, names in classes.items():
            n_out = vm.model.get_layer(head).output.shape[-1]
            if n_out != len(names):
                raise ValueError(f"head '{head}': model has {n_out} outputs but {len(names)} class names")
        return vm


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
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
    plant_classification: str
    conf_interval: float
    top_k_results: List[Tuple[str, float]]
    classification_heads: Dict[str, List[Tuple[str, float]]]



class VisualModel:
    def __init__(self, classes: Dict[str, List[str]],
                 img_size=(224, 224), weights="imagenet", use_mask=False):
        self.classes = classes
        self.img_size = img_size
        self.weights = weights
        self.use_mask = use_mask
        self.model = None

    @staticmethod
    def add_unknown_label(values):
        return next((v for v in values if str(v).lower() == "unknown"), "unknown")

    @staticmethod
    def one_row_per_image(df, classification_heads):
        images = pd.DataFrame({"image_path": df["image_path"].unique()})
        for head in classification_heads:
            values = df[head].astype(str)
            known_images = df[values.str.lower() != "unknown"] #filter out unknown
            first_image = (known_images.groupby(["image_path", head]).size()
                   .sort_values(ascending=False).reset_index()
                   .drop_duplicates("image_path").set_index("image_path")[head])
            images[head] = images["image_path"].map(first_image).fillna(VisualModel.add_unknown_label(values.unique()))

        if "disease" in images:
            healthy = images["disease"] == "healthy"
            if "category" in images:
                images.loc[healthy, "category"] = "healthy"
            if "severity" in images:
                images.loc[healthy, "severity"] = "HEALTHY"
        return images

    @staticmethod
    def build_classes(df, classification_heads):
        plant_classes = {}
        for head in classification_heads:
            names = sorted(df[head].astype(str).unique())
            unknown = VisualModel.add_unknown_label(names)
            plant_classes[head] = [n for n in names if n != unknown] + [unknown]
        return plant_classes

    @staticmethod
    def make_dataset(df, plant_classes, raw_root, size=(224, 224),
                     batch_size=32, training=False, use_mask=False):
        labels = {}
        for head, names in plant_classes.items():
            plant_index = {name: i for i, name in enumerate(names)}
            unknown = plant_index.get(VisualModel.add_unknown_label(names), len(names) - 1)
            labels[head] = df[head].astype(str).map(lambda v: plant_index.get(v, unknown)).to_numpy("int32")

        root = str(raw_root).rstrip("/")
        paths = df["image_path"].astype(str).map(
            lambda p: p if p.startswith("/") else f"{root}\{p}"
        ).to_numpy() # add image path for deduping later

        def load_plant_image(path, label):
            def preprocess_image(image_path):
                image_path = image_path.decode("utf-8")
                curr_image = img_pre.load_image(image_path)
                if curr_image is None:
                    curr_image = np.zeros((*size, 3), np.uint8)
                cropped_image = img_pre.crop_border(curr_image) #remove image border
                resized_image = img_pre.resize_image(cropped_image, size)
                image_array = np.asarray(resized_image, dtype=np.float32)
                if use_mask:
                    return image_array, seg.mask_from_path(image_path, size, cropped_image)
                return image_array

            if not use_mask:
                img = tf.numpy_function(preprocess_image, [path], tf.float32)
                img.set_shape((*size, 3))
                return img, label

            img, mask = tf.numpy_function(preprocess_image, [path], (tf.float32, tf.float32))
            img.set_shape((*size, 3))
            mask.set_shape((*size, seg.MASK_CHANNELS))
            return {"image": img, "mask": mask}, label

        data_set = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            data_set = data_set.shuffle(len(paths))
        data_set = data_set.map(load_plant_image, num_parallel_calls=tf.data.AUTOTUNE)
        return data_set.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def build(self):
        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights=self.weights,
            input_shape=(*self.img_size, 3),
            pooling="avg",
        )
        base.trainable = self.weights is None  # freezing the pretrained weights

        image_input = keras.Input(shape=(*self.img_size, 3), name="image")
        #x = layers.Resizing(*self.img_size)(image_input)
        x = base(image_input)
        x = layers.Dropout(0.3)(x)

        inputs = image_input
        if self.use_mask: #adding a masked channel if there is one being used
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

        predictions = self.model.predict(model_input, verbose=0)
        out = {}
        for (c_head, names), pred_out in zip(self.classes.items(), predictions):
            p = pred_out[0]
            top_pred = np.argsort(p)[::-1][:k]
            out[c_head] = [(names[i], float(p[i])) for i in top_pred]
        return VisualPrediction(
            plant_classification = out["crop"][0][0],
            conf_interval = out["crop"][0][1],
            top_k_results = out["crop"],
            classification_heads = out
        )

    def unfreeze_layers(self, n_layers=30, lr=1e-5):
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
        saved_classes = json.loads(Path(classes_path).read_text())

        visual_model = cls(classes=saved_classes, img_size=img_size)
        visual_model.model = keras.models.load_model(model_path)
        visual_model.use_mask = len(visual_model.model.inputs) > 1

        for c_head, names in saved_classes.items():
            number_out = visual_model.model.get_layer(c_head).output.shape[-1]
            if number_out != len(names):
                raise ValueError(f"head '{c_head}': model has {number_out} outputs listed and {len(names)} class names provided")
        return visual_model

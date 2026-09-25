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

class UNET:
    def __init__(self, img_size=(500,500)):
        self.img_size = img_size
        self.model = None


    def conv_block(inputs, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01) (x)

        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01) (x)

    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters=num_filters)(inputs)
        p = tf.keras.layers.MaxPool2D((2,2)) (x)
        return x, p

    def decoder_block(self, inputs, skip, num_filters):
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")
        x - tf.keras.layers.Concatenate()([x, skip])
        x = self.conv_block(x, num_filters=num_filters)

        return x


    def build_unet(self, input_shape):

        inputs = tf.keras.layers.Input(input_shape)

        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        b1 = self.conv_block(p4, 1024)

        d4 = self.decoder_block(b1, s4, 512)
        d3 = self.decoder_block(d4, s3, 256)
        d2 = self.decoder_block(d3, s2, 128)
        d1 = self.decoder_block(d2, s1, 64)

        outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d1)
        model = tf.keras.models.Model(inputs, outputs, name = "UNET")
        self.model =model

    def compile(self, lr=1e-3):
        self.model.compile(optimizer=keras.optimizers.Adam(lr),
                           loss=keras.losses.BinaryFocalCrossentropy())



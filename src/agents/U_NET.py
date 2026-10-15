
import keras
import numpy as np

class UNET:
    def __init__(self, img_size=(500,500)):
        self.img_size = img_size
        self.model = None


    def conv_block(self, inputs, num_filters):
        x = keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)

        x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)
        return x
    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters=num_filters)
        p = keras.layers.MaxPool2D((2,2))(x)
        return x, p

    def decoder_block(self, inputs, skip, num_filters):
        x = keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
        x = keras.layers.Concatenate(axis=-1)([x, skip])
        x = self.conv_block(x, num_filters=num_filters)

        return x


    def build_unet(self, input_shape):

        inputs = keras.layers.Input(input_shape)

        s1, p1 = self.encoder_block(inputs, 32)
        s2, p2 = self.encoder_block(p1, 64)
        s3, p3 = self.encoder_block(p2, 128)
        s4, p4 = self.encoder_block(p3, 256)

        b1 = self.conv_block(p4, 512)

        d4 = self.decoder_block(b1, s4, 256)
        d3 = self.decoder_block(d4, s3, 128)
        d2 = self.decoder_block(d3, s2, 64)
        d1 = self.decoder_block(d2, s1, 32)

        outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", dtype = 'float32')(d1)
        model = keras.models.Model(inputs, outputs, name = "UNET")
        self.model =model

    def compile(self, lr=1e-4):
        self.model.compile(optimizer=keras.optimizers.Adam(lr),
                           loss= keras.losses.Dice(axis=(1,2,3),name='dice'),
                           metrics=[keras.metrics.Accuracy(name='accuracy'), keras.metrics.BinaryIoU(target_class_ids=(0,1), threshold=0.7, name='BIoU')],
                           jit_compile=False)
        
    def predict(self, image, threshold=0.7):

         if len(image.shape) == 3:
              image = np.expand_dims(image, axis=0)

         probs = self.model.predict(image, verbose=0)

         binary_mask = (probs > threshold).astype(np.uint8)
         return np.squeeze(binary_mask, axis=0)
    
    def fit(self, train_ds, val_ds, epochs=20):
            calls = [
                keras.callbacks.EarlyStopping("val_IoU", patience=8, mode="max",
                                              restore_best_weights=True),
                keras.callbacks.ModelCheckpoint("best.keras", monitor="val_BIoU",
                                                mode="max", save_best_only=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_dice', mode='min',factor=0.2, patience=4, min_lr=1e-6)
            ]
            return self.model.fit(train_ds, validation_data=val_ds,
                                  epochs=epochs, callbacks=calls)
    def save(self, path):
        self.model.save(path)
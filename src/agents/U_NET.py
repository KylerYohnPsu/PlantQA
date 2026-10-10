
import keras
from keras import ops
import numpy as np
#implement own Binary IoU, Keras consistently would convert between TF tensor and Torch tensors adding a significant amount of training time
@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = ops.cast(y_true, y_pred.dtype)

    y_true = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
    y_pred = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

    inter = ops.sum(y_true * y_pred, axis=1)
    union = ops.sum(y_true, axis=1) + ops.sum(y_pred, axis=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - ops.mean(dice)
@keras.saving.register_keras_serializable()
def bce_dice(y_true, y_pred):
     bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
     return bce + dice_loss(y_true, y_pred)

@keras.saving.register_keras_serializable()
class BinaryIoUFast(keras.metrics.Metric):

    def __init__(self, threshold=0.7, name='IoU', **kw):
          super().__init__(name=name, **kw)
          self.threshold = threshold
          self.inter = self.add_weight(name='inter', initializer='zeros')
          self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        #ensure the data types are compatible
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

        #having issue with y_true being located on cpu and not GPU, this line will move y_true to gpu
        if hasattr(y_true, 'device') and y_true.device != y_pred.device:
             y_true = y_true.to(y_pred.device)

        #calculate Binary IoU, ensure Y_pred is binary representation
        y_pred = keras.ops.cast(y_pred > self.threshold, dtype = y_pred.dtype)
        y_true = keras.ops.cast(y_true, dtype = y_pred.dtype)

        inter = keras.ops.sum(y_pred * y_true)
        union = keras.ops.sum(y_true) + keras.ops.sum(y_pred) - inter

        self.inter.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
         smooth = 1e-7
         return (self.inter + smooth) / (self.union + smooth)

    def reset_state(self):
         self.inter.assign(0.0)
         self.union.assign(0.0)


class UNET:
    def __init__(self, img_size=(500,500)):
        self.img_size = img_size
        self.model = None


    def conv_block(self, inputs, num_filters):
        x = keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization(axis=1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)

        x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization(axis=1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)
        return x
    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters=num_filters)
        p = keras.layers.MaxPool2D((2,2))(x)
        return x, p

    def decoder_block(self, inputs, skip, num_filters):
        x = keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
        x = keras.layers.Concatenate(axis=1)([x, skip])
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
                           loss=bce_dice,
                           metrics=[BinaryIoUFast(name='IoU', threshold=0.7)])#IoU_Coef(name='IoU')])  #BinaryIoUFast(name='IoU')])

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
                keras.callbacks.ModelCheckpoint("best.keras", monitor="val_IoU",
                                                mode="max", save_best_only=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
            ]
            return self.model.fit(train_ds, validation_data=val_ds,
                                  epochs=epochs, callbacks=calls)

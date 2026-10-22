
import keras
import numpy as np
import keras.ops as k
import tensorflow as tf
@keras.saving.register_keras_serializable()
class f1_score(keras.metrics.F1Score):
    def __init__(self, average=None, threshold=None, name="f1_score", dtype=None):
        super().__init__(average, threshold, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1,1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1,1])
        return super().update_state(y_true, y_pred)
@keras.saving.register_keras_serializable()
class BCE_DICE(keras.losses.Loss):

    def __init__(self, threshold=0.5,smoothing=1,name='BCE_DICE', **kwargs):
        super().__init__(name=name,**kwargs)
        self.threshold = threshold
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        #ensure tensors are floats 
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        #get the losses returned from keras PER SAMPLE NOT OVER BATCH
        bce_loss = keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        bce_loss = tf.reduce_mean(bce_loss, axis=[1,2])
        intersection = k.sum(y_true * y_pred, axis=[1,2,3])
        union = k.sum(y_true, axis=[1,2,3]) + k.sum(y_pred, axis=[1,2,3])
        dice_loss = 1 - (2 * intersection + self.smoothing)/(union + self.smoothing)  
        return bce_loss + dice_loss
    
    def get_config(self):
        config = super().get_config() 
        config.update({'threshold': self.threshold, "smoothing": self.smoothing})
        return config
    

class UNET:
    def __init__(self, img_size=(512,512), dropout_value=0.1, add_dropout = True):
        self.img_size = img_size
        self.dropout_value = dropout_value
        self.add_dropout = add_dropout
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
        if self.add_dropout:
            d2 = keras.layers.SpatialDropout2D(self.dropout_value)(d2)
        d1 = self.decoder_block(d2, s1, 32)
        if self.add_dropout:
            d1 = keras.layers.SpatialDropout2D(self.dropout_value)(d1)

        outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", dtype = 'float32')(d1)
        model = keras.models.Model(inputs, outputs, name = "UNET")
        self.model =model

    def compile(self, lr=1e-4):
        self.model.compile(optimizer=keras.optimizers.Adam(lr),
                           loss= BCE_DICE(threshold=0.7),
                           metrics=[keras.metrics.BinaryAccuracy(threshold=0.7, name='accuracy'),
                                    keras.metrics.BinaryIoU(target_class_ids=(0,1), threshold=0.7, name='BIoU'),
                                    keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.7, name='Disease_BIoU'), 
                                    f1_score('macro',0.7,'MACRO_f1_score'), 
                                    keras.metrics.Precision(name='Precision', thresholds=0.7)
                                    ,keras.metrics.Recall(name='Recall', thresholds=0.7),],
                           jit_compile=False)
        
    def predict(self, image, threshold=0.7):

         if len(image.shape) == 3:
              image = np.expand_dims(image, axis=0)

         probs = self.model.predict(image, verbose=0)

         binary_mask = (probs > threshold).astype(np.uint8)
         return np.squeeze(binary_mask, axis=0)
    
    def fit(self, train_ds, val_ds, epochs=20):
            calls = [
                keras.callbacks.EarlyStopping("val_Disease_BIoU", patience=8, mode="max",
                                              restore_best_weights=True),
                keras.callbacks.ModelCheckpoint("best.keras", monitor="val_Disease_BIoU",
                                                mode="max", save_best_only=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min',factor=0.2, patience=4, min_lr=1e-6)
            ]
            return self.model.fit(train_ds, validation_data=val_ds,
                                  epochs=epochs, callbacks=calls)
    def save(self, path):
        self.model.save(path)
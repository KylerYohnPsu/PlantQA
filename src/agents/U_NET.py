"""
A Unet model is built for segmentation tasks. This is built using Keras.
All items within this file are utilized within or for the UNet model

"""
import keras
import numpy as np
import keras.ops as k
import tensorflow as tf

#Keras F1_Score is not functional for the type/shape of data that is utilized in this
#To compensate create a custom metric
#save to the model, allowing for keras to track the values
@keras.saving.register_keras_serializable()

#Inherit from the keras F1Score class
class f1_score(keras.metrics.F1Score):
    def __init__(self, average=None, threshold=None, name="f1_score", dtype=None):
        super().__init__(average, threshold, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        #get y_true and y_pred into correct shape for keras F1Score
        #get the values into a 2d matrix 
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1,1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1,1])

        #update the F1score utilizing the newly transformed y_true and y_pred values
        return super().update_state(y_true, y_pred)

@keras.saving.register_keras_serializable()
#implement a custom loss function inheriting from Keras Loss class
class BCE_DICE(keras.losses.Loss):

    def __init__(self, threshold=0.5,smoothing=1,name='BCE_DICE', **kwargs):
        #initilize the name and any other needed args for super class 
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
        #Calulate the binary intersection along all channels within the image
        intersection = k.sum(y_true * y_pred, axis=[1,2,3])
        union = k.sum(y_true, axis=[1,2,3]) + k.sum(y_pred, axis=[1,2,3])
        dice_loss = 1 - (2 * intersection + self.smoothing)/(union + self.smoothing)  
        return bce_loss + dice_loss
    
    def get_config(self):
        config = super().get_config() 
        config.update({'threshold': self.threshold, "smoothing": self.smoothing})
        return config
    

class UNET:
    #used to initilize the model
    def __init__(self, img_size=(512,512), dropout_value=0.1, add_dropout = True):
        self.img_size = img_size
        self.dropout_value = dropout_value
        self.add_dropout = add_dropout
        self.model = None

    #convolution block will be used to perform convolutions to extract features
    #batch normalization will introduce stability to the model and keep input values consistent
    #LeakyReLU will ensure all values are pushed to be 0 or above
    def conv_block(self, inputs, num_filters):
        x = keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)

        x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization(axis=-1, momentum=0.9)(x)
        x = keras.layers.LeakyReLU(negative_slope=0.01) (x)
        return x

    #encoder block will utilize convolutions and then utilize pooling to downsample the image to twice as small
    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters=num_filters)
        p = keras.layers.MaxPool2D((2,2))(x)
        return x, p
    #decoder block will first upsample the image to be twice as large then will be concatenated to skip image
    #The concatenation will be matched along the last channel
    #then perform convolutions with the new upsampled images
    def decoder_block(self, inputs, skip, num_filters):
        x = keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
        x = keras.layers.Concatenate(axis=-1)([x, skip])
        x = self.conv_block(x, num_filters=num_filters)
        return x


    def build_unet(self, input_shape):
        #set the input shape for the model
        inputs = keras.layers.Input(input_shape)
        #Encoding section, progressively shrink the image and expand number of features extracted
        #s_ variables are skip blocks (i.e. unpooled image, just raw feature extraction)
        #p_ variable are the pooled image, the shrunken result and map
        s1, p1 = self.encoder_block(inputs, 32)
        s2, p2 = self.encoder_block(p1, 64)
        s3, p3 = self.encoder_block(p2, 128)
        s4, p4 = self.encoder_block(p3, 256)

        #a Bridge block, avoids pooling and shrinking image further
        b1 = self.conv_block(p4, 512)

        #if intiilized with dropout implement dropout layer here
        if self.add_dropout:
            #spatial Dropout 2D removes feature maps
            b1 = keras.layers.SpatialDropout2D(self.dropout_value)(b1)

        #d_ represents the new reconstructed image that will be returned from the decoding block
        d4 = self.decoder_block(b1, s4, 256)
        d3 = self.decoder_block(d4, s3, 128)
        d2 = self.decoder_block(d3, s2, 64)
        d1 = self.decoder_block(d2, s1, 32)
    
        if self.add_dropout:
            d1 = keras.layers.SpatialDropout2D(self.dropout_value)(d1)

        #Push all values in all channels into a range of 0-1 (i.e. create probabilities)
        outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", dtype = 'float32')(d1)
        #Instantiate the keras model
        model = keras.models.Model(inputs, outputs, name = "UNET")
        self.model = model


    def compile(self, lr=1e-4, wd=1e-4):
        #compile the model utilizing custom metrics and loss, along with keras metrics
        self.model.compile(optimizer=keras.optimizers.AdamW(lr, wd),
                           loss= BCE_DICE(threshold=0.5),
                           metrics=[keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
                                    keras.metrics.BinaryIoU(target_class_ids=(0,1), threshold=0.5, name='BIoU'),
                                    keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='Disease_BIoU'), 
                                    f1_score('macro',0.5,'MACRO_f1_score'), 
                                    keras.metrics.Precision(name='Precision', thresholds=0.5)
                                    ,keras.metrics.Recall(name='Recall', thresholds=0.5),],
                           jit_compile=False)
    #used for inference
    def predict(self, image, threshold=0.7):
        #in case image is gray utilize while loop to add axis's until in a shape the model can utilize
        while len(image.shape) < 4:
              image = np.expand_dims(image, axis=0)

        probs = self.model.predict(image, verbose=0)
        #any probabilities above a set threshold are considered diseased and will be converted to a pure white pixel
        binary_mask = (probs > threshold).astype(np.uint8)
        #remove the batch dimension from the binary mask
        return np.squeeze(binary_mask, axis=0)
    
    def fit(self, train_ds, val_ds, epochs=20):
            #Track the most important metrics and losses for keras callback functions.
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
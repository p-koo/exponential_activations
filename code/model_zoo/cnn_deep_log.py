import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tfomics import layers, utils

def model(activation='log_relu', l2_norm=True):

    def l2_reg(weight_matrix):
        return 0.1 * K.sum(K.square(weight_matrix))

    if l2_norm:
        l2_first = l2_reg
    else:
        l2_first = None

    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    # layer 1
    nn = keras.layers.Conv1D(filters=32,
                             kernel_size=19,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             kernel_regularizer=l2_first, 
                             )(inputs)        
    nn = keras.layers.BatchNormalization()(nn)
    activation = utils.activation_fn(activation)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=48, 
                           kernel_size=7,   #176
                           padding='same', 
                           activation='relu', 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = layers.conv_layer(nn,
                           num_filters=96, 
                           kernel_size=7,     # 44
                           padding='valid', 
                           activation='relu', 
                           dropout=0.3,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = layers.conv_layer(nn,
                           num_filters=128, 
                           kernel_size=3,   # 9
                           padding='valid', 
                           activation='relu', 
                           dropout=0.4,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=3,  # 3
                                strides=3, 
                                padding='same'
                                )(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=512, activation='relu', 
                            dropout=0.5, l2=1e-6, bn=True)

    # Output layer
    logits = keras.layers.Dense(12, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


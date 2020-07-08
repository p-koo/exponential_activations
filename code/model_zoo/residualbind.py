import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tfomics import layers, utils

def model(activation='relu'):

    # input layer
    inputs = keras.layers.Input(shape=(400,4))

    # layer 1
    activation = utils.activation_fn(activation)
    nn = layers.conv_layer(inputs,
                           num_filters=24, 
                           kernel_size=19, 
                           padding='same', 
                           activation=activation, 
                           dropout=0.1,
                           l2=1e-6, 
                           bn=True)

    # layer 2
    nn = layers.residual_block(nn, filter_size=5, activation='relu', l2=1e-6)
    nn = keras.layers.MaxPool1D(pool_size=10)(nn)

    # layer 3
    nn = layers.conv_layer(nn,
                           num_filters=48, 
                           kernel_size=7, 
                           padding='same', 
                           activation='relu', 
                           dropout=0.3,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=5)(nn)

    # layer 4
    nn = layers.conv_layer(nn,
                           num_filters=64, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu', 
                           dropout=0.4,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=96, activation='relu', dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

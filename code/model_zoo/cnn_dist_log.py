import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tfomics import layers, utils

def model(activation='log_relu', l2_norm=True):

    def l2_reg(weight_matrix):
        return 0.1 * K.sum(K.square(weight_matrix))


    l2 = 1e-6
    bn = True

    dropout_block0 = 0.1
    dropout_block1 = 0.2
    dropout_block2 = 0.3
    dropout_block3 = 0.4
    dropout_block4 = 0.5
         
      

    if l2_norm:
        l2_first = l2_reg
    else:
        l2_first = None
    # input layer
    inputs = keras.layers.Input(shape=(200,4))
    activation = utils.activation_fn(activation)

    # block 1
    nn = keras.layers.Conv1D(filters=24,
                             kernel_size=19,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             #kernel_initializer=keras.initializers.RandomNormal(mean=0.1, stddev=0.05),
                             kernel_regularizer=l2_first, 
                             )(inputs)        
    nn = keras.layers.BatchNormalization()(nn)
    activation = utils.activation_fn(activation)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(dropout_block0)(nn)

    nn = layers.conv_layer(nn,
                           num_filters=32, 
                           kernel_size=7, 
                           padding='same', 
                           activation='relu', 
                           dropout=dropout_block1,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=4, 
                                strides=4, 
                                padding='same'
                                )(nn)

    nn = layers.conv_layer(nn,
                           num_filters=48, 
                           kernel_size=7, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_block2,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=4, 
                                strides=4, 
                                padding='same'
                                )(nn)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=64, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_block3,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(nn)

    # Fully-connected NN
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=96, activation='relu', dropout=dropout_block4, l2=l2, bn=bn)
    #nn = layers.dense_layer(nn, num_units=512, activation='relu', dropout=dropout_block4, l2=l2, bn=bn)

    # Output layer - additive + learned non-linearity
    logits = keras.layers.Dense(1, activation='linear', use_bias=True,  
                                 kernel_initializer='glorot_normal',
                                 bias_initializer='zeros')(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

"""import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import layers, utils

def model(activation='relu',
          dropout=True,
          l2=True,
          bn=True,
          gauss_noise=False):

    if l2:
        l2 = 1e-6

    if dropout:
        dropout_block0 = 0.1
        dropout_block1 = 0.5
    else:
        dropout_block0 = None
        dropout_block1 = None
      

    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    if gauss_noise:
        nn = keras.layers.GaussianNoise(0.2)(nn)

    # block 1
    nn = layers.conv_layer(inputs,
                           num_filters=24, 
                           kernel_size=7, 
                           padding='same', 
                           activation=activation, 
                           dropout=dropout_block0,
                           l2=l2, 
                           bn=bn)
    nn = layers.conv_layer(nn,
                           num_filters=32, 
                           kernel_size=6, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_block0,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(nn)

    nn = layers.conv_layer(nn,
                           num_filters=48, 
                           kernel_size=6, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_block0,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=4, 
                                strides=4, 
                                padding='same'
                                )(nn)

    nn = layers.conv_layer(nn,
                           num_filters=64, 
                           kernel_size=4, 
                           padding='valid', 
                           activation='relu', 
                           dropout=dropout_block0,
                           l2=l2, 
                           bn=bn)
    nn = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(nn)

    # Fully-connected NN
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=96, activation='relu', dropout=dropout_block1, l2=l2, bn=bn)

    # Output layer - additive + learned non-linearity
    logits = keras.layers.Dense(1, activation='linear', use_bias=True,  
                                 kernel_initializer='glorot_normal',
                                 bias_initializer='zeros')(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Layer


def dense_layer(input_layer, num_units, activation, dropout=0.5, l2=None, bn=True, kernel_initializer=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    nn = keras.layers.Dense(num_units, 
                            activation=None, 
                            use_bias=False,  
                            kernel_initializer=kernel_initializer,
                            bias_initializer='zeros', 
                            kernel_regularizer=l2, 
                            bias_regularizer=None,
                            activity_regularizer=None, 
                            kernel_constraint=None, 
                            bias_constraint=None)(input_layer)
    if bn:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    if dropout:
        nn = keras.layers.Dropout(dropout)(nn)
        
    return nn


def conv_layer(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2=None, bn=True, kernel_initializer=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=kernel_size,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding=padding,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2, 
                             bias_regularizer=None, 
                             activity_regularizer=None,
                             kernel_constraint=None, 
                             bias_constraint=None,
                             )(inputs)        
    if bn:                      
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    if dropout:
        nn = keras.layers.Dropout(dropout)(nn)
    return nn


    
def residual_block(input_layer, filter_size, activation='relu', l2=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    num_filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=filter_size,
                             strides=1,
                             activation='relu',
                             use_bias=False,
                             padding='same',
                             dilation_rate=1,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2
                             )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=filter_size,
                             strides=1,
                             activation='relu',
                             use_bias=False,
                             padding='same',
                             dilation_rate=1,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2
                             )(nn) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)

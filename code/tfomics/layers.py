
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from keras.layers import Layer


def dense_layer(input_layer, num_units, activation, dropout=0.5, l2=None, bn=True):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    nn = keras.layers.Dense(num_units, 
                            activation=None, 
                            use_bias=False,  
                            kernel_initializer='he_normal',
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


def conv_layer(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2=None, bn=True):
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
                             kernel_initializer='he_normal',
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

    num_filters = inputs.shape.as_list()[-1]  

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


def dilated_residual_block(input_layer, filter_size, num_dilation=2, activation='relu', l2=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    num_filters = inputs.shape.as_list()[-1]  

    dilation_rate = 1
    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=filter_size,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding='same',
                             dilation_rate=dilation_rate,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2
                             )(input_layer) 
    for i in range(num_dilation):
        dilation_rate *= 2
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation(activation)(nn)
        nn = keras.layers.Conv1D(filters=num_filters,
                                 kernel_size=filter_size,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2
                                 )(nn) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)



def dense_residual_block(input_layer, num_units, activation='relu', l2=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    num_units = inputs.shape.as_list()[-1]  

    nn = keras.layers.Dense(num_units,
                           activation=None,
                           use_bias=False,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2
                           )(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dense(num_units,
                           activation=None,
                           use_bias=False,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2
                           )(nn) 
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nnn)


class SeparableNonlinearDense(tf.keras.Model):

  def __init__(self, units=32):
    super(SeparableNonlinearDense, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.shape = input_shape
    self.w1 = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='he_normal',
                             trainable=True)
    self.w2 = self.add_weight(shape=(self.units, input_shape[-1]),
                             initializer='he_normal',
                             trainable=True)

  def call(self, input_tensor):
    return tf.matmul(tf.nn.relu(tf.matmul(input_tensor, self.w1)), self.w2)



class IndexLayer(tf.keras.Model):

  def __init__(self, index=0):
    super(IndexLayer, self).__init__()
    self.index = index

  def call(self, input_tensor):
    return tf.expand_dims(input_tensor[:, self.index], axis=1)

 

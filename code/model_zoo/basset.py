from tensorflow import keras
from tfomics import layers, utils


def model(activation='relu'):

    # input layer
    inputs = keras.layers.Input(shape=(600,4))
    
    activation = utils.activation_fn(activation)
    
    # layer 1
    nn = layers.conv_layer(inputs,
                           num_filters=300, 
                           kernel_size=19,   # 192
                           padding='same', 
                           activation=activation, 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=200, 
                           kernel_size=11,  # 56
                           padding='valid', 
                           activation='relu', 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = layers.conv_layer(nn,
                           num_filters=200, 
                           kernel_size=7,  # 56
                           padding='valid', 
                           activation='relu', 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=1000, activation='relu', dropout=0.5, l2=1e-6, bn=True)

    # layer 5
    nn = layers.dense_layer(nn, num_units=1000, activation='relu', dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(164, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


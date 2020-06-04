from tensorflow import keras
from tfomics import layers, utils


def model(pool_size=[25, 4], activation='relu', input_shape=200):

    if input_shape == 1000:
        multiplier = 2
    else:
        multiplier = 1
        
    # input layer
    inputs = keras.layers.Input(shape=(input_shape,4))

    # layer 1
    activation = utils.activation_fn(activation)
    nn = layers.conv_layer(inputs,
                           num_filters=32*multiplier, 
                           kernel_size=19, 
                           padding='same', 
                           activation=activation, 
                           dropout=0.1,
                           l2=1e-6,
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=pool_size[0], 
                                strides=pool_size[0], 
                                padding='same'
                                )(nn)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=124*multiplier, 
                           kernel_size=5, 
                           padding='same', 
                           activation='relu', 
                           dropout=0.1,
                           l2=1e-6,
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=pool_size[1], 
                                 strides=pool_size[1], 
                                 padding='same'
                                 )(nn)

    # layer 3
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=512*multiplier, activation='relu', 
                            dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(12, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


"""


"""

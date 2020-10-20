from tensorflow import keras
from tfomics import layers, utils


def model(activation='relu', input_shape=200, initialization=None):


    if not initialization:
        initialization == 'glorot_uniform'

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
                           kernel_size=19,  #200
                           padding='same', 
                           activation=activation, 
                           kernel_initializer=initialization,
                           dropout=0.1,
                           l2=1e-6, 
                           bn=True)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=48*multiplier, 
                           kernel_size=7,   #176
                           padding='same', 
                           activation='relu', 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = layers.conv_layer(nn,
                           num_filters=96*multiplier, 
                           kernel_size=7,     # 44
                           padding='valid', 
                           activation='relu', 
                           dropout=0.3,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = layers.conv_layer(nn,
                           num_filters=128*multiplier, 
                           kernel_size=3,   # 9
                           padding='valid', 
                           activation='relu', 
                           dropout=0.4,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=512*multiplier, activation='relu', 
                            dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(12, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

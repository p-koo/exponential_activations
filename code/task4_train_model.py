import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import helper
from tfomics import utils, explain, metrics
from model_zoo import basset

#------------------------------------------------------------------------------------------------

# save path
results_path = utils.make_directory('../results', 'task4')
params_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters')

#------------------------------------------------------------------------------------------------

# load dataset
data_path = '../data/er.h5'
data = helper.load_basset_dataset(data_path, reverse_compliment=True)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

#------------------------------------------------------------------------------------------------


file_path = os.path.join(results_path, 'task4_classification_performance.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    for activation in ['relu', 'exponential']:
        keras.backend.clear_session()
        
        # load model
        model = basset.model(activation)
        name = 'basset_'+activation

        # compile model
        helper.compile_model(model)

        # setup callbacks
        callbacks = helper.get_callbacks(monitor='val_auroc', patience=20, 
                                  decay_patience=5, decay_factor=0.2)

        # fit model
        history = model.fit(x_train, y_train, 
                            epochs=100,
                            batch_size=100, 
                            shuffle=True,
                            class_weight=None, 
                            validation_data=(x_valid, y_valid), 
                            callbacks=callbacks)

        # save model
        weights_path = os.path.join(params_path, name+'.hdf5')
        model.save_weights(weights_path)

        # get filter representations
        intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
        fmap = intermediate.predict(x_test)
        W = explain.activation_pwm(fmap, x_test, threshold=0.5, window=20)
                  
        # clip filters about motif to reduce false-positive Tomtom matches 
        W_clipped = utils.clip_filters(W, threshold=0.5, pad=3)
        output_file = os.path.join(save_path, name+'.meme')
        utils.meme_generate(W_clipped, output_file) 

        # write results to file
        predictions = model.predict(x_test)                
        mean_vals, std_vals = metrics.calculate_metrics(y_test, predictions, 'binary')
        f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(name, 
                                                  mean_vals[1],
                                                  std_vals[1], 
                                                  mean_vals[2],
                                                  std_vals[2]))

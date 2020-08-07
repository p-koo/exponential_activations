import os
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
from tensorflow import keras
import helper
from tfomics import utils, explain, metrics

#------------------------------------------------------------------------------------------------

num_trials = 10
model_names = ['cnn-deep', 'cnn-2', 'cnn-50']
activations = ['relu', 'exponential', 'sigmoid', 'tanh', 'softplus', 'linear', 'elu',
               'shift_scale_relu', 'shift_scale_tanh', 'shift_scale_sigmoid', 'exp_relu']

# save path
results_path = utils.make_directory('../results', 'task1')

#------------------------------------------------------------------------------------------------

# load dataset
data_path = '../data/synthetic_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# save results to file
file_path = os.path.join(results_path, 'task1_classification_performance.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    for model_name in model_names:
        results[model_name] = {}
        for activation in activations:
            trial_history = []
            for trial in range(num_trials):
                keras.backend.clear_session()
                
                # load model
                model = helper.load_model(model_name, 
                                                activation=activation, 
                                                input_shape=200)
                name = model_name+'_'+activation+'_'+str(trial)
                print('model: ' + name)

                # compile model
                helper.compile_model(model)

                # setup callbacks
                callbacks = helper.get_callbacks(monitor='val_aupr', patience=20, 
                                          decay_patience=5, decay_factor=0.2)

                # fit model
                history = model.fit(x_train, y_train, 
                                    epochs=100,
                                    batch_size=100, 
                                    shuffle=True,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=callbacks)
                trial_history.append(history)
            results[model_name][activation] = history

# pickle results
file_path = os.path.join(results_path, "task1_history.pickle")
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)


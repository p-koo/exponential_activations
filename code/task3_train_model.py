import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from tensorflow import keras
import helper
from tfomics import utils, metrics

#------------------------------------------------------------------------

num_trials = 10
model_names = ['cnn-dist', 'cnn-local']
activations = ['relu', 'exponential', 'sigmoid', 'tanh', 'softplus', 'linear', 'elu',
               'shift_scale_relu', 'shift_scale_tanh', 'shift_scale_sigmoid', 'exp_relu']

results_path = utils.make_directory('../results', 'task3')
params_path = utils.make_directory(results_path, 'model_params')

#------------------------------------------------------------------------

# load data
data_path = '../data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

#------------------------------------------------------------------------

with open(os.path.join(results_path, 'task3_classification_performance.tsv'), 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    for model_name in model_names:
        for activation in activations:
            base_name = model_name+'_'+activation
            print(base_name)
            results[base_name] = {}
            
            trial_roc_mean = []
            trial_roc_std = []
            trial_pr_mean = []
            trial_pr_std = []
            for trial in range(num_trials):
                keras.backend.clear_session()
                
                # load model
                model = helper.load_model(model_name, activation=activation)
                name = base_name+'_'+str(trial)
                print('model: ' + name)

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
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=callbacks)

                # save model
                weights_path = os.path.join(params_path, name+'.hdf5')
                model.save_weights(weights_path)

                # predict test sequences and calculate performance metrics
                predictions = model.predict(x_test)                
                mean_vals, std_vals = metrics.calculate_metrics(y_test, predictions, 'binary')

                trial_roc_mean.append(mean_vals[1])
                trial_roc_std.append(std_vals[1])
                trial_pr_mean.append(mean_vals[2])
                trial_pr_std.append(std_vals[2])

            results[base_name] = [np.array(trial_roc_mean), np.array(trial_pr_mean)]
            f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(base_name, 
                                                      np.mean(trial_roc_mean),
                                                      np.std(trial_roc_mean), 
                                                      np.mean(trial_pr_mean),
                                                      np.std(trial_pr_mean)))

# save results
file_path = os.path.join(results_path, 'task3_performance_results.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)



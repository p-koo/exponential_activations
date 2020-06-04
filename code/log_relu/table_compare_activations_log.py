import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import helper
from tfomics import utils, explain, metrics

import cnn_deep_log

#------------------------------------------------------------------------------------------------

num_trials = 10

# save path
results_path = utils.make_directory('../results', 'log')
params_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters')

#------------------------------------------------------------------------------------------------

activations = ['log_relu', 'relu']
l2_norm = [True, False]
model_name = 'cnn-deep'


#------------------------------------------------------------------------------------------------

# load dataset
data_path = '../data/synthetic_dataset.h5'
data = helper.load_dataset(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# save results to file
save_path = os.path.join(results_path, 'performance_activation_comparison_other_ablation.tsv')
with open(save_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    for activation in ['log_relu', 'relu']:
        for l2_norm in [True, False]:
            trial_roc_mean = []
            trial_roc_std = []
            trial_pr_mean = []
            trial_pr_std = []
            for trial in range(num_trials):
                keras.backend.clear_session()

                # load model
                model = cnn_deep_log.model(activation, l2_norm)

                name = model_name+'_'+activation

                if l2_norm:
                    name = name + '_l2'

                # set up optimizer and metrics
                auroc = keras.metrics.AUC(curve='ROC', name='auroc')
                aupr = keras.metrics.AUC(curve='PR', name='aupr')
                optimizer = keras.optimizers.Adam(learning_rate=0.001)
                loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=['accuracy', auroc, aupr])
                           
                # load trained weights
                weights_path = os.path.join(params_path, name+'_'+str(trial)+'.hdf5')
                model.load_weights(weights_path)

                # predict test sequences and calculate performance metrics
                predictions = model.predict(x_test)                
                mean_vals, std_vals = metrics.calculate_metrics(y_test, predictions, 'binary')

                trial_roc_mean.append(mean_vals[1])
                trial_roc_std.append(std_vals[1])
                trial_pr_mean.append(mean_vals[2])
                trial_pr_std.append(std_vals[2])

            f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(name, 
                                                      np.mean(trial_roc_mean),
                                                      np.std(trial_roc_mean), 
                                                      np.mean(trial_pr_mean),
                                                      np.std(trial_pr_mean)))


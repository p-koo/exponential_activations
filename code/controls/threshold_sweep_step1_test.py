import os
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
from tensorflow import keras
import helper
from tfomics import utils, explain, metrics

#------------------------------------------------------------------------------------------------

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

num_trials = 10
model_names = ['cnn-deep']
activations = ['relu', 'exponential']

# save path
results_path = utils.make_directory('../../results', 'task1')
params_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters_threshold_sweep')

#------------------------------------------------------------------------------------------------

# load dataset
data_path = '../../data/synthetic_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data


results = {}
for model_name in model_names:
    results[model_name] = {}
    for activation in activations:
        trial_roc_mean = []
        trial_roc_std = []
        trial_pr_mean = []
        trial_pr_std = []
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

            # save model
            weights_path = os.path.join(params_path, name+'.hdf5')
            model.load_weights(weights_path)
                     
            for threshold in thresholds:

                # get 1st convolution layer filters
                fig, W, logo = explain.plot_filers(model, x_test, layer=3, threshold=threshold, 
                                                   window=20, num_cols=8, figsize=(30,5))
                outfile = os.path.join(save_path, name+'_'+str(threshold)+'_'+str(trial)+'.pdf')
                fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
                plt.close()

                # clip filters about motif to reduce false-positive Tomtom matches 
                W_clipped = utils.clip_filters(W, threshold=0.5, pad=3)
                output_file = os.path.join(save_path, name+'_'+str(threshold)+'_'+str(trial)+'.meme')
                utils.meme_generate(W_clipped, output_file) 


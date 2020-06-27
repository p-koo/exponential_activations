import os
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
from tensorflow import keras
import helper
from tfomics import utils, explain, metrics

from model_zoo import cnn_deep_log

#------------------------------------------------------------------------------------------------

num_trials = 10

# save path
results_path = utils.make_directory('../results', 'task1')
params_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters')

#------------------------------------------------------------------------------------------------

activations = ['log_relu', 'relu']
l2_norm = [True, False]
model_name = 'cnn-deep'

# load dataset
data_path = '../data/synthetic_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# save results to file
file_path = os.path.join(results_path, 'task1_classification_performance_log.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    results[model_name] = {}
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
                name = name + '_' + str(trial)
                print('model: ' + name)

                # set up optimizer and metrics
                auroc = keras.metrics.AUC(curve='ROC', name='auroc')
                aupr = keras.metrics.AUC(curve='PR', name='aupr')
                optimizer = keras.optimizers.Adam(learning_rate=0.001)
                loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=['accuracy', auroc, aupr])


                es_callback = keras.callbacks.EarlyStopping(monitor='val_auroc', #'val_aupr',#
                                                            patience=20, 
                                                            verbose=1, 
                                                            mode='max', 
                                                            restore_best_weights=False)
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auroc', 
                                                              factor=0.2,
                                                              patience=5, 
                                                              min_lr=1e-7,
                                                              mode='max',
                                                              verbose=1) 

                history = model.fit(x_train, y_train, 
                                    epochs=100,
                                    batch_size=100, 
                                    shuffle=True,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[es_callback, reduce_lr])

                # save model
                weights_path = os.path.join(params_path, name+'.hdf5')
                model.save_weights(weights_path)
                          
                # get 1st convolution layer filters
                layer = 3
                fig, W, logo = explain.plot_filers(model, x_test, layer=layer, threshold=0.5, 
                                                   window=20, num_cols=8, figsize=(30,5))
                outfile = os.path.join(save_path, name+'.pdf')
                fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
                plt.close()
                
                # clip filters about motif to reduce false-positive Tomtom matches 
                W_clipped = utils.clip_filters(W, threshold=0.5, pad=3)
                output_file = os.path.join(save_path, name+'.meme')
                utils.meme_generate(W_clipped, output_file) 

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
            results[model_name][activation] = np.array([trial_roc_mean, trial_pr_mean])


# pickle results
file_path = os.path.join(results_path, "task1_performance_results_log.pickle")
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)


import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from tensorflow import keras
import helper
from tfomics import utils, metrics, explain

from model_zoo import cnn_dist_log

#------------------------------------------------------------------------

activations = ['log_relu', 'relu']
l2_norm = [True, False]
model_name = 'cnn-dist'

num_trials = 10
results_path = utils.make_directory('../results', 'task3')
params_path = utils.make_directory(results_path, 'model_params')

#------------------------------------------------------------------------

# load dat3
data_path = '../data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]
X_model = test_model[true_index][:500]

#------------------------------------------------------------------------

file_path = os.path.join(results_path, 'task3_classification_performance_log.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    for activation in ['log_relu', 'relu']:
        for l2_norm in [True, False]:
            
            trial_roc_mean = []
            trial_roc_std = []
            trial_pr_mean = []
            trial_pr_std = []
            saliency_scores = []
            for trial in range(num_trials):
                keras.backend.clear_session()
                    
                # load model
                model = cnn_dist_log.model(activation, l2_norm)

                base_name = model_name+'_'+activation

                if l2_norm:
                    base_name = base_name + '_l2'
                name = base_name + '_' + str(trial)
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
                if (activation == 'relu') & (l2_norm == False):

                    # save model
                    weights_path = os.path.join(params_path, name+'.hdf5')
                    model.load_weights(weights_path)
                    
                else:
                    history = model.fit(x_train, y_train, 
                                        epochs=100,
                                        batch_size=100, 
                                        shuffle=True,
                                        validation_data=(x_valid, y_valid), 
                                        callbacks=[es_callback, reduce_lr])

                    # save model
                    weights_path = os.path.join(params_path, name+'.hdf5')
                    model.save_weights(weights_path)

                # interpretability performance 
                saliency_scores.append(explain.saliency(model, X, class_index=0, layer=-1))

                # predict test sequences and calculate performance metrics
                predictions = model.predict(x_test)                
                mean_vals, std_vals = metrics.calculate_metrics(y_test, predictions, 'binary')

                trial_roc_mean.append(mean_vals[1])
                trial_roc_std.append(std_vals[1])
                trial_pr_mean.append(mean_vals[2])
                trial_pr_std.append(std_vals[2])

            f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(base_name, 
                                                      np.mean(trial_roc_mean),
                                                      np.std(trial_roc_mean), 
                                                      np.mean(trial_pr_mean),
                                                      np.std(trial_pr_mean)))

            results[name] = saliency_scores

# save results
file_path = os.path.join(results_path, 'task3_performance_results_log.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

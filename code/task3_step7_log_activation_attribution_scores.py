import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
import helper
from tfomics import utils, explain

from model_zoo import cnn_dist_log
#------------------------------------------------------------------------

activations = ['log_relu', 'relu']
l2_norm = [True, False]
model_name = 'cnn-dist'

num_trials = 10
results_path = os.path.join('../results', 'task3')
params_path = os.path.join(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'scores')

#------------------------------------------------------------------------

# load data
data_path = '../data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]
X_model = test_model[true_index][:500]

#------------------------------------------------------------------------

for activation in ['log_relu', 'relu']:
    for l2_norm in [True, False]:
        
        if (activation == 'relu') & (l2_norm == False):
            print('already did relu')
        else:
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

                # load model
                weights_path = os.path.join(params_path, name+'.hdf5')
                model.load_weights(weights_path)

                # interpretability performance 
                saliency_scores.append(explain.saliency(model, X, class_index=0, layer=-1))

            # save results
            file_path = os.path.join(save_path, base_name+'.pickle')
            with open(file_path, 'wb') as f:
                cPickle.dump(np.array(saliency_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)

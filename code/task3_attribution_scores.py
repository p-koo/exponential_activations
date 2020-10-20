import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
import helper
from tfomics import utils, explain

#------------------------------------------------------------------------

num_trials = 10
model_names = ['cnn-dist', 'cnn-local']
activations = ['relu', 'exponential', 'sigmoid', 'tanh', 'softplus', 'linear', 'elu']


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

for model_name in model_names:
    for activation in activations:
        
        saliency_scores = []
        mut_scores = []
        integrated_scores = []
        shap_scores = []
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            model = helper.load_model(model_name, activation=activation)
            name = model_name+'_'+activation+'_'+str(trial)
            print('model: ' + name)

            # compile model
            helper.compile_model(model)

            # load model
            weights_path = os.path.join(params_path, name+'.hdf5')
            model.load_weights(weights_path)

            # interpretability performance with saliency maps
            print('saliency maps')
            saliency_scores.append(explain.saliency(model, X, class_index=0, layer=-1))

            # interpretability performance with mutagenesis 
            print('mutagenesis maps')
            mut_scores.append(explain.mutagenesis(model, X, class_index=0, layer=-1))

            # interpretability performance with integrated gradients
            print('integrated gradients maps')
            integrated_scores.append(explain.integrated_grad(model, X, class_index=0, layer=-1,
                                                        num_background=10, num_steps=20,
                                                        reference='shuffle'))

            # interpretability performance with deepshap 
            print('shap maps')
            shap_scores.append(explain.deepshap(model, X, class_index=0, 
                                           num_background=10, reference='shuffle'))

        # save results
        file_path = os.path.join(save_path, model_name+'_'+activation+'.pickle')
        with open(file_path, 'wb') as f:
            cPickle.dump(np.array(saliency_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(np.array(mut_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(np.array(integrated_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(np.array(shap_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)

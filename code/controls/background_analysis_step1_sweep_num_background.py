import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
import helper
from tfomics import utils, explain

#------------------------------------------------------------------------

num_trials = 10
model_name = 'cnn-dist'
activations = ['relu', 'exponential']

results_path = os.path.join('../../results/', 'task3')
params_path = os.path.join(results_path, 'model_params')

#------------------------------------------------------------------------

# load data
data_path = '../../data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]
X_model = test_model[true_index][:500]

#------------------------------------------------------------------------

num_backgrounds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

results = {}
for activation in activations:
    results[activation] = {}

    results[activation]['deepshap'] = []
    results[activation]['integratedgrad'] = []
    for num_background in num_backgrounds:

        shap_roc_scores = []
        shap_pr_scores = []
        integrated_roc_scores = []
        integrated_pr_scores = []
        for trial in range(num_trials):
            keras.backend.clear_session()
            
            # load model
            model, name = helper.load_model(model_name, activation=activation)
            name = name+'_'+activation+'_'+str(trial)

            # compile model
            helper.compile_model(model)

            # load model
            weights_path = os.path.join(params_path, name+'.hdf5')
            model.load_weights(weights_path)

            print('model: ' + name +'_'+str(num_background))

            # interpretability performance with deepshap
            scores = explain.deepshap(model, X, class_index=0, layer=-1, 
                                     num_background=num_background, reference='shuffle')
            scores *= X
            roc_score, pr_score = helper.interpretability_performance(X, scores, X_model)
            shap_roc_scores.append(np.mean(roc_score))
            shap_pr_scores.append(np.mean(pr_score))    
            print('DeepShap: %.4f+/-%.4f\t'%(np.mean(roc_score), np.std(roc_score))) 
            print('DeepShap: %.4f+/-%.4f\t'%(np.mean(pr_score), np.std(pr_score))) 


            # interpretability performance with integrated gradients
            scores = explain.integrated_grad(model, X, class_index=0, layer=-1, 
                                             num_background=num_background, num_steps=20, 
                                             reference='shuffle')
            scores *= X
            roc_score, pr_score = helper.interpretability_performance(X, scores, X_model)
            integrated_roc_scores.append(np.mean(roc_score))
            integrated_pr_scores.append(np.mean(pr_score))               
            print('Integrated gradients: %.4f+/-%.4f\t'%(np.mean(roc_score), np.std(roc_score))) 
            print('Integrated gradients: %.4f+/-%.4f\t'%(np.mean(pr_score), np.std(pr_score))) 


        results[activation]['integratedgrad'].append([np.array(integrated_roc_scores), np.array(integrated_pr_scores)])
        results[activation]['deepshap'].append([np.array(shap_roc_scores), np.array(shap_pr_scores)])

    results[activation]['integratedgrad'] = np.array(results[activation]['integratedgrad'])
    results[activation]['deepshap'] = np.array(results[activation]['deepshap'])
    
# save results
file_path = os.path.join(results_path, 'num_background_sweep.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

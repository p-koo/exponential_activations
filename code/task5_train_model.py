import os
import numpy as np
from six.moves import cPickle
from tensorflow import keras
from tensorflow import keras
import helper
from tfomics import utils, metrics, explain

#------------------------------------------------------------------------

model_names = ['residualbind'] 
activations = ['exponential', 'relu']

results_path = utils.make_directory('../results', 'task5')
params_path = utils.make_directory(results_path, 'model_params')

#------------------------------------------------------------------------

file_path = '../data/ZBED2_400_h3k27ac.h5' 
data = helper.load_data(file_path, reverse_compliment=True)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

#------------------------------------------------------------------------

file_path = os.path.join(results_path, 'task5_classification_performance.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    for model_name in model_names:
        for activation in activations:
            keras.backend.clear_session()
            
            # load model
            model = helper.load_model(model_name, activation=activation)
            name = model_name+'_'+activation
            print('model: ' + name)

            # compile model
            helper.compile_model(model)

            # setup callbacks
            callbacks = helper.get_callbacks(monitor='val_auroc', patience=20, 
                                      decay_patience=5, decay_factor=0.2)

            # train model
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

            # print results to file
            f.write("%s\t%.3f\t%.3f\n"%(name, mean_vals[1], mean_vals[2]))

            # calculate saliency on a subset of data 
            true_index = np.where(y_test[:,0] == 1)[0]
            X = x_test[true_index][:500]
            results[name] = explain.saliency(model, X, class_index=0, layer=-1)

# save results
file_path = os.path.join(results_path, 'task5_saliency_results.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

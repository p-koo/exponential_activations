import sys, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import tensorflow.compat.v1.keras.backend as K1
from tfomics import metrics, utils

#-------------------------------------------------------------------------------------------
# clean Keras fitting 
#-------------------------------------------------------------------------------------------


def fit_mse(model, x_train, y_train, validation_data=None, 
            num_epochs=200, batch_size=100, decay_factor=1.0, 
            patience=None, decay_patience=5, noise_std=None, class_weight=None,
            save_best=False, save_path=None, verbose=False, monitor='loss'):

    x_valid, y_valid = validation_data

    # setup training callbacks
    if monitor == 'loss':
        mode = 'min'
    elif monitor == 'pearsonr':
        mode = 'max'
    early_stopping = EarlyStopping(model, patience, mode, save_path)
    decay_lr = DecayLearningRate(model.optimizer.lr, 
                                 decay_rate=decay_factor, 
                                 patience=decay_patience, 
                                 min_lr=1e-7, 
                                 mode=mode)

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch %d out of %d'%(epoch, num_epochs))

        # add gaussian noise to inputs
        if noise_std:
            x = x_train + np.random.normal(scale=noise_std, size=x_train.shape)
        else:
            x = x_train

        # training epoch
        history = model.fit(x_train, y_train, 
                            epochs=1,
                            batch_size=batch_size, 
                            shuffle=True,
                            class_weight=class_weight,
                            verbose=verbose)

        predictions = model.predict(x_valid, batch_size=batch_size)
        corr = utils.pearsonr_scores(y_test, predictions)
        if verbose: 
            print('  Validation corr: %.4f+/-%.4f'%(np.mean(corr), np.std(corr)))

        if monitor == 'pearsonr':
            val = corr
        else:
            val = history.history['val_'+monitor][-1]

        if decay_patience:
            decay_lr.update(val)

        if patience:
            # check early stopping
            if early_stopping.update(val):
                break


def fit_bce(model, x_train, y_train, validation_data=None, 
            num_epochs=200, batch_size=100, decay_factor=1.0, store_metrics=False,
            patience=None, decay_patience=5, noise_std=None, class_weight=None,
            save_best=False, save_path=None, verbose=False, monitor='loss', mode='min'):

    x_valid, y_valid = validation_data

    # setup training callbacks
    if patience:
        early_stopping = EarlyStopping(model, patience, mode, save_path)
    if decay_patience:
        decay_lr = DecayLearningRate(model.optimizer.lr, 
                                     decay_rate=decay_factor, 
                                     patience=decay_patience, 
                                     min_lr=1e-7, 
                                     mode=mode)

    train_metrics = []
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch %d out of %d'%(epoch, num_epochs))

        # add gaussian noise to inputs
        if noise_std:
            x = x_train + np.random.normal(scale=noise_std, size=x_train.shape)
        else:
            x = x_train

        # training epoch
        history = model.fit(x_train, y_train, 
                            epochs=1,
                            batch_size=batch_size, 
                            shuffle=True,
                            class_weight=class_weight,
                            validation_data=validation_data,
                            verbose=verbose)

        if store_metrics:
            vals = []
            for key in history.history.keys():
                vals.append(history.history[key][-1])
            train_metrics.append(vals)


        if decay_patience:
            decay_lr.update(history.history['val_'+monitor][-1])

        if patience:
            # check early stopping
            if early_stopping.update(history.history['val_'+monitor][-1]):
                break

    return train_metrics

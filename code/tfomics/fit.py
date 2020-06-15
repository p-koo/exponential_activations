import sys, time
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from keras import backend as K
import tensorflow.compat.v1.keras.backend as K1
from tfomics import metrics, adversarial, utils

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


#-------------------------------------------------------------------------------------------
# Tensorflow custom fitting 
#-------------------------------------------------------------------------------------------


def fit_custom(model, x_train, y_train, validation_data=None, 
            num_epochs=200, batch_size=100, decay_factor=1.0, progress=True,
            patience=None, decay_patience=None, noise_std=None, class_weight=None,
            save_path=None, verbose=False, monitor='loss'):

    # setup tensorflow data
    trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    x_valid, y_valid = validation_data

    # setup object to monitor metrics
    monitor_index, mode = monitor_params(monitor)
    train_monitor = MonitorPerformance('binary')
    valid_monitor = MonitorPerformance('binary')
    num_batches = np.floor(x_train.shape[0]/batch_size)

    # setup training callbacks
    if patience:
        early_stopping = EarlyStopping(model, patience, mode, save_path, verbose)
    if decay_patience:
        decay_lr = DecayLearningRate(model.optimizer.lr, 
                                     decay_rate=decay_factor, 
                                     patience=decay_patience, 
                                     min_lr=1e-7, 
                                     mode=mode,
                                     verbose=verbose)

    # train model
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch %d out of %d'%(epoch+1, num_epochs))
        
        # add gaussian noise to inputs
        batch_dataset = trainset.shuffle(buffer_size=batch_size).batch(batch_size)

        # train loop
        epoch_start = time.time()
        pred_batch = []
        y_batch = []
        for step, (x, y) in enumerate(batch_dataset):
           # add gaussian noise to inputs
            if noise_std:
                x = add_gaussian_noise(x, stddev=noise_std)
    
            # train update on mini-batch
            loss_value, pred, gradients = train_on_batch(model, x, y)
        
            # Update the weights of our linear layer.
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            pred_batch.append(pred)
            y_batch.append(y)
            if progress:
                progress_bar(step, num_batches, np.mean(loss_value), epoch_start, bar_length=30)

        # train metrics
        predictions = np.concatenate(pred_batch)
        y = np.concatenate(y_batch)
        train_monitor.update(model.loss(y, predictions), y, predictions)
        if verbose:
            train_monitor.print_results("train")


        if any([validation_data]):

            # validation metrics        
            predictions = model.predict(x_valid, batch_size=batch_size)  
            current_loss = model.loss(y_valid, predictions)
            valid_monitor.update(current_loss, y_valid, predictions)
            if verbose:
                valid_monitor.print_results("valid")

            # check to see if decay learning rate
            mean_vals, std_vals = valid_monitor.current_metric()
            val = [current_loss, mean_vals[0], mean_vals[1], mean_vals[2]]
            if decay_patience:
                decay_lr.update(val[monitor_index])

            if patience:
                # check early stopping
                if early_stopping.update(val[monitor_index]):
                    break            

    return train_monitor, valid_monitor



def robust_custom_fit_pgd(model, x_train, y_train, validation_data=None, 
                    num_epochs=100, batch_size=100, decay_factor=1.0, 
                    patience=None, decay_patience=5, noise_std=None, class_weight=None,
                    epsilon=0.2, num_steps=10, burn_in=4,
                    save_best=False, save_path=None, verbose=False, monitor='loss'):

    # setup tensorflow data
    trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    x_valid, y_valid = validation_data

    # setup object to monitor metrics
    monitor_index, mode = monitor_params(monitor)
    train_monitor = MonitorPerformance('binary')
    valid_monitor = MonitorPerformance('binary')

    # setup training callbacks
    if patience:
        early_stopping = EarlyStopping(model, patience, mode, save_path, verbose)
    if decay_patience:
        decay_lr = DecayLearningRate(model.optimizer.lr, 
                                     decay_rate=decay_factor, 
                                     patience=decay_patience, 
                                     min_lr=1e-7, 
                                     mode=mode)

    input_var = tf.Variable(x_train[:batch_size,:,:], shape=[None, 200,4], dtype=tf.float32, trainable=True)   
    adam_updates = AdamUpdates(x_train[:batch_size,:,:].shape)

    # train model
    num_steps = 0
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch %d out of %d'%(epoch+1, num_epochs))
                
        batch_dataset = trainset.shuffle(buffer_size=batch_size).batch(batch_size)

        # train loop
        if epoch < burn_in:
            pred_batch = []
            y_batch = []
            for step, (x, y) in enumerate(batch_dataset):        
                # train update on mini-batch
                loss_value, pred, gradients = train_on_batch(model, x, y)
            
                # Update the weights of our linear layer.
                model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                pred_batch.append(pred)
                y_batch.append(y)
            predictions = np.concatenate(pred_batch)
            y = np.concatenate(y_batch)
            train_monitor.update(model.loss(y, predictions), y, predictions)
            if verbose:
                train_monitor.print_results("train")
        else:
            if np.mod(epoch, 5) == 0:
                num_steps += 1
                print(num_steps)
            pred_batch = []
            y_batch = []
            for step, (x, y) in enumerate(batch_dataset):
                adam_updates.reset(x.shape)

                # add gaussian noise to inputs
                if noise_std:
                    x = add_gaussian_noise(x, stddev=noise_std)

                # Get PGD perturbation l-infinity ball
                x_pgd = adversarial.pgd_batch(model, input_var, x, y, num_steps, epsilon, adam_updates)        
                x_new = tf.concat([x, x_pgd], axis=0)
                y_new = tf.concat([y, y], axis=0)

                # train update on mini-batch
                loss_value, pred, gradients = train_on_batch(model, x_new, y_new)
            
                # Update the weights of our linear layer.
                model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                pred_batch.append(pred)
                y_batch.append(y_new)
            # calculate training metrics
            predictions = np.concatenate(pred_batch)
            y = np.concatenate(y_batch)
            train_monitor.update(model.loss(y, predictions), y, predictions)
            if verbose:
                train_monitor.print_results("train")
    

        # validation metrics        
        predictions = model.predict(x_valid, batch_size=batch_size)  
        current_loss = model.loss(y_valid, predictions)
        valid_monitor.update(current_loss, y_valid, predictions)
        if verbose:
            valid_monitor.print_results("valid")

        # check to see if decay learning rate
        mean_vals, std_vals = valid_monitor.current_metric()
        val = [current_loss, mean_vals[0], mean_vals[1], mean_vals[2]]

        if decay_patience:
            decay_lr.update(val[monitor_index])

        if patience:
            # check early stopping
            if early_stopping.update(val[monitor_index]):
                break            
    

    return train_monitor, valid_monitor

#----------------------------------------------------------------------------------------------------
# Helper functions and classes
#----------------------------------------------------------------------------------------------------

@tf.function
def train_on_batch(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss_value = model.loss(y, predictions)
        gradients = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, predictions, gradients





def add_gaussian_noise(x, stddev=0.2):
    return x + tf.random.normal(x.shape, mean=0.0, stddev=1.0)


def monitor_params(monitor):
    if monitor == 'loss':
        monitor_index = 0
        mode = 'min'
    elif monitor == 'auroc':
        monitor_index = 2
        mode = 'max'
    elif monitor == 'aupr':
        monitor_index = 3
        mode = 'max'        
    return monitor_index, mode




def progress_bar(step, num_batches, loss, start_time, bar_length=30):
    remaining_time = (time.time() - start_time)*(num_batches-(step+1))/(step+1)
    percent = step/num_batches
    progress = '='*int(percent*bar_length)
    spaces = ' '*int(bar_length-round(percent*bar_length))
    sys.stdout.write("\r[%s] %.1f%% -- remaining time=%ds -- loss=%.5f  " \
        %(progress+spaces, percent*100, remaining_time, loss))
    if step == num_batches:
        sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%.2fs -- loss=%.5f \n" \
            %(progress+spaces, percent*100, time.time()-start_time, loss))
    sys.stdout.flush()



class MonitorPerformance():
    """helper class to monitor and store performance metrics during
       training. This class uses the metrics for early stopping. """

    def __init__(self, objective='binary'):
        self.objective = objective
        self.loss = []
        self.metric = []
        self.metric_std = []


    def _add_loss(self, loss):
        self.loss = np.append(self.loss, loss)


    def _add_metrics(self, scores):
        self.metric.append(scores[0])
        self.metric_std.append(scores[1])



    def update(self, loss, label, prediction):
        scores = metrics.calculate_metrics(label, prediction, self.objective)
        self._add_loss(loss)
        self._add_metrics(scores)


    def mean_loss(self):
        return np.mean(self.loss)


    def current_metric(self):
        return self.metric[-1], self.metric_std[-1]


    def min_loss(self):
        min_loss = min(self.loss)
        min_index = np.argmin(self.loss)
        num_loss = len(self.loss)
        return min_loss, min_index, num_loss


    def print_results(self, name):

        print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]))
        mean_vals, error_vals = self.current_metric()

        if (self.objective == "binary") | (self.objective == "categorical"):
            print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
            print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
            print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
        elif (self.objective == 'squared_error'):
            print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
            print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
            print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))


    def save_metrics(self, filepath):
        print("  saving metrics to " + filepath)
        with open(filepath, 'wb') as f:
            cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)



class EarlyStopping():
    """helper class to monitor and store performance metrics during
       training. This class uses the metrics for early stopping. """

    def __init__(self, model, patience=20, mode='min', save_path=None, verbose=1):

        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.patience = patience
        self.best_metric = 1e10
        self.status = False
        self.save_path = save_path
        self.model = model
        if self.mode == 'max':
            self.factor = -1
            self.best_metric = -1e10
        else:
            self.factor = 1
            self.best_metric = 1e10

    def update(self, metric):
        if self.factor*self.best_metric > self.factor*metric:
            self.best_metric = metric
            self.counter = 0
            if self.save_path:
                if self.verbose:
                    print(  'Saving model to: '+self.save_path)
                self.model.save_weights(self.save_path)
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.status = True

        return self.status


class DecayLearningRate():
    """helper class to monitor and store performance metrics during
       training. This class uses the metrics for early stopping. """

    def __init__(self, lr, decay_rate=0.5, patience=5, min_lr=1e-7, mode='min', verbose=True):

        self.lr = lr
        self.decay_rate = decay_rate
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        if self.mode == 'max':
            self.factor = -1
            self.best_metric = -1e10
        else:
            self.factor = 1
            self.best_metric = 1e10
            

        self.learning_rate = lr.numpy()   # learning rate
        self.counter = 0

    def update(self, metric):
        if self.factor*self.best_metric > self.factor*metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.learning_rate *= self.decay_rate
                self.learning_rate = np.maximum(self.learning_rate, self.min_lr)
                K.set_value(self.lr, self.learning_rate)
                self.counter = 0
                if self.verbose:
                    print('  Decay learning rate: %f'%self.learning_rate)


class AdamUpdates():
    """helper class to monitor and store performance metrics during
       training. This class uses the metrics for early stopping. """

    def __init__(self, shape, learning_rate=0.001, beta1=0.9, beta2=0.999):

        input_shape = list(shape)
        input_shape[0] = None
        self.v = tf.Variable(np.zeros(shape), shape=input_shape, dtype=tf.float32,  trainable=False)
        self.m = tf.Variable(np.zeros(shape), shape=input_shape, dtype=tf.float32, trainable=False)
        self.beta1 = tf.constant(beta1, dtype=tf.float32)
        self.beta2 = tf.constant(beta2, dtype=tf.float32)
        self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)

    def update(self, x, delta):
        self.m.assign(self.beta1*self.m + (1-self.beta1)*delta)
        self.v.assign(self.beta2*self.v + (1-self.beta2)*(tf.math.pow(delta,2)))
        return x + tf.math.divide(self.learning_rate*self.m, (tf.math.sqrt(self.v) + 1e-10))
        
    def reset(self, shape):
        self.m.assign(np.zeros(shape))
        self.v.assign(np.zeros(shape))





#----------------------------------------------------------------------------------------------------
# Keras fitting functions
#----------------------------------------------------------------------------------------------------

def robust_fit_fgsm(model, x_train, y_train, validation_data, num_epochs=200, 
                    lr=0.001, lr_decay=0.4, patience=5, batch_size=100, 
                    epsilon=0.1, burn_in=10, noise=None):

    # setup ops for fast gradient sign attack
    y_true = K.placeholder()
    loss = keras.losses.binary_crossentropy(y_true, model.output, from_logits=False)
    gradient = K1.gradients(loss, model.input)
    signed_grad = K.sign(gradient[0])
    placeholders = [model.inputs[0], y_true]
    sess = K1.get_session()

    best_auroc = 0
    counter = 0
    metrics = []
    for epoch in range(num_epochs):
        epsilon2 = np.random.rand()*epsilon
        print('Epoch %d out of %d'%(epoch, num_epochs))
        if epoch >= burn_in:
            if noise:
                x_noise = x_train + np.random.normal(scale=noise, size=x_train.shape)
                inputs = [x_noise, y_train]
            else:
                inputs = [x_train, y_train]

            delta = utils.run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size)
            x_adv = x_train + delta*epsilon2
            x = np.concatenate([x_train, x_adv], axis=0)
            y = np.concatenate([y_train, y_train], axis=0)
        else:
            x = x_train
            y = y_train

        # training epoch
        history = model.fit(x, y, 
                            epochs=1,
                            batch_size=batch_size, 
                            shuffle=True,
                            validation_data=validation_data)
        
        # store performance metrics
        metrics.append([history.history['loss'], 
                        history.history['auroc'], 
                        history.history['val_loss'], 
                        history.history['val_auroc']])

        if epoch >= burn_in:
            # learning rate decay
            val_auroc = history.history['val_auroc'][0]
            if best_auroc < val_auroc:
                best_auroc = val_auroc
                counter = 0
            else:
                counter += 1
                if counter == patience:
                    lr *= lr_decay
                    lr = np.maximum(lr, 1e-6)
                    print('  Decaying learning rate to: %f'%(lr))
                    K.set_value(model.optimizer.lr, lr)
                    counter = 0
        
    return metrics



def robust_fit_pgd(model, x_train, y_train, validation_data, 
                   num_epochs=200, batch_size=100,  
                   lr=0.001, lr_decay=0.4, patience=5, 
                   epsilon=0.1, burn_in=10, num_steps=10, noise=None):

    # setup ops for fast gradient sign attack
    y_true = K.placeholder()
    loss = keras.losses.binary_crossentropy(y_true, model.output, from_logits=False)
    gradient = K1.gradients(loss, model.input)
    signed_grad = K.sign(gradient[0])
    placeholders = [model.inputs[0], y_true]
    sess = K1.get_session()


    best_auroc = 0
    counter = 0
    metrics = []
    for epoch in range(num_epochs):
        print('Epoch %d out of %d'%(epoch, num_epochs))
        if epoch >= burn_in:
            if noise:
                x = x_train + np.random.normal(scale=noise, size=x_train.shape)
            else:
                x = np.copy(x_train)

            v = 0
            m = 0
            beta1 = 0.9
            beta2 = 0.999
            learning_rate = 0.001
            for i in range(num_steps):
                inputs = [x, y_train]
                delta = utils.run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size)

                # update inputs with scale signed gradients and Adam
                m = beta1*m + (1-beta1)*delta
                v = beta2*v + (1-beta2)*(delta**2)
                x += learning_rate*m / (np.sqrt(v) + 1e-10)
                # x += 0.2 /(i+10)*delta

                # project to l-infinity ball
                x = np.clip(x, x_train-epsilon, x_train+epsilon) 

            x = np.concatenate([x_train, x], axis=0)
            y = np.concatenate([y_train, y_train], axis=0)
        else:
            x = x_train
            y = y_train

        # training epoch
        history = model.fit(x, y, 
                            epochs=1,
                            batch_size=batch_size, 
                            shuffle=True,
                            validation_data=validation_data)

        # store performance metrics
        metrics.append([history.history['loss'], 
                        history.history['auroc'], 
                        history.history['val_loss'], 
                        history.history['val_auroc']])

        if epoch >= burn_in:        # learning rate decay
            val_auroc = history.history['val_auroc'][0]
            if best_auroc < val_auroc:
                best_auroc = val_auroc
                counter = 0
            else:
                counter += 1
                if counter == patience:
                    lr *= lr_decay
                    lr = np.maximum(lr, 1e-6)
                    print('  Decaying learning rate to: %f'%(lr))
                    K.set_value(model.optimizer.lr, lr)
                    counter = 0
        
    return metrics
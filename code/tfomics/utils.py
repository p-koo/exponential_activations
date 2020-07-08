import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1


def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

    
def run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size=128):
    
    def feed_dict_batch(placeholders, inputs, index):
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = inputs[i][index]
        return feed_dict
    
    N = len(inputs[0])
    num_batches = int(np.floor(N/batch_size))
    
    values = []
    for i in range(num_batches):
        index = range(i*batch_size, (i+1)*batch_size)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    if num_batches*batch_size < N:
        index = range(num_batches*batch_size, N)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    values = np.concatenate(values, axis=0)

    return values


def calculate_class_weight(y_train):
    # calculate class weights
    count = np.sum(y_train, axis=0)
    weight = np.sqrt(np.max(count)/count)
    class_weight = {}
    for i in range(y_train.shape[1]):
        class_weight[i] = weight[i]
    return class_weight
    

def compile_regression_model(model, learning_rate=0.001, mask_val=None, **kwargs):

    optimizer = optimizers(optimizer=optimizer, learning_rate=learning_rate, **kwargs)

    if mask:
        def masked_loss_function(y_true, y_pred):
            mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, mask_val)), dtype=tf.float32)
            return keras.losses.mean_squared_error(y_true*mask, y_pred*mask)
        loss = masked_loss_function
    else:
        loss = keras.losses.mean_squared_error

    model.compile(optimizer=optimizer, loss=loss)



def compile_classification_model(model, loss_type='binary', optimizer='adam', 
                                 learning_rate=0.001, monitor=['acc', 'auroc', 'aupr'], 
                                 label_smoothing=0.0, from_logits=False, **kwargs):

    optimizer = optimizers(optimizer=optimizer, learning_rate=learning_rate, **kwargs)

    metrics = []
    if 'acc' in monitor:    
        metrics.append('accuracy')
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='ROC', name='auroc'))
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='PR', name='aupr'))

    if loss_type == 'binary':
        loss = keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    elif loss_type == 'categorical':
        loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)



def optimizers(optimizer='adam', learning_rate=0.001, **kwargs):

    if optimizer == 'adam':
        if 'beta_1' in kwargs.keys():
            beta_1 = kwargs['beta_1']
        else:
            beta_1 = 0.9
        if 'beta_2' in kwargs.keys():
            beta_2 = kwargs['beta_2']
        else:
            beta_2 = 0.999
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    elif optimizer == 'sgd':
        if 'momentum' in kwargs.keys():
            momentum = kwargs['momentum']
        else:
            momentum = 0.0
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer



def clip_filters(W, threshold=0.5, pad=3):

    W_clipped = []
    for w in W:
        L,A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped



def meme_generate(W, output_file='meme.txt', prefix='filter'):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
        for i in range(L):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
        f.write('\n')

    f.close()



def match_hits_to_ground_truth(file_path, motifs, size=30):
    
    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')

    # loop through filters
    best_qvalues = np.ones(size)
    best_match = np.zeros(size)
    for name in np.unique(df['Query_ID'].as_matrix()):

        if name[:6] == 'filter':
            filter_index = int(name.split('r')[1])

            # get tomtom hits for filter
            subdf = df.loc[df['Query_ID'] == name]
            targets = subdf['Target_ID'].as_matrix()

            # loop through ground truth motifs
            for k, motif in enumerate(motifs): 

                # loop through variations of ground truth motif
                for motifid in motif: 

                    # check if there is a match
                    index = np.where((targets == motifid) ==  True)[0]
                    if len(index) > 0:
                        qvalue = subdf['q-value'].as_matrix()[index]

                        # check to see if better motif hit, if so, update
                        if best_qvalues[filter_index] > qvalue:
                            best_qvalues[filter_index] = qvalue
                            best_match[filter_index] = k 

    # get the minimum q-value for each motif
    min_qvalue = np.zeros(13)
    for i in range(13):
        index = np.where(best_match == i)[0]
        if len(index) > 0:
            min_qvalue[i] = np.min(best_qvalues[index])

    match_index = np.where(best_qvalues != 1)[0]
    match_fraction = len(match_index)/float(size)

    return best_qvalues, best_match, min_qvalue, match_fraction 



def activation_fn(activation):
    
    if activation == 'exp_relu':
        return exp_relu
    elif activation == 'shift_scale_tanh':
        return shift_scale_tanh
    elif activation == 'shift_scale_relu':
        return shift_scale_relu
    elif activation == 'shift_scale_sigmoid':
        return shift_scale_sigmoid
    elif activation == 'shift_relu':
        return shift_relu
    elif activation == 'shift_sigmoid':
        return shift_sigmoid
    elif activation == 'shift_tanh':
        return shift_tanh
    elif activation == 'scale_relu':
        return scale_relu
    elif activation == 'scale_sigmoid':
        return scale_sigmoid
    elif activation == 'scale_tanh':
        return scale_tanh
    elif activation == 'log_relu':
        return log_relu
    elif activation == 'log':
        return log
    elif activation == 'exp':
        return 'exponential'
    else:
        return activation
        
def exp_relu(x, beta=0.001):
    return K.relu(K.exp(.1*x)-1)

def log(x):
    return K.log(K.abs(x) + 1e-10)

def log_relu(x):
    return K.relu(K.log(K.abs(x) + 1e-10))

def shift_scale_tanh(x):
    return K.tanh(x-6.0)*500 + 500

def shift_scale_sigmoid(x):
    return K.sigmoid(x-8.0)*4000

def shift_scale_relu(x):
    return K.relu(K.pow(x-0.2, 3))

def shift_tanh(x):
    return K.tanh(x-6.0)

def shift_sigmoid(x):
    return K.sigmoid(x-8.0)

def shift_relu(x):
    return K.relu(x-0.2)

def scale_tanh(x):
    return K.tanh(x)*500 + 500

def scale_sigmoid(x):
    return K.sigmoid(x)*4000

def scale_relu(x):
    return K.relu((x)**3)



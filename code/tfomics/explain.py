import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1
import shap



def saliency(model, X, class_index=0, layer=-2, batch_size=256):
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]
    sess = K1.get_session()

    N = len(X)
    num_batches = int(np.floor(N/batch_size))

    attr_score = []
    for i in range(num_batches):
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[i*batch_size:(i+1)*batch_size]}))
    if num_batches*batch_size < N:
        attr_score.append(sess.run(saliency, {model.inputs[0]: X[num_batches*batch_size:N]}))

    return np.concatenate(attr_score, axis=0)


def mutagenesis(model, X, class_index=0, layer=-2):

    def generate_mutagenesis(X):
        L,A = X.shape 

        X_mut = []
        for l in range(L):
            for a in range(A):
                X_new = np.copy(X)
                X_new[l,:] = 0
                X_new[l,a] = 1
                X_mut.append(X_new)
        return np.array(X_mut)

    N, L, A = X.shape 
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

    attr_score = []
    for x in X:

        # get baseline wildtype score
        wt_score = intermediate.predict(np.expand_dims(x, axis=0))[:, class_index]

        # generate mutagenized sequences
        x_mut = generate_mutagenesis(x)
        
        # get predictions of mutagenized sequences
        predictions = intermediate.predict(x_mut)[:,class_index]

        # reshape mutagenesis predictiosn
        mut_score = np.zeros((L,A))
        k = 0
        for l in range(L):
            for a in range(A):
                mut_score[l,a] = predictions[k]
                k += 1
                
        attr_score.append(mut_score - wt_score)
    return np.array(attr_score)


def deepshap(model, X, class_index=0, layer=-2, num_background=10, reference='shuffle'):

    N, L, A = X.shape 
    if reference is not 'shuffle':
        num_background = 1
        
    # set of background sequences to take expectation over
    shap_values = []
    for j, x in enumerate(X):
        if np.mod(j, 50) == 0:
            print("%d out of %d"%(j,N))
        if reference == 'shuffle':
            background = []
            for i in range(num_background):
                shuffle = np.random.permutation(L)
                background.append(x[shuffle, :])
            background = np.array(background)
        else: 
            background = np.zeros([1,L,A])     

        x = np.expand_dims(x, axis=0)
        # calculate SHAPLEY values 
        background.shape
        e = shap.DeepExplainer(model, background)
        shap_values.append(e.shap_values(x)[0])

    attr_score = np.concatenate(shap_values, axis=0)
    return attr_score


 

def integrated_grad(model, X, class_index=0, layer=-2, num_background=10, num_steps=20, reference='shuffle'):

    def linear_path_sequences(x, num_background, num_steps, reference):
        def linear_interpolate(x, base, num_steps=20):
            x_interp = np.zeros(tuple([num_steps] +[i for i in x.shape]))
            for s in range(num_steps):
                x_interp[s] = base + (x - base)*(s*1.0/num_steps)
            return x_interp

        L, A = x.shape 
        seq = []
        for i in range(num_background):
            if reference == 'shuffle':
                shuffle = np.random.permutation(L)
                background = x[shuffle, :]
            else: 
                background = np.zeros(x.shape)        
            seq.append(linear_interpolate(x, background, num_steps))
        return np.concatenate(seq, axis=0)

    # setup op to get gradients from class-specific outputs to inputs
    saliency = K1.gradients(model.layers[layer].output[:,class_index], model.input)[0]

    # start session
    sess = K1.get_session()

    attr_score = []
    for x in X:
        # generate num_background reference sequences that follow linear path towards x in num_steps
        seq = linear_path_sequences(x, num_background, num_steps, reference)
       
        # average/"integrate" the saliencies along path -- average across different references
        attr_score.append([np.mean(sess.run(saliency, {model.inputs[0]: seq}), axis=0)])
    attr_score = np.concatenate(attr_score, axis=0)

    return attr_score


    
def attribution_score(model, X, method='saliency', norm='times_input', class_index=0,  layer=-2, **kwargs):

    N, L, A = X.shape 
    if method == 'saliency':
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size=256
        
        attr_score = saliency(model, X, class_index, layer, batch_size)

        
    elif method == 'mutagenesis':
        
        attr_score = mutagenesis(model, X, class_index, layer)
        
    elif method == 'deepshap':
        if 'num_background' in kwargs:
            num_background = kwargs['num_background']
        else:
            num_background = 5
        if 'reference' in kwargs:
            reference = kwargs['reference']
        else:
            reference = 'shuffle'
    
        attr_score = deepshap(model, X, class_index, num_background, reference)

        
    elif method == 'integrated_grad':
        if 'num_background' in kwargs:
            num_background = kwargs['num_background']
        else:
            num_background = 10
        if 'num_steps' in kwargs:
            num_steps = kwargs['num_steps']
        else:
            num_steps = 20
        if 'reference' in kwargs:
            reference = kwargs['reference']
        else:
            reference = 'shuffle'
        
        attr_score = integrated_grad(model, X, class_index, layer, num_background, num_steps, reference)

    if norm == 'l2norm':
        attr_score = np.sqrt(np.sum(np.squeeze(attr_score)**2, axis=2, keepdims=True) + 1e-10)
        attr_score =  X * np.matmul(attr_score, np.ones((1, X.shape[-1])))
        
    elif norm == 'times_input':
        attr_score *= X

    return attr_score


#-------------------------------------------------------------------------------------------------
# Plot conv filters
#-------------------------------------------------------------------------------------------------


def plot_filers(model, x_test, layer=3, threshold=0.5, window=20, num_cols=8, figsize=(30,5)):

    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    fmap = intermediate.predict(x_test)
    W = activation_pwm(fmap, x_test, threshold=threshold, window=window)

    num_filters = len(W)
    num_widths = int(np.ceil(num_filters/num_cols))

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    logos = []
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_widths, num_cols, n+1)
    
        # calculate sequence logo heights
        I = np.log2(4) + np.sum(w * np.log2(w+1e-10), axis=1, keepdims=True)
        logo = np.maximum(I*w, 1e-7)

        L, A = w.shape
        counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(L)))
        for a in range(A):
            for l in range(L):
                counts_df.iloc[l,a] = logo[l,a]

        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0,2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
    
        logos.append(logo)
        
    return fig, W, logo



def activation_pwm(fmap, X, threshold=0.5, window=20):

    # extract sequences with aligned activation
    window_left = int(window/2)
    window_right = window - window_left

    N,L,A = X.shape
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):

        # find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)

        if len(coords) > 1:
            x, y = coords

            # sort score
            index = np.argsort(fmap[x,y,filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            # make a sequence alignment centered about each activation (above threshold)
            seq_align = []
            for i in range(len(pos_index)):

                # determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                    seq = X[data_index[i], start_window:end_window, :]
                    seq_align.append(seq)

            # calculate position probability matrix
            if len(seq_align) > 1:#try:
                W.append(np.mean(seq_align, axis=0))
            else: 
                W.append(np.ones((window,4))/4)
        else:
            W.append(np.ones((window,4))/4)

    return np.array(W)


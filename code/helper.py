import os, sys
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from tensorflow import keras



def load_data(file_path, reverse_compliment=False):

    # load dataset
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.float32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.float32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])
        
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test



def load_synthetic_models(filepath, dataset='test'):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')
    if dataset == 'train':
        return np.array(trainmat['model_train']).astype(np.float32)
    elif dataset == 'valid':
        return np.array(trainmat['model_valid']).astype(np.float32)
    elif dataset == 'test':
        return np.array(trainmat['model_test']).astype(np.float32)


def load_basset_dataset(filepath, reverse_compliment=False):

    trainmat = h5py.File(filepath, 'r')

    x_train = np.array(trainmat['train_in']).astype(np.float32)
    y_train = np.array(trainmat['train_out']).astype(np.int32)
    x_valid = np.array(trainmat['valid_in']).astype(np.float32)
    y_valid = np.array(trainmat['valid_out']).astype(np.int32)
    x_test = np.array(trainmat['test_in']).astype(np.float32)
    y_test = np.array(trainmat['test_out']).astype(np.int32)

    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    x_test = np.squeeze(x_test)

    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])


    if reverse_compliment:
        x_train_rc = x_train[:,::-1,:][:,:,::-1]
        x_valid_rc = x_valid[:,::-1,:][:,:,::-1]
        x_test_rc = x_test[:,::-1,:][:,:,::-1]
        
        x_train = np.vstack([x_train, x_train_rc])
        x_valid = np.vstack([x_valid, x_valid_rc])
        x_test = np.vstack([x_test, x_test_rc])
        
        y_train = np.vstack([y_train, y_train])
        y_valid = np.vstack([y_valid, y_valid])
        y_test = np.vstack([y_test, y_test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test
    

def load_model(model_name, activation='relu', input_shape=200):

    if model_name == 'cnn-50':
        from model_zoo import cnn_model
        model = cnn_model.model([50, 2], activation, input_shape)

    elif model_name == 'cnn-2':
        from model_zoo import cnn_model
        model = cnn_model.model([2, 50], activation, input_shape)

    elif model_name == 'cnn-deep':
        from model_zoo import cnn_deep
        model = cnn_deep.model(activation, input_shape)

    elif model_name == 'cnn-local':
        from model_zoo import cnn_local
        model = cnn_local.model(activation)

    elif model_name == 'cnn-dist':
        from model_zoo import cnn_dist
        model = cnn_dist.model(activation)

    elif model_name == 'basset':
        from model_zoo import basset
        model = basset.model(activation)

    elif model_name == 'residualbind':
        from model_zoo import residualbind
        model = residualbind.model(activation)

    return model



def match_hits_to_ground_truth(file_path, motifs, size=32):
    
    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')
    
    # loop through filters
    best_qvalues = np.ones(size)
    best_match = np.zeros(size)
    correction = 0  
    for name in np.unique(df['Query_ID'][:-3].to_numpy()):
        filter_index = int(name.split('r')[1])

        # get tomtom hits for filter
        subdf = df.loc[df['Query_ID'] == name]
        targets = subdf['Target_ID'].to_numpy()

        # loop through ground truth motifs
        for k, motif in enumerate(motifs): 

            # loop through variations of ground truth motif
            for motifid in motif: 

                # check if there is a match
                index = np.where((targets == motifid) ==  True)[0]
                if len(index) > 0:
                    qvalue = subdf['q-value'].to_numpy()[index]

                    # check to see if better motif hit, if so, update
                    if best_qvalues[filter_index] > qvalue:
                        best_qvalues[filter_index] = qvalue
                        best_match[filter_index] = k 

        index = np.where((targets == 'MA0615.1') ==  True)[0]
        if len(index) > 0:
            if len(targets) == 1:
                correction += 1

    # get the minimum q-value for each motif
    num_motifs = len(motifs)
    min_qvalue = np.zeros(num_motifs)
    for i in range(num_motifs):
        index = np.where(best_match == i)[0]
        if len(index) > 0:
            min_qvalue[i] = np.min(best_qvalues[index])

    match_index = np.where(best_qvalues != 1)[0]
    if any(match_index):
        match_fraction = len(match_index)/float(size)
    else:
        match_fraction = 0
    
    num_matches = len(np.unique(df['Query_ID']))-3
    match_any = (num_matches - correction)/size

    return best_qvalues, best_match, min_qvalue, match_fraction, match_any



        
def interpretability_performance(X, score, X_model):

    score = np.sum(score, axis=2)
    pr_score = []
    roc_score = []
    for j, gs in enumerate(score):

        # calculate information of ground truth
        gt_info = np.log2(4) + np.sum(X_model[j]*np.log2(X_model[j]+1e-10),axis=0)

        # set label if information is greater than 0
        label = np.zeros(gt_info.shape)
        label[gt_info > 0.01] = 1

        # precision recall metric
        precision, recall, thresholds = precision_recall_curve(label, gs)
        pr_score.append(auc(recall, precision))

        # roc curve
        fpr, tpr, thresholds = roc_curve(label, gs)
        roc_score.append(auc(fpr, tpr))

    roc_score = np.array(roc_score)
    pr_score = np.array(pr_score)

    return roc_score, pr_score
    


def get_callbacks(monitor='val_auroc', patience=20, decay_patience=5, decay_factor=0.2):
    es_callback = keras.callbacks.EarlyStopping(monitor=monitor, 
                                                patience=patience, 
                                                verbose=1, 
                                                mode='max', 
                                                restore_best_weights=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, 
                                                  factor=decay_factor,
                                                  patience=decay_patience, 
                                                  min_lr=1e-7,
                                                  mode='max',
                                                  verbose=1) 

    return [es_callback, reduce_lr]



def compile_model(model):

    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', auroc, aupr])

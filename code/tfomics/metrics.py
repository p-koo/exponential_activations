import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score


def accuracy(label, prediction):
    num_labels = label.shape[1]
    metric = np.zeros((num_labels))
    for i in range(num_labels):
        metric[i] = accuracy_score(label[:,i], np.round(prediction[:,i]))
    return metric


def auroc(label, prediction):
    num_labels = label.shape[1]
    metric = np.zeros((num_labels))
    for i in range(num_labels):
        fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
        score = auc(fpr, tpr)
        metric[i]= score
    return metric


def aupr(label, prediction):
    num_labels = label.shape[1]
    metric = np.zeros((num_labels))
    for i in range(num_labels):
        precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
        score = auc(recall, precision)
        metric[i] = score
    return metric


def pearsonr(label, prediction, mask_value=None):
    num_labels = label.shape[1]
    corr = np.zeros((num_labels))
    for i in range(num_labels):
        if mask_value:
            index = np.where(label[:,i] != mask_value)[0]
            corr[i] = stats.pearsonr(label[index,i], prediction[index,i])[0]
        else:
            corr[i] = stats.pearsonr(label[:,i], prediction[:,i])[0]

    return corr


def rsquare(label, prediction):  
    num_labels = label.shape[1]
    metric = np.zeros((num_labels))
    slope = np.zeros((num_labels))
    for i in range(num_labels):
        y = label[:,i]
        X = prediction[:,i]
        m = np.dot(X,y)/np.dot(X, X)
        resid = y - m*X; 
        ym = y - np.mean(y); 
        rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
        metric[i] = rsqr2
        slope[i] = m
    return metric, slope



def calculate_metrics(label, prediction, objective):
    """calculate metrics for classification"""

    if (objective == "binary"):
        acc = accuracy(label, prediction)
        auc_roc = auroc(label, prediction)
        auc_pr = aupr(label, prediction)
        mean = [np.nanmean(acc), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(acc), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    elif objective == "categorical":
        acc = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
        auc_roc = auroc(label, prediction)
        auc_pr = aupr(label, prediction)
        mean = [np.nanmean(acc), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(acc), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    elif (objective == 'squared_error'):
        corr = pearsonr(label,prediction)
        rsqr, slope = rsquare(label, prediction)
        mean = [np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        std = [np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

    else:
        mean = 0
        std = 0

    return [mean, std]
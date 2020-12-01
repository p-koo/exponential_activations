import os, sys
from six.moves import cPickle
import numpy as np
import pandas as pd
import helper
from tfomics import utils

#------------------------------------------------------------------------------------------------
# JASPAR filter indices

arid3 = ['MA0151.1', 'MA0601.1', 'PB0001.1']
cebpb = ['MA0466.1', 'MA0466.2']
fosl1 = ['MA0477.1']
gabpa = ['MA0062.1', 'MA0062.2']
mafk = ['MA0496.1', 'MA0496.2']
max1 = ['MA0058.1', 'MA0058.2', 'MA0058.3']
mef2a = ['MA0052.1', 'MA0052.2', 'MA0052.3']
nfyb = ['MA0502.1', 'MA0060.1', 'MA0060.2']
sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3']
srf = ['MA0083.1', 'MA0083.2', 'MA0083.3']
stat1 = ['MA0137.1', 'MA0137.2', 'MA0137.3', 'MA0660.1', 'MA0773.1']
yy1 = ['MA0095.1', 'MA0095.2']


Gmeb1 = ['MA0615.1']

motifs = [[''],arid3, cebpb, fosl1, gabpa, mafk, max1, mef2a, nfyb, sp1, srf, stat1, yy1]
motifnames = [ '','arid3', 'cebpb', 'fosl1', 'gabpa', 'mafk', 'max', 'mef2a', 'nfyb', 'sp1', 'srf', 'stat1', 'yy1']

#----------------------------------------------------------------------------------------------------



num_trials = 10
model_name = 'cnn-deep'
activations = ['relu', 'exponential']

scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5]

results_path = utils.make_directory('../../results', 'exp_scale_sweep')
save_path = utils.make_directory(results_path, 'conv_filters')
size = 32

# save results to file
with open(os.path.join(results_path, 'filter_results.tsv'), 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'match JASPAR', 'match ground truth'))

    results = {}
    for scale in scales:
            trial_match_any = []
            trial_qvalue = []
            trial_match_fraction = []
            trial_coverage = []
            for trial in range(num_trials):
                try:
                    file_path = os.path.join(save_path, model_name+'_'+str(scale)+'_'+str(trial), 'tomtom.tsv')
                    best_qvalues, best_match, min_qvalue, match_fraction, match_any  = helper.match_hits_to_ground_truth(file_path, motifs, size)
		    # store results
                    trial_qvalue.append(min_qvalue)
                    trial_match_fraction.append(match_fraction)
                    trial_coverage.append((len(np.where(min_qvalue != 1)[0])-1)/12) # percentage of motifs that are covered
                    trial_match_any.append(match_any)
                except:
                    trial_qvalue.append(0)
                    trial_match_fraction.append(0)
                    trial_coverage.append(0) # percentage of motifs that are covered
                    trial_match_any.append(0)
                    
            f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(model_name+'_'+str(scale), 
                                                      np.mean(trial_match_any), 
                                                      np.std(trial_match_any),
                                                      np.mean(trial_match_fraction), 
                                                      np.std(trial_match_fraction) ) )
            results[scale] = {}
            results[scale]['match_fraction'] = np.array(trial_match_fraction)
            results[scale]['match_any'] = np.array(trial_match_any)
            
# pickle results
file_path = os.path.join(results_path, "intialization_sweep_filter_results.pickle")
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)






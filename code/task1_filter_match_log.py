import os, sys
from six.moves import cPickle
import numpy as np
import pandas as pd
from tensorflow import keras
import helper
from tfomics import utils
from model_zoo import cnn_deep_log

#------------------------------------------------------------------------------------------------

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
activations = ['log_relu', 'relu']
l2_norm = [True, False]
model_name = 'cnn-deep'

results_path = os.path.join('../results', 'task1')
save_path = os.path.join(results_path, 'conv_filters')
size = 32

print('Synthetic results')
print("%s\t%s\t%s"%('model name', 'match JASPAR', 'match ground truth') )

# save results to file
file_path = os.path.join(results_path, 'task1_filter_results_log.tsv')
with open(file_path, 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'match JASPAR', 'match ground truth'))

    results = {}
    for activation in ['log_relu', 'relu']:
        for l2_norm in [True, False]:
            trial_match_any = []
            trial_qvalue = []
            trial_match_fraction = []
            trial_coverage = []
            for trial in range(num_trials):
                keras.backend.clear_session()
                
                # load model
                model = cnn_deep_log.model(activation, l2_norm)

                base_name = model_name+'_'+activation

                if l2_norm:
                    base_name = base_name + '_l2'
                name = base_name + '_' + str(trial)
                print(name)

                # save path
                file_path = os.path.join(save_path, name, 'tomtom.tsv')
                best_qvalues, best_match, min_qvalue, match_fraction, match_any  = helper.match_hits_to_ground_truth(file_path, motifs, size)

                # store results
                trial_qvalue.append(min_qvalue)
                trial_match_fraction.append(match_fraction)
                trial_coverage.append((len(np.where(min_qvalue != 1)[0])-1)/12) # percentage of motifs that are covered
                df = pd.read_csv(os.path.join(file_path), delimiter='\t')

                num_matches = len(np.unique(df['Query_ID']))-3 # -3 is because new version of tomtom adds 3 lines of comments under Query_ID 
                trial_match_any.append(num_matches/size)


            f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(base_name, 
                                                      np.mean(trial_match_any), 
                                                      np.std(trial_match_any),
                                                      np.mean(trial_match_fraction), 
                                                      np.std(trial_match_fraction) ) )
            results[base_name] = {}
            results[base_name]['match_fraction'] = np.array(trial_match_fraction)
            results[base_name]['match_any'] = np.array(trial_match_any)
                
    # pickle results
    file_path = os.path.join(results_path, "task1_filter_results_log.pickle")
    with open(file_path, 'wb') as f:
        cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)






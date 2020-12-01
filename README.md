This repository contains datasets and scripts to reproduce results from "Improving representation learning of genomic sequence motifs in convolutional netowrks with exponential activations" by Peter K. Koo and Matthew Ploenzke.


#### Dependencies
* Python 3
* Dependencies: Tensorflow r1.15 (tested), matplotlib, numpy, scipy, sklearn, shap, logomaker, pandas, h5py
* meme suite (5.1.0) to run Tomtom


## Overview of the code

There are 6 main tasks and code is labeled accordingly. The description of each script is given below.

#### Task 1 -- Classifying TF binding from synthetic data
* task1_step1_train_models.py
* task1_step2_tomtom.sh
* task1_step3_filter_match.py
* task1_step4_plot_filter_match.ipynb
* task1_step5_plot_filters_with_hits.ipynb
* task1_step6_PWM_scan_comparison.ipynb


#### Task 2 -- Classifying TF binding from in vivo data
* task2_step1_train_models.py
* task2_step2_tomtom.sh
* task2_step3_filter_match.py
* task2_step4_plot_filter_match.ipynb
* task2_step5_plot_filters_with_hits.ipynb


#### Task 3 -- Classifying cis-regulatory codes from synthetic data
* task3_step1_train_model.py
* task3_step2_attribution_scores.py
* task3_step3_plot_attr_score_comparison.ipynb
* task3_step4_plot_attr_logo_comparison.ipynb


#### Task 4 -- Classifying chromatin accessibility for DNase-seq data from the Basset dataset
* task4_step1_train_model.py
* task4_step2_tomtom.sh
* task4_step3_plot_attr_score_comparison.ipynb

#### Task 5 -- Classifying TF binding for ZBED2 ChIP-seq data  
* task5_step1_train_model.py
* task5_step2_plot_attr_score_comparison.ipynb


#### Task 6 -- Classifying TF binding for IRF1 ChIP-seq data 
* task6_step1_train_model.py
* task6_step2_plot_attr_score_comparison.ipynb


#### Other
* helper.py
	* functions that are useful to run analysis
* tfomics
	* additional useful functions for NN models and interpretability
* model_zoo
	* CNN models used in this study

* generate_data
	* Notebooks used to generate data for Task 1, Task2, and Task3
	

## Control Experiments


#### Training stability analysis


#### Log analysis
* task1_log_analysis_step1_train.py
* task1_log_analysis_step2_tomtom.sh
* task1_log_analysis_step3_filter_match.py
* task1_log_analysis_step4_plot_filter_match.ipynb
* task3_train_model_log.py
* task3_attribution_scores_log.py
* task3_plot_attr_score_comparison_log.ipynb

#### Initialization analysis


#### Initialization sweep analysis


#### Exponential scale sweep analysis

#### Number of background sequences for attribution methods analysis

* task3_sweep_num_background.py
	* systematically try different number of reference sequences from 1 to 25 and store in num_background_sweep.pickle
* task3_plot_num_background_sweep.ipynb
	* plot results from task3_sweep_num_background.py


#### Misc



## Figures

#### Main Figures
- Fig. 1a: code/controls/plot_activations.ipynb
- Fig. 1b: code/task1_step4_plot_filters_with_hits.ipynb
- Fig. 1c: code/task1_step5_plot_filter_match.ipynb
- Fig. 1d: code/task2_step4_plot_filter_match.ipynb
- Fig. 2a-c: code/task3_step4_plot_attr_score_comparisons.ipynb
- Fig. 2d: code/task3_step5_plot_attr_logo_comparisons.ipynb
- Fig. 3a: code/task4_step3_plot_attr_score_comparison.ipynb
- Fig. 3b: code/task5_step2_plot_attr_score_comparison.ipynb
- Fig. 3c: code/task6_step2_plot_attr_score_comparison.ipynb

#### Extended Data Figures
- Extended Data Fig. 1a,b: code/task1_step5_plot_filter_match.ipynb
- Extended Data Fig. 1c,d: code/task1_step6_PWM_scan_comparison.ipynb
- Extended Data Fig. 1e: code/controls/task1_log_analysis_step4_plot_filter_match.ipynb
- Extended Data Fig. 2a,b: code/task3_step4_plot_attr_score_comparisons.ipynb 
- Extended Data Fig. 2c,d: code/task3_step5_plot_attr_logo_comparisons.ipynb

#### Supplemental Figures
- Supp Fig. 1: code/task1_step4_plot_filters_with_hits.ipynb
- Supp Fig. 2: code/task1_step4_plot_filters_with_hits.ipynb
- Supp Fig. 3: code/controls/visualize_training_history.ipynb
- Supp Fig. 4: code/controls/gradient_analysis_step2_visualize.ipynb
- Supp Fig. 5a: code/controls/initialization_step4_visualize_filter_match.ipynb
- Supp Fig. 5b: code/controls/initialization_sweep_step4_visualize_filter_match.ipynb
- Supp Fig. 6: code/task2_step5_plot_filters_with_hits.ipynb
- Supp Fig. 7: code/controls/plot_activations.ipynb
- Supp Fig. 8: code/task2_step4_plot_filter_match.ipynb
- Supp Fig. 9: code/controls/exp_scale_sweep_step4_visualize.ipynb
- Supp Fig. 10: code/controls/plot_activation_comparison.ipynb
- Supp Fig. 11a: code/task3_step4_plot_attr_score_comparisons.ipynb
- Supp Fig. 11b: code/controls/task3_log_analysis_step3_plot_attr_score_comparisons.ipynb
- Supp Fig. 12: code/task4_step3_plot_attr_score_comparison.ipynb
- Supp Fig. 13: code/task5_step2_plot_attr_score_comparison.ipynb
- Supp Fig. 14: code/task6_step2_plot_attr_score_comparison.ipynb
- Supp Fig. 15: code/controls/threshold_sweep_step4_visualize_filter_match.ipynb
- Supp Fig. 16: code/controls/background_analysis_step2_plot_num_background_sweep.ipynb





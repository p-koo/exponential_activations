#!/bin/bash

dirpath="../../results/task1/conv_filters_threshold_sweep"
for ACTIVATION in relu exponential
do
    for THRESH in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
    do
        for TRIAL in {0..9} 
	do
	    tomtom -evalue -thresh 0.1 -o $dirpath/cnn-deep_${ACTIVATION}_${THRESH}_${TRIAL} $dirpath/cnn-deep_${ACTIVATION}_${TRIAL}_${THRESH}_${TRIAL}.meme ../../data/JASPAR_CORE_2016_vertebrates.meme
	done
    done
done


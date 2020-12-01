#!/bin/bash

dirpath="../../results/initialization_sweep/conv_filters"


for ACTIVATION in relu exp  
do
	for INITIALIZATION in 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 3 4 5
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/cnn-deep_${ACTIVATION}_${INITIALIZATION}_${TRIAL} $dirpath/cnn-deep_${ACTIVATION}_${INITIALIZATION}_${TRIAL}.meme ../../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done

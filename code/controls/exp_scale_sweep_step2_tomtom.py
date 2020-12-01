#!/bin/bash

dirpath="../../results/exp_scale_sweep/conv_filters"

for MODEL in cnn-deep 
do
	for SCALE in 0.001 0.005 0.01 0.05 0.1 0.5 1 2 3 4 5
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/${MODEL}_${SCALE}_${TRIAL} $dirpath/${MODEL}_${SCALE}_${TRIAL}.meme ../../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done

#!/bin/bash

dirpath="../results/task1/conv_filters"

for MODEL in cnn-deep
do
	for ACTIVATION in log_relu relu log_relu_l2 relu_l2
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/${MODEL}_${ACTIVATION}_${TRIAL} $dirpath/${MODEL}_${ACTIVATION}_${TRIAL}.meme ../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done








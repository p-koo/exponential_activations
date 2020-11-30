#!/bin/bash

dirpath="../results/task2/conv_filters"
for MODEL in cnn-2 cnn-50 cnn-deep 
do
	for ACTIVATION in exponential relu sigmoid tanh softplus linear elu  
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/${MODEL}_${ACTIVATION}_${TRIAL} $dirpath/${MODEL}_${ACTIVATION}_${TRIAL}.meme ../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done

for MODEL in cnn-2 cnn-50 cnn-deep 
do
	for ACTIVATION in shift_scale_relu shift_scale_tanh shift_scale_sigmoid exp_relu shift_relu scale_relu shift_tanh scale_tanh shift_sigmoid scale_sigmoid 
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/${MODEL}_${ACTIVATION}_${TRIAL} $dirpath/${MODEL}_${ACTIVATION}_${TRIAL}.meme ../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done

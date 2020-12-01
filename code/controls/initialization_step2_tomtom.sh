#!/bin/bash

dirpath="../../results/initialization/conv_filters"

for ACTIVATION in relu exp  
do
	for INITIALIZATION in glorot_normal glorot_uniform he_normal he_uniform lecun_normal lecun_uniform
	do
		for TRIAL in {0..9} 
		do
		    tomtom -evalue -thresh 0.1 -o $dirpath/cnn-deep_${ACTIVATION}_${INITIALIZATION}_${TRIAL} $dirpath/cnn-deep_${ACTIVATION}_${INITIALIZATION}_${TRIAL}.meme ../../data/JASPAR_CORE_2016_vertebrates.meme
		done
	done
done
